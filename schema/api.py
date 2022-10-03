import traceback
import inspect
import os, io, sys
import re
import tensorflow as tf
import numpy as np
import itertools
from collections import defaultdict, OrderedDict
from . import predicates as pr
from . import generators as ge
from . import base
from .redirect import stderr_redirector
from .base import Kind, kname, kpfx, kind
from . import fgraph
from . import flib
from .error import *
from .fgraph import PredNode as P, GenNode as G, FuncNode as F

"""
Every API call will mutate the Generative Graph and the Predicate Graph
logically in tandem.  It should maintain the invariants:

1. Every value set produced by the Generative Graph should be valid as judged
   by the Predicate Graph

2. The set of values produced by the Generative Graph is "complete" in the
sense that it it explores every combination of dtypes, ranks, and option
arguments.  It need not explore every possible setting of dimensions or tensor
contents.

Both finished graphs must have exactly one node corresponding to each input
argument, either the arg_name:tensor node (for tensor arguments) or the
arg_name:arg node for non-tensors.
"""
class _TestResult(object):
    """
    One schema test result.  Holds all sufficient information about the test.
    Provides convenient reporting and classification functions
    """
    def __init__(self, op, _id, config):
        self.op = op
        self.id = _id
        self.config = config

        statuses = self.config[Kind.EXPECT_STATUS]
        err = list(s for s in statuses if s != Success)
        if len(err) > 1:
            stat = None
        else:
            stat = err[0] if len(err) > 0 else Success
        self.config['EXPECT_STATUS'] = stat

    def add_result(self):
        self.config['OPCHECK_STATUS'] = self.op.input_status
        self.config['FRAMEWORK_STATUS'] = self.op.framework_status
        ex_stat_cls = self.config['EXPECT_STATUS']
        op_stat = self.config['OPCHECK_STATUS']
        fr_stat = self.config['FRAMEWORK_STATUS']
        if not isinstance(op_stat, ex_stat_cls):
            cat = 'FAIL'

        op_neg = isinstance(op_stat, Success)
        fr_neg = isinstance(fr_stat, Success)
        if op_neg:
            cat = 'TN' if fr_neg else 'FN'
        else:
            cat = 'FP' if fr_neg else 'TP'
        self.config['CATEGORY'] = cat

    def make_args(self):
        # create arguments 
        d = { p: self.config.get(k, None) for p,k in self.op.params.items() }
        arg_dict = {}
        for name, val in d.items():
            kname = self.op.params[name]
            if kname is not None and kind(kname) == Kind.DATA_TENSOR:
                val = ge.from_stub(val)
            arg_dict[name] = val
        return arg_dict

    def stat_keys(self):
        # return an ordered set of keys
        keys = ['ID', 'CATEGORY', 'EXPECT_STATUS', 'OPCHECK_STATUS',
                'FRAMEWORK_STATUS', Kind.DATA_FORMAT, Kind.RANKS, Kind.DTYPES]
        for n,kn in self.op.params.items():
            if kind(kn) in (Kind.DATA_TENSOR, Kind.UNCHECKED,
                    Kind.DATA_FORMAT):
                continue
            if kn not in self.config:
                raise SchemaError(
                    f'Node \'{kn}\' is listed as a parameter node, but '
                    f'not found in the test config object')
            keys.append(kn)
        return keys

    def stats(self):
        # retrieve sufficient statistics to determine the kind of test
        stats = []
        keys = self.stat_keys()
        for k in keys:
            item = self.config.get(k, None)
            if k == 'ID':
                v = str(self.id)
            if k == 'EXPECT_STATUS':
                v = item.__name__
            elif k == 'OPCHECK_STATUS':
                v = item.__class__.__name__
            elif k == 'FRAMEWORK_STATUS':
                if isinstance(item, FrameworkError):
                    item = item.ex
                v = item.__class__.__name__
            elif k == Kind.DTYPES:
                v = ','.join(d.name for d in item.values())
            elif k == Kind.RANKS:
                v = ','.join(str(r) for r in item.values())
            elif k == Kind.DATA_FORMAT:
                v = '<none>' if item is None else item
            else:
                v = str(item)
            stats.append(v)
        return stats

    def stats_report(self):
        stats = self.stats()

    def run(self):
        # run the op and store the results
        string_err = io.BytesIO()
        arg_dict = self.make_args()
        try:
            with stderr_redirector(string_err):
                self.op.wrapped_op(**arg_dict)
        except OpCheckInternalError as e:
            print(string_err.getvalue().decode('UTF-8'))
            raise e
        except BaseException as e:
            # this should always be from TensorFlow
            trace = inspect.trace()
            for frame in reversed(trace):
                mod = inspect.getmodule(frame[0])
                if mod is not None:
                    break
            modname = mod.__name__
            # print(modname, flush=True)
            if modname.split('.')[0] == 'tensorflow':
                # print('exception inside tensorflow:')
                # traceback.print_stack()
                pass
            else:
                assert False, 'exception outside tf should not be possible'
                print('exception outside of tensorflow')
                traceback.print_stack()
                raise e
        self.add_result()

    def report(self):
        """
        Generate a human-readable report
        """
        cat = self.config['CATEGORY']
        op_stat = self.config['OPCHECK_STATUS']
        fr_stat = self.config['FRAMEWORK_STATUS']
        op_msg = op_stat.message(self.op)
        fr_msg = fr_stat.message(self.op)
        if cat == 'TP':
            return f'Framework\n{fr_msg}\nOpCheck\n{op_msg}\n'
        elif cat == 'TN':
            return ''
        elif cat == 'FP':
            return f'OpCheck\n{op_msg}\n'
        else:
            return f'Framework\n{fr_msg}\n'

        
class SchemaApi(object):
    def __init__(self, op_path):
        self.op_path = op_path

        # idx => description
        self.index = {}
        # arg_name => kname
        self.params = None # will be an ordered dict
        self.layout_param = None 
        self.gen_graph = None
        self.input_pred_graph = None
        self.return_pred_graph = None
        self.rank_candidates = base.RankCandidates(self)
        self.rank_cons = [] 
        self.dtype_cons = base.DTypeConstraints()
        self.dims_graph = base.CompDimsGraph()
        self.gen_indices = base.GenIndices()
        # self.gd_dims = ge.IndexDimsGD(self, 1e5, 2e6)
        self.comp_dims_templates = {} # idx => PredNode with TemplateFunc
        self.num_returns = 0
        self.num_layouts = 1

        # error status
        self.input_status = None
        self.framework_status = None

        # call time values
        self.arguments = {}
        self.returns = []

    def _init_schema(self, func_sig, init_schema_func):
        # edges to create for the pred graph
        self.pending_pred_edges = {} # kname -> [parent_kname, parent_kname, ...]
        self.func_sig = func_sig
        pars = func_sig.parameters.keys()
        self.params = OrderedDict({ k: None for k in pars })
        self._init_pred_graph()
        self._init_gen_graph()
        init_schema_func(self)
        self._add_pred_graph()
        self._add_gen_graph()
        self._validate_constraints()

    def _prepare_call(self, *args, **kwargs):
        """Call during the framework call phase"""
        bind = self.func_sig.bind(*args, **kwargs)
        bind.apply_defaults()
        self.arguments = bind.arguments
        self.returns.clear()
        self.input_status = None
        self.framework_status = None

    def _check_args(self):
        """
        The main function to check all input arguments for all constraints
        registered on the schema
        """
        error = fgraph.pred_graph_evaluate(self.input_pred_graph.values())
        self.input_status = Success() if error is None else error

    def _check_return(self, op_return):
        """
        Check the return tensors' shapes and types against those predicted by
        the framework
        """
        if not isinstance(self.input_status, Success):
            return

        if not isinstance(op_return, (list, tuple)):
            op_return = (op_return,)
        self.returns = list(op_return)
        error = fgraph.pred_graph_evaluate(self.return_pred_graph.values())
        if error is not None:
            raise SchemaError(error.msg(self))

    def _ranks_sigs_format(self):
        """
        Generates all valid combinations of ranks_map, sigs_map, and
        data_format as ordered tuples
        """
        names = [Kind.RANKS, Kind.SIG_MAP, Kind.DATA_FORMAT]
        ngen = (self.gen_graph[nn] for nn in names if nn in self.gen_graph)
        nodes = tuple(ngen)
        tups = fgraph.all_values(*nodes)
        if len(nodes) == 2:
            return [ (*tup, None) for tup in tups ]
        else:
            return tups

    def _shape_key_order(self, shape_keys):
        # order the shape keys in argument order
        arg_order = list(self.params.keys())
        arg_order.extend(str(i) for i in range(self.num_returns))

        def key_fun(shape_key):
            pfx = shape_key.split('.')[0]
            return arg_order.index(pfx)
        key_order = sorted(shape_keys, key=key_fun)
        return key_order

    def _sig_inventory(self):
        """
        Generate the formatted signature inventory for this op
        """
        geometry = self._ranks_sigs_format()
        _, sig_map, _ = geometry[0]
        args = [ *sig_map ]
        if self.layout_param is not None:
            args.append(self.layout_param)
        arg_order = self._shape_key_order(args)

        rows = [arg_order]
        for rank_map, sig_map, cand_format in geometry:
            row = []
            for arg in arg_order:
                if arg == self.layout_param:
                    row.append(cand_format)
                else:
                    sig = sig_map[arg]
                    inst = ''.join(s * rank_map[s] for s in sig)
                    row.append(inst)
            rows.append(row)

        table, _ = tabulate(rows, '  ', left_align=True)
        return table

    def _index_inventory(self):
        """
        Generate a formatted report of the indices with their rank constraints
        """
        rows = []
        rows.append(['Index', 'Description'])
        rows.extend([ix,desc] for ix,desc in self.index.items())
        tab, _ = tabulate(rows, '  ', left_align=True)
        return tab

    def _rank_error_report(self, shape_map, data_format, report):
        """
        This report is generated when the framework op arguments are such that
        no consistent set of index ranks could be inferred.  The report
        consists of a set of possible ways to fix the inputs.

        Each item is a table, followed by one or more text suggestions on how
        to fix the inputs.  The table has the following rows:

        arguments
        shapes
        interpretation
        errors

        'arguments' shows formatted argument names highlighting the relevant
        aspect of the argument.  (see api.py:_shape_header)

        DATA_TENSOR:  {arg_name}.shape
        SHAPE_TENSOR: {arg_name}.numpy()
        SHAPE_LIST: {arg_name}
        SHAPE_INT: {arg_name}

        'shapes' shows the actual submitted values of the argument aspect.
        All of these represent shapes to be applied to OpCheck indices.

        'interpretation' shows, for each component of a shape, the one-letter
        OpCheck index name which is inferred in this item.

        'errors' is an ASCII representation highlighting where the error
        occurred.
        
        The list of plain-text suggestions provides one way to fix the errors.
        It is necessarily a guess about what the user might have intended.
        ...

        """
        # need to augment this with another map of other argument values
        args = [ *shape_map ]
        if self.layout_param is not None:
            args.append(self.layout_param)
        arg_order = self._shape_key_order(args)
        cand_reports = []

        leader_col = [ 'arguments', 'shapes', 'interpretation', 'errors' ]

        for cand in report:
            # the sub_table is a map of arg_name => rows
            # the rows are: actual shape, signature instantiation, highlight
            # carats
            sub_table = {} 
            for n, shape in shape_map.items():
                shape_rank = len(shape)
                shape_row = [str(sz) for sz in shape]
                sub_table[n] = [shape_row]

                # populate the signature instantiation row
                sig = cand.sigs[n]
                inst_row = []
                for s in sig:
                    r = cand.ranks[s]
                    if r == 1:
                        inst_row.append(s)
                    else:
                        inst_row.extend(f'{s}{i+1}' for i in range(r))
                inst_rank = len(inst_row)
                sub_table[n].append(inst_row)

                # populate the highlight carat row
                pos = cand.highlight[n]
                num_components = max(shape_rank, inst_rank)
                highlight_row = []
                for c in range(num_components):
                    if c in pos:
                        w1 = len(shape_row[c]) if c < shape_rank else 0
                        w2 = len(inst_row[c]) if c < inst_rank else 0
                        carat = '^' * max(w1, w2)
                    else:
                        carat = ''
                    highlight_row.append(carat)
                sub_table[n].append(highlight_row)

            # format the sub-tables
            columns = {} 
            for name, tab in sub_table.items():
                tab = sub_table[name]
                shape_rows, _ = tabulate(tab, ' ', left_align=True)
                hdr = self._shape_header(name)
                col = [hdr, *shape_rows]
                columns[name] = col

            if self.layout_param is not None:
                if (
                        (data_format == cand.format) or
                        (data_format is None and cand.format is None)
                        ):
                    hl = ''
                else:
                    hl = '^' * max(len(data_format), len(cand.format))
                columns[self.layout_param] = [
                        self.layout_param, 
                        data_format, 
                        cand.format,
                        hl
                        ]

            col_array = [leader_col] + [ columns[name] for name in arg_order ]
            main_table = np.array(col_array).transpose().tolist()
            main_rows, _ = tabulate(main_table, '   ', True)
            table = '\n'.join(main_rows)
            suggs = '\n'.join(cand.suggestions)
            cand_reports.append(f'{table}\n{suggs}\n')

        full_report = '\n'.join(cand_reports)
        return full_report

    def _index_usage_phrase(self, idx, component_usages, ranks):
        def phrase_join(names):
            qnames = [f'\'{n}\'' for n in names]
            phrase = ', '.join(qnames[:-1])
            sep = '' if phrase == '' else ' and '
            return sep.join((phrase, qnames[-1]))

        phrases = []
        r = ranks[idx]
        for c, usage in enumerate(component_usages):
            if len(usage) == 1:
                continue
            sep = ''
            main_phrase = ''
            for sz, arg_list in usage.items():
                phrase = phrase_join(arg_list)
                main_phrase += sep + f'size {sz} in {phrase}'
                sep = ', '
            idxc = idx if r == 1 else f'{idx}{c+1}'
            msg = f'Index \'{idxc}\' ({self.index[idx]}) has {main_phrase}.'
            phrases.append(msg)
        return '\n'.join(phrases)

    # compute the highlight mask from the component usage maps
    @staticmethod
    def _highlight_mask(ranks, sigs, shapes, idx_usage):
        highlight = defaultdict(list)
        for arg, sig in sigs.items():
            shape = shapes[arg]
            for idx in sig:
                comp = idx_usage.get(idx, None)
                if comp is None:
                    mask = [False] * ranks[idx]
                else:
                    mask = [ (len(c) != 1) for c in comp ]
                highlight[arg].extend(mask)
        return dict(highlight)

    def _index_diagram(self, highlight_map, ranks, sigs, shapes):
        arg_order = [ n for n in self.params.keys() if n in sigs.keys() ]
        dims = { n: [shp] for n, shp in shapes.items() }
        table_data = {} # arg => [shape, inst, highlight]
                        # shape is e.g.:     [15, 3, 10, 5]
                        # inst is e.g.       ['b', 'i1', 'i2', 'k']
                        # highlight is e.g.: ['', '', '^^', '']

        for arg, sig in sigs.items():
            shape = shapes[arg]
            mask = highlight_map[arg]
            table_data[arg] = [shape]
            inst = []
            highlight = []
            for idx in sig:
                if ranks[idx] == 1:
                    inst.append(idx)
                else:
                    inst.extend(f'{idx}{i+1}' for i in range(ranks[idx]))

            z = zip(mask, inst)
            hl = ['^' * len(i) if m else '' for m, i in z]
            highlight.extend(hl)
            table_data[arg].append(inst)
            table_data[arg].append(highlight)
        
        columns = []
        for arg, rows in table_data.items():
            fmt, _ = tabulate(rows, ' ', left_align=True)
            col = [arg, *fmt]
            columns.append(col)
        table = np.array(columns).transpose().tolist()
        fmt, _ = tabulate(table, '   ', left_align=True)
        return '\n'.join(fmt)

    def _index_usage_error(self, idx_usage, ranks, sigs, shapes):
        """
        Generate the message for an IndexUsageError.
        {idx_usage} is: idx => [ (dim => [arg1, ...]),
                                 (dim => [arg1, ...]),
                                 ...
                               ]
        {sigs} is: arg => sig
        {shapes} is: arg => shape
        """
        highlight_map = self._highlight_mask(ranks, sigs, shapes, idx_usage)
        diagram = self._index_diagram(highlight_map, ranks, sigs, shapes)

        index_msgs = []
        for idx, comp in idx_usage.items():
            phrase = self._index_usage_phrase(idx, comp, ranks)
            index_msgs.append(phrase)

        text = '\n'.join(index_msgs)
        return diagram + '\n' + text

    def _index_constraint_error(self, text, index_highlight, ranks, sigs,
            shapes):
        # compute the arg => mask from idx => mask
        arg_highlight = defaultdict(list)
        for arg, sig in sigs.items():
            for s in sig:
                mask = index_highlight.get(s, [False] * ranks[s])
                arg_highlight[arg].extend(mask)

        diagram = self._index_diagram(arg_highlight, ranks, sigs, shapes)
        return diagram + '\n' + text

    def _dtype_excluded_report(self, ten_names, ten_dtypes, rank_map, layout):
        """
        Generates an error report to the user indicating that The combination
        of {ten_names} having {ten_dtypes}, with the particular setting of
        index ranks given in {rank_map} and for the given layout is disallowed.

        If rank_map is empty, the dtype combination is disallowed for every
        rank combination, and ranks will not be mentioned in the report.

        If layout is None, the dtype combination is disallowed for every
        layout, and layout will not be mentioned in the report
        """
        header = ['configuration', 'value']
        table = [ header ]
        for n, d in zip(ten_names, ten_dtypes):
            row = [ f'tensor \'{n}\' dtype', d.name ]
            table.append(row)
        for idx, rank in rank_map.items():
            row = [ f'\'{self.index[idx]}\' # dims', rank ]
            table.append(row)
        if layout is not None:
            data_format = self._get_arg(self.layout_param)
            row = [ f'\'{self.layout_param}\'', data_format ]
            table.append(row)

        rows, _ = tabulate(table, '  ', left_align=[False, True])
        main = ['Received unavailable configuration:']
        main.extend(rows)
        main.append('')
        inst = (f'Use opcheck.list_configs(\'{self.op_path}\') to see all '
                f'available configurations')
        main.append(inst)
        msg = '\n'.join(main)
        return msg

    def _validate_schema(self, out_dir):
        """
        Uses the gen_graph to produce a list of (<target_status>, <arg_dict>)
        pairs for the op.  <target_status> is the correct type of SchemaStatus.
        Runs the wrapped op on the <arg_dict> and collects the actual
        SchemaStatus and any possible exception from the framework.

        It is a SchemaError if the target status type does not equal the
        returned status type.

        For each result, the following four outcomes are possible, where
        'positive' denotes detection of error, and 'negative' the absence of an
        error.

        <opcheck_status>    <framework>      <category>
        Success             Success          true negative
        Success             FrameworkError   false negative
        not Success         Success          false positive
        not Success         FrameworkError   true positive
        """
        if not os.path.exists(out_dir):
            raise RuntimeError(
                f'{type(self).__qualname__}: Could not open output path '
                f'\'{out_dir}\' for report generation')

        # list of successful arg_dict settings
        tests = []
        test_id = 1
        for config in fgraph.gen_graph_iterate(self.gen_graph.values()):
            t = _TestResult(self, test_id, config)
            if t.config['EXPECT_STATUS'] is None:
                continue
            tests.append(t)
            test_id += 1

        stats_fname = f'{self.op_path}.stats.txt'
        with open(os.path.join(out_dir, stats_fname), 'w') as fh:
            for t in tests:
                t.run()
                row = t.stats()
                line = '\t'.join(r for r in row)
                print(line, file=fh)

        stats = { 'TP': [], 'FP': [], 'TN': [], 'FN': [], 'FAIL': [] }
        for t in tests:
            cat = t.config['CATEGORY']
            stats[cat].append(t)

        for cat in ('FP', 'FN', 'FAIL', 'TP'):
            file_name = f'{self.op_path}.{cat.lower()}.txt'
            with open(os.path.join(out_dir, file_name), 'w') as fh: 
                for t in stats[cat]:
                    row = t.stats()
                    line = '\t'.join(r for r in row)
                    print(line, file=fh)
                    print(t.report(), file=fh)

        print('Summary')
        for cat, res in stats.items():
            print(f'{cat}: {len(res)}')

    def _passed(self):
        return (
                isinstance(self.input_status, Success) and
                isinstance(self.framework_status, Success)
                )

    def _shape_header(self, shape_arg):
        # translate a plain argument name  
        try:
            name, idx = shape_arg.split('.')
        except:
            name, idx = shape_arg, None

        if name not in self.params:
            raise RuntimeError(
                f'{type(self).__qualname__}: name \'{name}\' not a named '
                f'parameter.')
        k = kind(self.params[name])
        if k == Kind.DATA_TENSOR:
            sfx = 'shape'
        elif k == Kind.SHAPE_TENSOR:
            sfx = 'numpy()'
        elif k == Kind.SHAPE_TENSOR2D:
            if idx is not None:
                name = f'{name}[{idx},:]'
            sfx = 'numpy()'
        else:
            sfx = ''
        return name if sfx == '' else f'{name}.{sfx}'

    def _call_string(self, arg_dict):
        """
        Summarize the call arguments
        """
        reps = {}
        for name, arg_val in arg_dict.items():
            hdr = self._shape_header(name)
            kname = self.params[name]
            k = kind(kname) if kname is not None else None
            if k == Kind.DATA_TENSOR:
                val = arg_val.shape.as_list()
            elif k == Kind.SHAPE_TENSOR:
                val = arg_val.numpy().tolist()
            elif k == Kind.SHAPE_TENSOR2D:
                val = arg_val.numpy().tolist()
            else:
                val = arg_val
            reps[name] = f'{hdr}={repr(val)}'
        call_string = ', '.join(reps[n] for n in self.params.keys() if
                n in reps) 
        return call_string

    def _report(self):
        msg = self.input_status.message(self)
        print(msg, file=sys.stderr)

    # for debugging
    def _print_pred_graph(self):
        for n in self.input_pred_graph:
            print(f'{n.name}: {n.cached_val}')

    def _get_arg(self, arg_name):
        """Retrieve the value of {arg_name} argument at call-time."""
        if arg_name not in self.params:
            raise SchemaError(
                f'\'{arg_name}\' not a known parameter. '
                f'Known parameters are: {self.params.keys()}')
        return self.arguments[arg_name]

    def _set_arg_kname(self, arg_name, arg_kname):
        """
        Expect {arg_name} to have type {arg_kname}
        """
        if arg_name not in self.params:
            raise SchemaError(
                f'{type(self).__name__}: Attempted to add {arg_name} parameter '
                f'but it is not found in the framework op parameters. '
                f'Valid parameters are: {self.params.keys()}')
        
        if self.params[arg_name] is not None:
            raise SchemaError(
                f'{type(self).__name__}: Attempting to add {arg_name} as kname '
                f'{arg_kname} to the registry, but it is already registered '
                f'as type {self.params[arg_name].__name__}')
        self.params[arg_name] = arg_kname

    def _check_sig(self, signature, name):
        if any(s not in self.index.keys() for s in signature):
            raise SchemaError(
                f'Signature "{signature}" associated with \'{name}\' '
                f'contains one or more unregistered indices. '
                f'Current known indices are: '
                f"{','.join(self.index.keys())}"
                f'Call OpSchema.add_index with the missing index.')

    def _get_return(self, idx):
        try:
            return self.returns[idx]
        except IndexError:
            raise SchemaError(
                f'{type(self).__qualname__}({idx}) called but only '
                f'{len(self.returns)} returns')
    
    @staticmethod
    def _resolve_arg_names(caller, graph_cls, arg_names):
        """
        Find the unique arg_kname for each arg_name.  arg_name is either a
        simple string prefix, or a kname.  If a prefix,
        search for a unique kname in the registry with that prefix.  If a
        kname, take the name as-is
        """
        # find the unique arg_kname for each arg_name.  it is an error if it is
        # not unique
        knames = []
        all_knames = graph_cls.registry.keys()
        for arg_name in arg_names:
            if arg_name in all_knames:
                candidates = [arg_name]
            else:
                candidates = [ k for k in all_knames if kpfx(k) == arg_name ]
            if len(candidates) != 1:
                raise SchemaError(
                    f'{type(caller).__qualname__}: argument name \'{arg_name}\''
                    f' must identify a node with \':arg\' suffix or be a fully '
                    f'qualified kname')
            knames.append(candidates[0])
        return tuple(knames)

    def _init_pred_graph(self):
        P.clear_registry()
        P.add_node(Kind.SCHEMA, pr.Closure((True, self)))
        P.add_node(Kind.SHAPE_MAP, pr.ShapeMap())
        P.add_node(Kind.SIG_MAP, pr.SigMap())
        ranks_obj = pr.IndexRanks(self, self.rank_candidates, self.rank_cons)
        P.add_node(Kind.RANKS, ranks_obj, Kind.SHAPE_MAP)
        dtypes_obj = pr.ValidDTypes(self.dtype_cons)
        P.add_node(Kind.DTYPES, dtypes_obj, Kind.RANKS)
        P.add_node(Kind.IDIMS, pr.IndexDimsUsage(), Kind.RANKS, Kind.SIG_MAP,
                Kind.SHAPE_MAP)

    def _init_gen_graph(self):
        G.clear_registry()
        F.clear_registry()
        G.add_node(Kind.SIG_MAP, ge.SigMap())
        shape_gobj = ge.SignatureShapes(self.dims_graph, self.rank_candidates,
                self.gen_indices, 1e6)
        G.add_node(Kind.ARG_SHAPES_RANKS, shape_gobj, Kind.SIG_MAP) 
        G.add_node(Kind.RANKS, ge.TupleElement(0), Kind.ARG_SHAPES_RANKS)
        G.add_node(Kind.ARG_SHAPES_STATUS, ge.TupleElement(1),
                Kind.ARG_SHAPES_RANKS)
        G.add_node(Kind.ARG_SHAPES, ge.TupleElement(1), Kind.ARG_SHAPES_STATUS)
        # ranks_gobj = ge.Ranks(self, self.rank_candidates)
        # G.add_node(Kind.RANKS_STATUS, ranks_gobj, Kind.SIG_MAP) 
        # G.add_node(Kind.RANKS, ge.SecondInPair(), Kind.RANKS_STATUS)
        dtypes_obj = ge.DTypes(self.dtype_cons)
        G.add_node(Kind.DTYPES_STATUS, dtypes_obj, Kind.RANKS) 
        G.add_node(Kind.DTYPES, ge.TupleElement(1), Kind.DTYPES_STATUS)
        # G.add_node(Kind.GD_DIMS_STATUS, self.gd_dims, Kind.RANKS, Kind.SIG_MAP)
        # G.add_node(Kind.GD_DIMS, ge.TupleElement(1), Kind.GD_DIMS_STATUS)
        G.add_node(Kind.EXPECT_STATUS, ge.StatusAggregator(), 
                Kind.DTYPES_STATUS, Kind.ARG_SHAPES_STATUS)

    def _add_pred_graph(self):
        # add single-index dims nodes that are not already added
        for idx in self.index.keys():
            kn = kname(idx, Kind.SINGLE_DIMS)
            node = P.maybe_get_node(kn)
            if node is None:
                P.add_node(kn, pr.SingleIndexDims(idx), Kind.IDIMS) 

        for kn, pknodes in self.pending_pred_edges.items():
            node = P.get_node(kn)
            for pkn in pknodes:
                pnode = P.get_node(pkn)
                node.append_parent(pnode)
            
        pred_nodes = dict(P.registry)
        def is_return(node):
            ret_knames = (Kind.RETURN_TENSOR, Kind.VALID_RETURN)
            return base.kind(node.name) in ret_knames 

        self.input_pred_graph = { name: nd for name, nd in pred_nodes.items()
                if not is_return(nd) }

        self.return_pred_graph = { name: nd for name, nd in pred_nodes.items()
                if is_return(nd) }

    def _add_gen_graph(self):
        self.dims_graph.finalize()
        self.gen_graph = dict(G.registry)

    def _validate_constraints(self):
        """
        Called at the end of schema construction to check that schema
        constraints are self-consistent 
        """
        # Ensure that every tensor has exactly one dtype constraint
        for arg_name, arg_kname in self.params.items():
            if arg_kname is None:
                raise SchemaError(
                    f'{type(self).__qualname__}: \'{self.op_path}\' argument '
                    f'\'{arg_name}\' has no registered kname.  '
                    f'To ignore it, call arg_unchecked(\'{arg_name}\')')

    # ============ PUBLIC API ====================
    def add_index(self, idx, description, min_rank=None, max_rank=None):
        """
        Add index {idx} with {description} to the schema.  {idx} must be a
        single letter and can be referred to in later signatures.

        If {min_rank} is provided, declare that the rank of this index be >=
        this value.

        If {max_rank} is provided, declare that the rank of this index be <=
        this value.
        """
        self.index[idx] = description
        if min_rank is not None or max_rank is not None:
            self.rank_candidates.add_rank_limits(idx, min_rank, max_rank)

    def arg_unchecked(self, arg_name):
        """
        Declare {arg_name} to be an argument unchecked by OpCheck 
        """
        self._set_arg_kname(arg_name, Kind.UNCHECKED)

    def computed_index(self, comp_index, comp_func, template_func,
            input_indexes, min_val, *extra_args):
        """
        Registers {comp_func} to compute the dimensions of {comp_index}.
        Registers {template_func} which produces a text string explaining how
        the index is computed.

        Adds an index predicate to ensure all components of the computed index
        are >= {min_val}

        The following calls are made:

        comp_func(*index_dims, *extra_vals)
        (index_dims are the resolved dimensions of {input_indexes})
        (extra_vals are the runtime values of {extra_args})

        If a downstream constraint (registered with add_index_predicate) fails,
        then OpCheck makes these calls:

        template_func(*index_desc, *extra_vals)
        template_func(*index_dims, *extra_vals)
        (index_desc are the snake_cased descriptions of {input_indexes})

        for any computed indices that are used directly or indirectly by the
        predicate (ancestor indices).  The output strings are then assembled to
        create an explanatory error message.
        """
        if not all(idx in self.index for idx in input_indexes):
            raise SchemaError(
                f'{type(self).__qualname__}: In schema \'{self.op_path}\'.\n'
                f'Indices string \'{input_indexes}\' contains unregistered '
                f'indices.\nRegistered indices are: {list(self.index.keys())}\n'
                )

        extra_knames = self._resolve_arg_names(self, P, extra_args)
        cdims_kname = kname(comp_index, Kind.SINGLE_DIMS)
        comp_pobj = pr.ComputedDims(comp_index, comp_func, len(input_indexes))
        P.add_node(cdims_kname, comp_pobj)
        tem_kname = kname(comp_index, Kind.COMP_DIMS_TEM)
        temp_pobj = pr.TemplateFunc(template_func, comp_index, self)
        tem_pnode = P.add_node(tem_kname, temp_pobj, cdims_kname)
        self.comp_dims_templates[comp_index] = tem_pnode

        pknames = [ kname(idx, Kind.SINGLE_DIMS) for idx in input_indexes ]
        pknames.extend(extra_knames)

        self.pending_pred_edges[tem_kname] = pknames
        self.pending_pred_edges[cdims_kname] = pknames

        if comp_index in self.dims_graph.computed_indexes():
            raise SchemaError(
                f'{type(self).__qualname__}: index \'{comp_index}\' has '
                f'already been registered as a computed index')
        if comp_index in self.dims_graph.input_indexes():
            raise SchemaError(
                f'{type(self).__qualname__}: index \'{comp_index}\' has '
                f'already been used as an input index for some computed '
                f'index.  Calls to computed_index must be in dependency order')

        self.dims_graph.add_comp_index(comp_index, comp_func, input_indexes,
                *extra_knames)
        dims_gnode = G.get_node(Kind.ARG_SHAPES_RANKS)
        for kn in extra_knames:
            dims_gnode.maybe_append_parent(kn)

        # add a predicate to ensure the computed index is >= some minimum value
        bounds_pobj = flib.PredAbove(min_val)
        self.add_index_predicate(f'{comp_index} >= {min_val}', bounds_pobj,
                comp_index)  

    def equate_ranks(self, target_index, source_index):
        """
        Declare that the rank of {target_index} be equal to {source_index}.
        It is required that all indices in {source_index} appear in some
        signature in a limit_ranks call.
        """
        if target_index not in self.index:
            raise SchemaError(
                f'{type(self).__qualname__}: target_index \'{target_index}\''
                f'is not a registered index')
        if (self.rank_candidates.index_limited(target_index) or
                self.rank_candidates.index_equated(target_index)):
            raise SchemaError(
                f'{type(self).__qualname__}: target index \'{target_index}\''
                f'is already registered as constrained')
        if not self.rank_candidates.index_limited(source_index):
            raise SchemaError(
                f'{type(self).__qualname__}: source index \'{source_index}\''
                f'is not constrained with limit_ranks')
        self.rank_candidates.equate_ranks(target_index, source_index)

    def limit_ranks(self, sig, min_val, max_val):
        """
        Declare that the rank of {sig} be in [{min_val}, {max_val}]
        """
        self._check_sig(sig, 'rank limits')
        self.rank_candidates.add_rank_limits(sig, min_val, max_val)

    @staticmethod
    def _dtype_expr(type_expr):
        exprs = {
                'int': [8, 16, 32, 64],
                'uint': [8, 16, 32, 64],
                'float': [16, 32, 64],
                'qint': [8, 16, 32],
                'bfloat': [16],
                'bool': [''],
                'complex': [64, 128]
                }

        types = [ ', '.join(f'{k}{v}' for v in exprs[k]) for k in exprs ]
        type_str = '\n'.join(t for t in types)
        err_msg = SchemaError(
            f'Received invalid dtype expression \'{type_expr}\'.\n'
            f'dtype expression must match the pattern:\n'
            f'([a-z]+)(8|16|32|64|128)?([\+\-])?\n'
            f'The first capture is the data type and must be one of: '
            f'int, uint, float, qint, bfloat, bool, complex\n'
            f'The second capture is the size.  It is optional. '
            f'The third is an optional \'+\' or \'-\''
            f'The list of valid constructed types are:\n'
            f'{type_str}\n'
            )

        # expect format to be {pfx}{q}[+-]*
        ma = re.match('([a-z]+)(8|16|32|64|128)?([\+\-])?', type_expr)
        if ma is None:
            raise err
        pfx, q, rng = ma.groups()
        if q is None:
            ids = [ f'{pfx}{sz}' for sz in exprs[pfx] ]
        else:
            if rng is None:
                ids = [ type_expr ]
            elif rng == '+':
                ids = [ f'{pfx}{sz}' for sz in exprs[pfx] if sz >= int(q) ]
            else:
                ids = [ f'{pfx}{sz}' for sz in exprs[pfx] if sz <= int(q) ]
        try:
            dtypes = [ tf.dtypes.as_dtype(i) for i in ids ]
        except TypeError:
            raise err
        return dtypes

    def _is_data_tensor(self, name):
        return (name in self.params and 
                kind(self.params[name]) == Kind.DATA_TENSOR)

    def valid_dtypes(self, tensor_name, type_list):
        """
        Declare that {tensor_name} can have any of the dtype strings in
        {type_list}.  Names in {type_list} fit the pattern:

        ([a-z]+)(8|16|32|64|128)?([\+\-])?
        The first capture is the data type and must be one of:
        int, uint, float, qint, bfloat, bool, complex
        The second capture is the size.  It is optional.
        The third is an optional '+' or '-'.
        If the second is not present, the third must not be present.

        A prefix alone denotes all sizes of that data type are valid.
        A prefix with a quantity and no '+' or '-' specifies that single dtype.
        If a '+' is included, it means, that size and larger.
        If a '-' is included, it means that size and smaller.

        Can only be called once for a given {tensor_name}
        """
        if not self._is_data_tensor(tensor_name):
            raise SchemaError(
                f'{type(self).__qualname__}: Parameter \'{tensor_name}\' is '
                f'not registered as a tensor')
        if self.dtype_cons.has_valid_dtypes(tensor_name):
            raise SchemaError(
                f'{self.__qualname__}: Tensor \'{tensor_name}\' is already '
                f'registered with valid dtypes')

        dtypes = [ t for ex in type_list for t in self._dtype_expr(ex) ]
        self.dtype_cons.add_valid(tensor_name, dtypes)

    def equate_dtypes(self, trg_tensor, src_tensor):
        """
        Declare that {trg_tensor} have the same dtype as {src_tensor}.
        Both must be tensors declared with arg_tensor.
        Can only be called once for a given {trg_tensor}
        """
        if not (self._is_data_tensor(src_tensor) and
                self._is_data_tensor(trg_tensor)):
            raise SchemaError(
                f'{type(self).__name__}: Can only be called on two tensors. '
                f'Parameters \'{src_tensor}\' and \'{trg_tensor}\' are not '
                f'both tensors.')
        prev_equate_src = self.dtype_cons.get_equate_source(trg_tensor)
        if prev_equate_src is not None:
            raise SchemaError(
                f'{type(self).__name__}: Tensor \'{trg_tensor}\' has already '
                f'been assigned dtype equated source tensor '
                f'\'{prev_equate_src}\' from a previous call to equate_dtypes')
        self.dtype_cons.add_equiv(trg_tensor, src_tensor)

    def exclude_dtypes(self, fields, *exclude):
        """
        This API call allows to mark any combinations of tensor dtypes, index
        ranks and layouts as excluded.  It is useful in cases where such
        combinations are not implemented by the framework.

        Register {exclude} combinations of tensor dtypes, index ranks and
        layout to be excluded.

        {fields} is a comma-separated list of fields, with any of:
        - data tensor names registered with arg_tensor
        - one-letter index names registered with add_index
        - the constant Kind.LAYOUT

        Each member of {exclude} contains a tuple corresponding to {fields}.
        - data tensor fields have a dtype string, such as 'int32'
        - one-letter indexes have an integer specifying a rank of that index
        - the Kind.LAYOUT field has an integer in [0, num_layouts), as defined
          by the call to arg_layout.

        A value of None for any field indicates a wild-card, meaning 'exclude
        all values'.

        In the rare case a tensor is a one-letter name and conflicts with an
        index name, the first occurrence is interpreted as a tensor name, and
        the second as an index name.
        """
        tensors = []
        indexes = []
        has_layout = False
        
        for f in fields:
            if self._is_data_tensor(f) and f not in tensors:
                tensors.append(f)
            elif f in self.index:
                indexes.append(f)
            elif f == Kind.LAYOUT:
                has_layout = True
            else:
                raise SchemaError(
                    f'{type(self).__qualname__}: Item \'{f}\' in fields was '
                    f'not a data tensor registered with arg_tensor or '
                    f'one letter index name registered with add_index, or '
                    f'the constant Kind.LAYOUT')

        num_fields = len(fields)
        num_tensors = len(tensors)
        num_indexes = len(indexes)
        for ex in exclude:
            if len(ex) != num_fields:
                raise SchemaError(
                    f'{type(self).__qualname__}: Each item in \'exclude\' '
                    f'must have the same number of elements as \'fields\'.\n'
                    f'Found {len(fields)} fields but exclude item '
                    f'{ex} has {len(ex)} fields.')
            it = iter(ex)
            dtype_bases = []
            ranks = {} 
            for i in range(num_tensors):
                dtype_expr = next(it)
                dtype_list = self._dtype_expr(dtype_expr)
                dtype_bases.append(dtype_list)
            for p, idx in enumerate(indexes):
                rank = next(it)
                if rank is None:
                    continue
                elif isinstance(rank, int):
                    ranks[idx] = rank
                else:
                    raise SchemaError(
                        f'{type(self).__qualname__}: Got invalid rank item in '
                        f'exclusion tuple \'{ex}\'. Item {num_tensors + p - 1}'
                        f' was \'{rank}\' but should be None or an integer.')
            if has_layout:
                layout = next(it)
                if layout is None:
                    pass
                elif (isinstance(layout, int) and layout in
                        range(self.num_layouts)):
                    pass
                else:
                    raise SchemaError(
                        f'{type(self).__qualname__}: Got invalid layout '
                        f'\'{layout}\'.  Must be None or an integer in '
                        f'[0, {self.num_layouts})')
            for dtypes in itertools.product(*dtype_bases):
                self.dtype_cons.add_excluded(tensors, dtypes, ranks, layout)

    def arg_int(self, arg_name, lo=None, hi=None):
        """
        Declare {arg_name} to be an integer that can take on values in a range.
        If {lo} is None, it is sys.maxint
        If {hi} is None, it is -sys.maxint-1 
        """
        arg_kname = base.kname(arg_name, Kind.ARG)
        self._set_arg_kname(arg_name, arg_kname)
        pred_obj = pr.ArgInt(arg_name, lo, hi)
        gen_obj = ge.Int(lo, hi)
        P.add_node(arg_kname, pred_obj, Kind.SCHEMA)
        G.add_node(arg_kname, gen_obj)

    def _arg_pseudo(self, pseudo_kname, pred_func, gen_func, arg_name):
        """
        Creates a pseudo-input argument called {pseudo_name}, which is used to
        break a dependency cycle in nodes of the Generation Graph or Predicate
        graph.

        {gen_func}() generates all legal values for the pseudo argument during
        the schema validation phase.

        {pred_func}(arg_val) returns a derived value which represents the
        pseudo-input's value.  It is as if that value were provided directly to
        the framework operation.
        """
        pfunc_obj = pr.ArgFunc(arg_name, pred_func)
        P.add_node(pseudo_kname, pfunc_obj, Kind.SCHEMA) 
        G.add_node(pseudo_kname, gen_func)

    def _arg_func(self, arg_name, arg_kname, pred_func, gen_func,
            *func_arg_names):
        """
        Register {arg_name} to be validated with the call
        pred_func(arg_val, *func_arg_vals).  (Note the first argument is the
        supplied schema)

        For testing, generate values with a call to gen_func(*func_arg_vals).

        pred_func must return tuples of either:
        True, <value>
        False, SchemaError
        
        Produces PredNode and GenNode of arg_kname
        """
        knames = self._resolve_arg_names(self, P, func_arg_names)
        self._set_arg_kname(arg_name, arg_kname)
        pfunc_obj = pr.ArgFunc(arg_name, pred_func)
        P.add_node(arg_kname, pfunc_obj, Kind.SCHEMA, *knames)
        G.add_node(arg_kname, gen_func, *knames)

    def arg_option(self, arg_name, options):
        """
        Expect {arg_name} to take on one of the values in {options}
        """
        try:
            iter(options)
        except TypeError:
            raise SchemaError(
                f'{type(self).__qualname__}: \'options\' argument must be '
                f'iterable.  Got {type(options)}')
        def options_gen():
            return options

        arg_kname = base.kname(arg_name, Kind.ARG)
        self._set_arg_kname(arg_name, arg_kname)
        G.add_node(arg_kname, options_gen)
        def options_pred(arg_val):
            if arg_val in options:
                return True, arg_val
            else:
                return False, NonOptionError(arg_name, arg_val) 
        pred_obj = pr.ArgFunc(arg_name, options_pred)
        P.add_node(arg_kname, pred_obj, Kind.SCHEMA)

    def arg_layout(self, arg_name, layouts, rank_idx):
        """
        Declares {arg_name} to control layout-dependent signatures for tensors. 
        {layouts} is an array, where each element is a map of: rank => code
        The rank of {rank_idx} determines which layout is mapped.
        """
        # define the pseudo-arg
        self.num_layouts = len(layouts)
        self.layout_param = arg_name
        pseudo_gen = ge.Layout(layouts)
        pseudo_pred = pr.ArgLayout(arg_name, layouts)
        self._arg_pseudo(Kind.LAYOUT, pseudo_pred, pseudo_gen, arg_name)

        # define the real arg 
        arg_pred = pr.ArgDataFormat(arg_name, layouts, rank_idx)
        arg_gen = ge.DataFormat(layouts, rank_idx)
        self._arg_func(arg_name, Kind.DATA_FORMAT, arg_pred, arg_gen,
                Kind.RANKS, Kind.LAYOUT)

        layout_gnode = G.get_node(Kind.LAYOUT)
        dtypes_gnode = G.get_node(Kind.DTYPES_STATUS)
        dtypes_gnode.append_parent(layout_gnode)

        ranks_pnode = P.get_node(Kind.RANKS)
        layout_pnode = P.get_node(Kind.LAYOUT)
        ranks_pnode.append_parent(layout_pnode)

        dtypes_pnode = P.get_node(Kind.DTYPES)
        dtypes_pnode.append_parent(layout_pnode)

    def _check_sigs_layout(self, arg_name, *sigs):
        if len(sigs) != 1 and len(sigs) != self.num_layouts:
            raise SchemaError(
                f'{type(self).__qualname__}: registering \'{arg_name}\' '
                f'there are {self.num_layouts} '
                f'layouts (as established by the call to \'arg_layout\') but '
                f'{len(sigs)} elements of \'*sigs\' argument.')

    def _arg_shape_func(self, arg_name, sigs, _type, arg_kind, pred_obj,
            gen_obj):
        """
        Backend function for arg_shape_* API functions 
        {gen_obj} is a n object from generators.py which expects the GD_DIMS
        node, and a SIG node.  It produces the current shape of that signature,
        as interpreted by the object.

        {pred_obj} 
        """
        self._check_sigs_layout(arg_name, *sigs)

        arg_kname = kname(arg_name, arg_kind)
        self._set_arg_kname(arg_name, arg_kname)

        shape_pobj = pred_obj

        shp_kname = kname(arg_name, Kind.SHAPE)
        sig_kname = kname(arg_name, Kind.SIG)

        if len(sigs) == 1:
            sig_pobj = pr.Closure((True, sigs[0]))
            sig_gobj = ge.Closure([sigs[0]])
            layout = tuple()
        else:
            sig_pobj = pr.LayoutOption(arg_name, sigs)
            sig_gobj = ge.LayoutOption(sigs) 
            layout = (Kind.LAYOUT,)

        arg_pobj = pr.ArgType(arg_name, _type)
        P.add_node(arg_kname, arg_pobj, Kind.SCHEMA)

        shape_pnode = P.add_node(shp_kname, shape_pobj, arg_kname)
        sig_pnode = P.add_node(sig_kname, sig_pobj, *layout)
        sig_gnode = G.add_node(sig_kname, sig_gobj, *layout)

        G.add_node(arg_kname, gen_obj, Kind.ARG_SHAPES)

        sig_map_gnode = G.get_node(Kind.SIG_MAP)
        sig_map_gnode.append_parent(sig_gnode)

        shapemap_pnode = P.get_node(Kind.SHAPE_MAP)
        shapemap_pnode.append_parent(shape_pnode)

        sigmap_pnode = P.get_node(Kind.SIG_MAP)
        sigmap_pnode.append_parent(sig_pnode)

        cons = base.ShapeRankConstraint(arg_name, arg_kind)
        self.rank_cons.append(cons)

    def arg_tensor(self, arg_name, *sigs):
        """
        Register {arg_name} as a tensor.  

        sigs are all strings of signatures.  If len(sigs) == 1, then it
        specifies a static signature regardless of whether 'arg_layout' was
        called.  If len(sigs) > 1, then arg_layout is required to be called
        before this call.
        """
        pred_obj = pr.tensor_shape
        gen_obj = ge.TensorStub(arg_name)
        self._arg_shape_func(arg_name, sigs, tf.Tensor, Kind.DATA_TENSOR,
                pred_obj, gen_obj)
        arg_kname = kname(arg_name, Kind.DATA_TENSOR)
        gnode = G.get_node(arg_kname)
        gnode.maybe_append_parent(Kind.DTYPES)

        # dtypes
        dtype_kname = kname(arg_name, Kind.DTYPE)
        dtype_pnode = P.add_node(dtype_kname, pr.dtype, arg_kname)
        dtypes_pnode = P.get_node(Kind.DTYPES)
        dtypes_pnode.append_parent(dtype_pnode)

    def arg_shape_list(self, arg_name, *sigs):
        """
        Register {arg_name} as an integer list parameter which defines the
        shape of a signature.  
        """
        # Creates nodes:
        # arg_name:arg (int list)
        # arg_name:shape (int list, the same value as :arg)
        # arg_name:sig (str, the associated signature)
        pred_obj = pr.ShapeList(arg_name)
        gen_obj = ge.ShapeList(arg_name)
        self._arg_shape_func(arg_name, sigs, list, Kind.SHAPE_LIST, pred_obj,
                gen_obj)

    def arg_shape_int(self, arg_name, index):
        """
        Register {arg_name} as an integer parameter which defines the shape of
        an index.  The shape will be the broadcasted value of the argument if
        the index has rank greater than 1.

        This is the only arg_shape_* API function which does not define the
        rank of the index. 
        """
        # TODO: currently uses arg_shape_func, which assumes the arg defines
        # the rank.  In this case, it only defines the shape as the integer 
        # value broadcasted {rank} times.  But, the rank is not determined from
        # this input
        pred_obj = pr.ShapeInt(arg_name)
        gen_obj = ge.ShapeInt(arg_name)
        self._arg_shape_func(arg_name, (index,), int, Kind.SHAPE_INT, pred_obj,
                gen_obj) 

    def arg_shape_tensor(self, arg_name, *sigs):
        """
        Register {arg_name} as a 1D integer tensor whose elements define the
        shape of a signature.  
        """
        # Creates nodes:
        # arg_name:arg (tensor)
        # arg_name:shape (int list, the contents of the tensor)
        # arg_name:sig (str, the associated signature)
        # (no dtype node is created)
        pred_obj = pr.ShapeTensor(arg_name)
        gen_obj = ge.ShapeTensor(arg_name)
        self._arg_shape_func(arg_name, sigs, tf.Tensor, Kind.SHAPE_TENSOR,
                pred_obj, gen_obj)

    def arg_shape_tensor2d(self, arg_name, *sigs):
        """
        Register {arg_name} as a 2D integer tensor 'ten' defining the shape of
        sigs.  

        In the single layout case, sigs[i] are strings, and ten[d,i]
        defines dims(sigs[i])[d].  

        In the multiple layout case, sigs[i][l] is the i'th signature for
        layout l, and ten[d,i] defines dims(sigs[i][l])[d]

        Examples:

        Single layout case:

        ten = [ [1,2], [3,4], [5,6] ]
        sigs = ('b', 'e')
        
        defines 
        dims('b') := [1,3,5]
        dims('e') := [2,4,6]

        Multiple layout case:

        ten = [ [1,2], [3,4], [5,6] ]
        sigs = (('b', 'e'), ('e', 'b'))

        defines:
        layout 0: dims('b') = [1,3,5], dims('e') = [2,4,6] 
        layout 1: dims('b') = [2,4,6], dims('b') = [1,3,5]
        """
        # Creates nodes:
        # arg_name:arg (the 2D tensor)
        # arg_name.i:shape (the i'th shape)
        # arg_name.i:sig (the i'th signature)
        arg_kname = kname(arg_name, Kind.SHAPE_TENSOR2D)
        self._set_arg_kname(arg_name, arg_kname)
        arg_pobj = pr.ArgType(arg_name, tf.Tensor)
        arg_gobj = ge.ShapeTensor2D(arg_name, len(sigs))
        P.add_node(arg_kname, arg_pobj, Kind.SCHEMA)
        ten_gnode = G.add_node(arg_kname, arg_gobj, Kind.ARG_SHAPES)

        sigmap_gnode = G.get_node(Kind.SIG_MAP)
        shapemap_pnode = P.get_node(Kind.SHAPE_MAP)
        sigmap_pnode = P.get_node(Kind.SIG_MAP)

        for i, sig in enumerate(sigs):
            prefix = f'{arg_name}.{i}'
            sig_kname = kname(prefix, Kind.SIG)
            shp_kname = kname(prefix, Kind.SHAPE)
            shp_pobj = pr.ShapeTensorSlice(arg_name, i)
            cons = base.SliceRankConstraint(arg_name, i)
            self.rank_cons.append(cons)

            if isinstance(sig, str):
                sig_gobj = ge.Closure(list(sig))
                sig_pobj = pr.Closure((True, sig))
                sig_pnode = P.add_node(sig_kname, sig_pobj)
                sig_gnode = G.add_node(sig_kname, sig_gobj) 
            else:
                sig_pobj = pr.LayoutOption(arg_name, list(sig))
                sig_gobj = ge.LayoutOption(list(sig))
                sig_pnode = P.add_node(sig_kname, sig_pobj, Kind.LAYOUT)
                sig_gnode = G.add_node(sig_kname, sig_gobj, Kind.LAYOUT) 

            shp_pnode = P.add_node(shp_kname, shp_pobj, arg_kname)
            # ten_gnode.append_parent(sig_gnode)

            sigmap_gnode.append_parent(sig_gnode)
            shapemap_pnode.append_parent(shp_pnode)
            sigmap_pnode.append_parent(sig_pnode)

    def arg_rank(self, arg_name, sig):
        """
        Register {arg_name} to be an integer argument which defines the rank of
        {sig}
        """
        arg_kname = base.kname(arg_name, Kind.ARG)
        cons_name = f'rank({sig}) == \'{arg_name}\''
        cons = base.IntRankConstraint(cons_name, arg_kname, sig)
        self.rank_cons.append(cons)
        self._set_arg_kname(arg_name, arg_kname)
        P.add_node(arg_kname, pr.ArgInt(arg_name, 0, None), Kind.SCHEMA) 

        G.add_node(arg_kname, ge.Rank(sig), Kind.RANKS)
        rank_node = P.get_node(Kind.RANKS)
        rank_node.maybe_append_parent(arg_kname)

    def rank_dims_constraint(self, constraint_name, get_dims, rank_sig,
            dims_index, shape_arg):
        """
        Creates a constraint called {constraint_name} with the logic:
        RANK(rank_sig) == get_dims(shape_arg).

        Creates a generated index dimension:
        DIMS(dims_index) <- RANK(rank_sig)

        get_dims(*args) must return the quantity equal to DIMS(dims_index).
        Since this quantity is not directly available during the rank inference
        phase, it must use other means to derive the quantity.
        """
        cons = base.DimRankConstraint(constraint_name, rank_sig, shape_arg,
                get_dims, dims_index)
        self.rank_cons.append(cons)
        shape_karg = self.params[shape_arg]
        rank_node = P.get_node(Kind.RANKS)
        rank_node.maybe_append_parent(shape_karg)

        # 'sum' simply sums up the individual ranks of indices in rank_sig 
        dims_kname = kname(dims_index, Kind.GEN_DIMS)
        def gen_single_index(ranks_list):
            val = sum(ranks_list)
            return [([val],)]

        gobj = ge.Dims(gen_single_index, dims_index, rank_sig, tuple())
        gen_dims_gnode = G.add_node(dims_kname, gobj, Kind.RANKS)
        gd_dims_gnode = G.get_node(Kind.ARG_SHAPES_RANKS)
        gd_dims_gnode.append_parent(gen_dims_gnode)
        self.dims_graph.maybe_add_input_index(dims_index)

    def add_index_predicate(self, pred_name, status_func, indices):
        """
        Registers {status_func} with the schema to be used as an additional
        predicate for {indexes} dimensions.

        {pred_name} is a name given to this custom predicate.  It may be used
        in error messages.

        Called as status_func(*index_shapes), where index_shapes are the
        resolved shapes of each index, in order, in {indices}.  They are
        provided as numpy arrays.

        status_func must return an instance of SchemaStatus

        Custom status functions are found in the flib module.
        """
        pobj = pr.IndexDimsConstraint(pred_name, status_func)
        dims_kname = kname(pred_name, Kind.NONE)
        dims_pnode = P.add_node(dims_kname, pobj, Kind.RANKS, Kind.SIG_MAP, 
                Kind.SHAPE_MAP, Kind.SCHEMA)

        # cache the list of single index dims to add parents later
        pknames = [ kname(idx, Kind.SINGLE_DIMS) for idx in indices ]
        self.pending_pred_edges[dims_kname] = pknames

    def add_index_generator(self, output_indices, gen_func, input_indices, 
            *gen_args):
        """
        Registers {gen_func} with the schema to be used to generate
        dimension combinations for {output_indices}, which is a string
        consisting of individual index one-letter codes.

        It is called as gen_func(input_ranks, *gen_args) and returns a list of
        shape tuples.  The shapes in each shape tuple correspond with the
        indices in output_indices.

        input_ranks are the resolved ranks of input_indices

        Custom generator function objects are found in the flib module.
        """
        self.gen_indices.add_generator(gen_func, output_indices, input_indices,
                gen_args)
        # dims_kname = kname(output_indices, Kind.GEN_DIMS)
        # gobj = ge.Dims(gen_func, output_indices, input_indices, gen_args)
        # gen_dims_gnode = G.add_node(dims_kname, gobj, Kind.RANKS)
        # gd_dims_gnode = G.get_node(Kind.ARG_SHAPES_RANKS)
        # gd_dims_gnode.append_parent(gen_dims_gnode)

        # for idx in output_indices:
         #    self.dims_graph.maybe_add_input_index(idx)

    def return_tensor(self, *sigs):
        """
        Append a return tensor to the list of expected return tensors.

        *sigs may contain either one element, or {num_layout} elements.  If one
        element, it defines the static signature for the return tensor.  If
        multiple, they are defined by the provided layout as declared in
        'arg_layout'
        """
        index = self.num_returns
        self._check_sigs_layout(f'return[{index}]', *sigs)
        ret_name = str(index)
        ret_kname = kname(ret_name, Kind.RETURN_TENSOR)
        valid_return = kname(ret_name, Kind.VALID_RETURN)
        pshape_kname = kname(ret_name, Kind.PSHAPE)
        sig_kname = kname(ret_name, Kind.SIG)
        if len(sigs) == 1:
            sig_pobj = pr.Closure((True, sigs[0]))
            sig_gobj = ge.Closure([sigs[0]])
            layout = tuple()
        else:
            sig_pobj = pr.LayoutOption(str(index), sigs)
            sig_gobj = ge.LayoutOption(sigs) 
            layout = (Kind.LAYOUT,)

        rten_pobj = pr.GetReturnTensor(index)
        rvalid_pobj = pr.ValidReturnShape(index)
        P.add_node(ret_kname, rten_pobj, Kind.SCHEMA)
        P.add_node(sig_kname, sig_pobj, *layout)  
        P.add_node(pshape_kname, pr.predicted_shape, sig_kname)

        idx_knames = [kname(idx, Kind.SINGLE_DIMS) for idx in self.index.keys()]
        self.pending_pred_edges[pshape_kname] = idx_knames 

        P.add_node(valid_return, rvalid_pobj, ret_kname, pshape_kname) 
        sig_gnode = G.add_node(sig_kname, sig_gobj, *layout) 
        self.num_returns += 1

        sig_map_gnode = G.get_node(Kind.SIG_MAP)
        sig_map_gnode.append_parent(sig_gnode)

