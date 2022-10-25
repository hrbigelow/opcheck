import traceback
import inspect
import sys
import re
import tensorflow as tf
import numpy as np
import itertools
from collections import defaultdict
from . import predicates as pr
from . import generators as ge
from . import base
from . import fgraph
from . import flib
from .error import *
from .fgraph import PredNode as P, GenNode as G, FuncNode as F
from .base import GenMode, ShapeKind, GenKind

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


class SchemaApi(object):
    def __init__(self, op_path):
        self.op_path = op_path
        self.index = {} # idx => description

        # sidx => tidx.  RANK(sidx) is determined from RANK(tidx)
        # indices in which sidx == tidx are called 'primary'
        self.equiv_index = {} 

        # mode for running the gen_graph
        self.generation_mode = None

        self.avail_edits = 0
        # self.errors = []

        # TODO: enable setting this
        self.max_search_dist = 1

        # used by IndexDims and ArgShapes to compute index dimensions 
        self.target_nelem = 1e6

        # indices which a change in rank always affects generated inputs.
        # mutations to rank 1 for these indices are not accepted.
        self.definite_rank_indices = set()

        # params is used to retrieve values during testing
        self.arg_order = None
        self.arg_gen_nodes = {} # arg_name => GenNode
        self.arg_pred_nodes = {} # arg_name => PredNode
        self.args_node = None

        # Graphs
        self.pred_graph = {}
        self.gen_graph = {} # idx => node, for generating inventory

        # These will be set to ge.ObservedValue nodes
        self.obs_dtypes = None
        self.obs_shapes = None
        self.obs_args = None
        self.dtypes_filt = None 
        self.fixes = None
        self.predicate_nodes = None
        self.data_format_gobj = None

        # Objects shared between graphs
        self.data_formats = None
        
        self.data_tensors = []
        self.shape_args = []
        self.excluded_combos = base.CombosNotImplemented()
        self.dims_graph = base.CompDimsGraph()
        self.gen_indices = base.GenIndices()
        self.comp_dims_templates = {} # idx => PredNode with TemplateFunc
        self.num_returns = 0
        self.return_nodes = []

        # error status
        self.input_errors = None
        self.framework_error = None

        # call time values
        self.arguments = {}
        self.returns = {}  # 'return[0]' => tensor, 'return[1]' => tensor, ...

    def _pred_node(self, pred_class, name=None):
        name = fgraph.node_name(pred_class, name)
        return self.pred_graph.get(name, None)

    def _pred_nodes(self, *pred_classes):
        return tuple(self._pred_node(n) for n in pred_classes)

    def _gen_node(self, gen_class, name=None):
        name = fgraph.node_name(gen_class, name)
        return self.gen_graph.get(name, None)

    def _init_schema(self, framework_mod, framework_op, init_schema_func):
        # edges to create for the pred graph
        self.framework_mod = framework_mod
        self.framework_op = framework_op
        self.pending_pred_edges = {} # node name -> [parent node name, ...]
        self.pending_index_edges = {} # node name -> [idx, idx, ...]
        self.func_sig = inspect.signature(framework_op)
        self.arg_order = list(self.func_sig.parameters.keys())
        self._init_pred_graph()
        self._init_gen_graph()
        init_schema_func(self)
        self._finalize()

        def wrapped_op(*args, **kwargs):
            # executes during 'framework call phase'
            try:
                self.input_errors = self._check_args(*args, **kwargs)
            except BaseException as ex:
                raise OpGrindInternalError(ex)
            try:
                ret_val = self.framework_op(**self.arguments)
                # ret_val = None
            except BaseException as ex:
                self.framework_error = FrameworkError(ex)
                self.return_status = NotApplicable()
            else:
                self.framework_error = None
                self._check_return(ret_val)
            finally:
                if not (self._opgrind_passed() and self._framework_passed()):
                    self._report()
                if isinstance(self.framework_error, FrameworkError):
                    raise self.framework_error.ex
                return ret_val

        self.wrapped_op = wrapped_op
        return wrapped_op

    def _check_args(self, *args, **kwargs):
        """
        The main function to check all input arguments for all constraints
        registered on the schema
        """
        input_errors = []
        bind = self.func_sig.bind(*args, **kwargs)
        bind.apply_defaults()
        self.arguments = bind.arguments
        self.returns.clear()
        self.framework_error = None
        for dist in range(self.max_search_dist+1):
            self.avail_edits = dist
            ret = fgraph.pred_graph_evaluate(*self.predicate_nodes)
            if ret is None:
                input_errors.append(EditSuggestion())
                break
            else:
                input_errors.extend(ret)
        return input_errors

    def _check_return(self, op_return):
        """
        Check the return tensors' shapes and types against those predicted by
        the framework
        """
        if not isinstance(self.input_errors, Success):
            return

        if not isinstance(op_return, (list, tuple)):
            op_return = (op_return,)
        self.returns = { f'return[{i}]': v for i, v in enumerate(op_return) }
        error = fgraph.pred_graph_evaluate(self.return_pred_graph.values())
        if error is not None:
            raise SchemaError(error.msg(self))

    def _shape_key_order(self, shape_keys):
        def key_fun(shape_key):
            pfx = shape_key.split('.')[0]
            if pfx in self.arg_order:
                return self.arg_order.index(pfx)
            else:
                m = re.match('return\[(\d+)\]', shape_key)
                ind = int(m.group(1))
                return len(self.arg_order) + ind

        key_order = sorted(shape_keys, key=key_fun)
        return key_order

    # TODO: fix headers (possibly integrate a header function in a NodeFunc base
    # class?)
    def _inventory(self):
        """
        Generate a usage inventory for the op.  Includes all combinations of
        input signatures, data format, dtypes
        """
        self.generation_mode = GenMode.Inventory
        self.obs_dtypes.set_cached(None)
        self.obs_shapes.set_cached(None)
        # self.obs_layout.set_cached(None)
        self.obs_args.set_cached(None)
        ranks = self._gen_node(ge.IndexRanks)
        sigs = self._gen_node(ge.SigMap)
        dtypes = self.dtypes_filt
        data_format = self._gen_node(ge.DataFormat)
        out_nodes = (ranks, sigs, dtypes, data_format)
        gen = fgraph.gen_graph_values(self.gen_live_nodes, out_nodes)

        inventory = list(gen)

        # includes args and returns.  args may have a '.k' suffix
        all_sigs = inventory[0][1]
        shape_args = [ *all_sigs ]
        if self.data_formats.configured:
            shape_args.append(self.data_formats.arg_name)
        arg_order = self._shape_key_order(shape_args)

        rows = [arg_order]
        shape_types = (ge.DataTensor, ge.ShapeList, ge.ShapeInt, ge.ShapeTensor)

        for ranks, sigs, dtypes, data_format in inventory:
            row = []
            for arg in arg_order:
                node = self.arg_gen_nodes.get(arg, None)
                func = None if node is None else node.func
                if isinstance(func, ge.DataFormat): 
                    row.append(cand_format)
                elif isinstance(func, shape_types):
                    sig = sigs[arg]
                    inst = ''.join(s * ranks[s] for s in sig)
                    row.append(inst)
                if isinstance(func, ge.DataTensor):
                    dtype = dtypes[arg].name
                    row.append(dtype)
                else:
                    pass
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
        All of these represent shapes to be applied to OpGrind indices.

        'interpretation' shows, for each component of a shape, the one-letter
        OpGrind index name which is inferred in this item.

        'errors' is an ASCII representation highlighting where the error
        occurred.
        
        The list of plain-text suggestions provides one way to fix the errors.
        It is necessarily a guess about what the user might have intended.
        ...

        """
        # need to augment this with another map of other argument values
        args = [ *shape_map ]
        if self.data_formats.configured:
            args.append(self.data_formats.arg_name)
        arg_order = self._shape_key_order(args)
        cand_reports = []

        leader_col = [ 'arguments', 'shapes', 'interpretation', 'errors' ]

        for cand in report:
            # the sub_table is a map of arg_name => rows
            # the rows are: actual shape, signature instantiation, highlight
            # carats
            sub_table = {} 
            for n, shape in shape_map.items():
                assert isinstance(shape, list), 'Shape is not a list'
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

            if self.data_formats.configured:
                if (
                        (data_format == cand.format) or
                        (data_format is None and cand.format is None)
                        ):
                    hl = ''
                else:
                    hl = '^' * max(len(data_format), len(cand.format))
                columns[self.data_formats.arg_name] = [
                        self.data_formats.arg_name, 
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

    def _index_diagram(self, highlight_map, ranks, arg_sigs, shapes):
        arg_order = [ n for n in self.arg_order if n in arg_sigs.keys() ]
        dims = { n: [shp] for n, shp in shapes.items() }
        table_data = {} # arg => [shape, inst, highlight]
                        # shape is e.g.:     [15, 3, 10, 5]
                        # inst is e.g.       ['b', 'i1', 'i2', 'k']
                        # highlight is e.g.: ['', '', '^^', '']

        for arg, sig in arg_sigs.items():
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

    def _index_usage_error(self, idx_usage, ranks, arg_sigs, shapes):
        """
        Generate the message for an IndexUsageError.
        {idx_usage} is: idx => [ (dim => [arg1, ...]),
                                 (dim => [arg1, ...]),
                                 ...
                               ]
        {sigs} is: arg => sig
        {shapes} is: arg => shape
        """
        highlight_map = self._highlight_mask(ranks, arg_sigs, shapes, idx_usage)
        diagram = self._index_diagram(highlight_map, ranks, arg_sigs, shapes)

        index_msgs = []
        for idx, comp in idx_usage.items():
            phrase = self._index_usage_phrase(idx, comp, ranks)
            index_msgs.append(phrase)

        text = '\n'.join(index_msgs)
        return diagram + '\n' + text

    def _index_constraint_error(self, text, index_highlight, ranks, arg_sigs,
            shapes):
        # compute the arg => mask from idx => mask
        arg_highlight = defaultdict(list)
        for arg, sig in arg_sigs.items():
            for s in sig:
                mask = index_highlight.get(s, [False] * ranks[s])
                arg_highlight[arg].extend(mask)

        diagram = self._index_diagram(arg_highlight, ranks, arg_sigs, shapes)
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
        if self.data_formats.configured:
            data_format = self._get_arg(self.data_formats.arg_name)
            row = [ f'\'{self.data_formats.arg_name}\'', data_format ]
            table.append(row)

        rows, _ = tabulate(table, '  ', left_align=[False, True])
        main = ['Received unavailable configuration:']
        main.extend(rows)
        main.append('')
        inst = (f'Use opgrind.list_configs(\'{self.op_path}\') to see all '
                f'available configurations')
        main.append(inst)
        msg = '\n'.join(main)
        return msg

    def _opgrind_passed(self):
        return len(self.input_errors) == 1 and self.input_errors[0].empty()

    def _framework_passed(self):
        return self.framework_error is None

    def _passed(self):
        return (len(self.input_errors) == 1 and 
                self.input_errors[0].empty() and
                self.framework_error is None)

    def _shape_header(self, shape_arg):
        # translate a plain argument name  
        try:
            name, idx = shape_arg.split('.')
        except:
            name, idx = shape_arg, None

        if name not in self.arg_order:
            raise RuntimeError(
                f'{type(self).__qualname__}: name \'{name}\' not a named '
                f'parameter.')
        cls = type(self.arg_gen_nodes[name].func)
        if cls == ge.DataTensor:
            sfx = 'shape'
        elif cls == ge.ShapeTensor:
            sfx = 'numpy()'
        elif cls == ge.ShapeTensor2D:
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
            cls = type(self.arg_gen_nodes[name].func)
            if cls == ge.TensorShape:
                val = arg_val.shape.as_list()
            elif cls == ge.ShapeTensor:
                val = arg_val.numpy().tolist()
            elif cls == ge.ShapeTensor2D:
                val = arg_val.numpy().tolist()
            else:
                val = arg_val
            reps[name] = f'{hdr}={repr(val)}'
        call_string = ', '.join(reps[n] for n in self.arg_order if n in reps) 
        return call_string

    def _report(self):
        pass
        # print(repr(self.input_errors), file=sys.stderr)
        # msg = self.input_errors.message(self)
        # print(msg, file=sys.stderr)

    def _get_arg(self, arg_name):
        """Retrieve the value of {arg_name} argument at call-time."""
        if arg_name not in self.arg_order:
            raise SchemaError(
                f'\'{arg_name}\' not a known parameter. '
                f'Known parameters are: {self.arg_order}')
        return self.arguments[arg_name]

    def _check_sig(self, signature, name):
        if any(s not in self.index.keys() for s in signature):
            raise SchemaError(
                f'Signature "{signature}" associated with \'{name}\' '
                f'contains one or more unregistered indices. '
                f'Current known indices are: '
                f"{','.join(self.index.keys())}"
                f'Call OpSchema.add_index with the missing index.')

    def _get_return(self, ret_name):
        try:
            return self.returns[ret_name]
        except IndexError:
            raise SchemaError(
                f'{type(self).__qualname__}: name \'{ret_name}\' not a '
                f'registered return name')
    
    def _init_pred_graph(self):
        P.set_registry(self.pred_graph)
        P.add_node(pr.Schema(self))
        shapes = P.add_node(pr.ShapeMap())
        dtypes = P.add_node(pr.DTypes())
        argmap = P.add_node(pr.ArgMap())
        inventory = P.add_node(pr.Inventory(self), dtypes, shapes, argmap)
        get_shapes = P.add_node(pr.GetShapes(), inventory)
        self.return_nodes.append(get_shapes)

    def _init_gen_graph(self):
        G.set_registry(self.gen_graph)
        self.obs_dtypes = G.add_node(ge.ObservedValue('dtypes'))
        self.obs_shapes = G.add_node(ge.ObservedValue('shapes'))
        self.obs_args = G.add_node(ge.ObservedValue('args'))
        layout_gobj = ge.Layout(self, base.LAYOUT)
        layout = G.add_node(layout_gobj)
        ranks = G.add_node(ge.IndexRanks())
        impl_obj = ge.DTypesFilter(self)
        self.dtypes_filt = G.add_node(impl_obj, ranks, layout, self.obs_dtypes)
        sigs = G.add_node(ge.SigMap())

        arg_indels = G.add_node(ge.ArgIndels(self), ranks, sigs,
                self.obs_shapes)

        arg_muts_obj = ge.ArgMutations(self)
        arg_muts = G.add_node(arg_muts_obj, arg_indels, ranks, sigs,
                self.obs_shapes)

        self.fixes = G.add_node(ge.Fixes(self), arg_muts, arg_indels,
                self.dtypes_filt, sigs)

        """
        argranks = G.add_node(ge.ArgRanks(self), ranks, sigs)
        indels = G.add_node(ge.Indels(self), argranks, sigs, self.obs_shapes)
        mut_argranks = G.add_node(ge.MutatedArgRanks(self), argranks, indels) 

        # must append each shape arg
        self.gen_hash = ge.ArgRankHash(self)
        hash_node = G.add_node(self.gen_hash, mut_argranks) 

        # must append dims_comp_args
        index_dims_obj = ge.IndexDims(self)
        arg_shapes_obj = ge.ArgShapes(self, index_dims_obj)
        index_dims = G.add_node(index_dims_obj, mut_argranks, ranks, sigs,
                self.obs_shapes)
        index_usage = G.add_node(ge.IndexUsage(self), sigs)
        arg_shapes = G.add_node(arg_shapes_obj, index_dims, sigs, index_usage,
                indels, mut_argranks)
        """
        self.args_node = G.add_node(ge.Args())

        # test_err = G.add_node(ge.EditSuggestionFunc(self))
        # test_err.append_parent(hash_node)
        # test_err.append_parent(self.dtypes_filt)
        # test_err.append_parent(arg_shapes)
        # self.error_node = test_err

        # move pr.GetReturnTensor* and pr.ValidReturnShape* nodes 
        for i in range(self.num_returns):
            ret_name = f'return[{i}]'
            ret_tensor = fgraph.node_name(pr.GetReturnTensor, ret_name)
            valid_return = fgraph.node_name(pr.ValidReturnShape, ret_name)
            ret_node = self.pred_graph.pop(ret_tensor)
            self.return_pred_graph[ret_tensor] = ret_node
            valid_ret_node = self.pred_graph.pop(valid_return)
            self.return_pred_graph[valid_return] = valid_ret_node

    def _finalize(self):
        self.dims_graph.finalize()
        if self.data_formats is None:
            self.arg_layout(None, None, None)

        pred = set(self.pred_graph.values()).difference(self.return_nodes)
        self.predicate_nodes = pred

    def _prep_gen_inference(self, obs_dtypes, obs_shapes, obs_args):
        self.generation_mode = GenMode.Inference
        self.obs_dtypes.set_cached(obs_dtypes)
        self.obs_shapes.set_cached(obs_shapes)
        self.obs_args.set_cached(obs_args)

    def _prep_gen_inventory(self):
        self.avail_edits = 0
        self.generation_mode = GenMode.Inventory

    def _prep_gen_tests(self):
        self.avail_edits = 1
        self.generation_mode = GenMode.Test

    def _generate_args(self):
        self._prep_gen_tests()
        nodes = self.gen_graph.values()
        live_nodes = [n for n in nodes if GenKind.TestLive in n.func.kinds]
        out_nodes = [n for n in nodes if GenKind.TestShow in n.func.kinds]
        for op_args in fgraph.gen_graph_values(live_nodes, out_nodes):
            yield op_args[0] # extract tuple element

    def _validate(self):
        cats = [ 'TP', 'TN', 'FP', 'FN', 'FAIL' ]
        stats = { k: 0 for k in cats }
        for op_args in self._generate_args():
            arg_dict = { k: v.value() for k, v in op_args.items() }
            self.wrapped_op(**arg_dict)
            fr_passed = self._framework_passed()
            if self._opgrind_passed():
                if fr_passed:
                    stats['TN'] += 1
                else:
                    stats['FN'] += 1
                continue

            # establish validity of the checks
            checks = self.input_errors

            # validate this set of checks
            failed_checks = 0
            for fix in checks:
                fixed_args = fix.apply(op_args)
                fixed_arg_dict = {k: v.value() for k, v in fixed_args.items()}
                self.wrapped_op(**fixed_arg_dict)
                if not (self._opgrind_passed() and self._framework_passed()):
                    failed_checks += 1
            if failed_checks > 0:
                stats['FAIL'] += 1
            else:
                if fr_passed:
                    stats['FP'] += 1
                else:
                    stats['TP'] += 1
            print('\r', end='')
            print('  '.join(f'{c}: {stats[cat]:-5d}' for c in cats), end='')


    # ============ PUBLIC API ====================
    def add_index(self, idx, description, constraint=None):
        """
        Add index {idx} with {description} to the schema.  {idx} must be a
        single letter and can be referred to in later signatures.

        {constraint} may be one of:

        1. an integer pair tuple of (<min_rank>, <max_rank>) 
        2. the name of a previously registered index to equate the rank
        3. None - a constraint will be placed downstream limiting these ranks
        """
        if idx in self.index:
            raise SchemaError(f'Index \'{idx}\' already registered') 

        ranks_node = self._gen_node(ge.IndexRanks)
        G.set_registry(self.gen_graph)
        if isinstance(constraint, str):
            primary_idx = constraint
            if primary_idx not in self.index:
                raise SchemaError(f'Source index \'{primary_idx}\' is not '
                        f'a registered index')
            elif self.equiv_index[primary_idx] != primary_idx:
                raise SchemaError(f'Source index \'{primary_idx}\' is not '
                        f'a primary index')
            else:
                obj = ge.EquivRange(self, idx)
                pa = self.gen_graph[primary_idx]
                idx_node = G.add_node_sn(obj, pa) 
                ranks_node.append_parent_sn(idx_node)
                self.equiv_index[idx] = primary_idx

        elif isinstance(constraint, (tuple, type(None))):
            # indel_node = self._gen_node(ge.Indels)
            # obj = ge.RankRange(self, idx, indel_node)
            obj = ge.RankRange(self, idx)
            sigs_node = self._gen_node(ge.SigMap)
            idx_node = G.add_node_sn(obj, self.obs_shapes, sigs_node)
            ranks_node.append_parent_sn(idx_node)
            if isinstance(constraint, tuple):
                pair = constraint
                if not (len(pair) == 2 
                        and isinstance(pair[0], int) 
                        and isinstance(pair[1], int) 
                        and 0 <= pair[0] 
                        and pair[0] <= pair[1]):
                    raise SchemaError(
                        f'{type(self).__qualname__}: Got constraint tuple '
                        f'\'{pair}\' but it is not a 2-integer tuple')
                lo, hi = pair
                cons = base.SumRangeConstraint(idx, lo, hi)
                obj.add_schema_constraint(cons)

            primary_inds = { *self.equiv_index.values() }
            for primary_idx in primary_inds:
                pa = self.gen_graph[primary_idx]
                idx_node.append_parent_sn(pa)

            self.equiv_index[idx] = idx
        else:
            raise SchemaError(
                f'{type(self).__qualname__}: Got constraint \'{constraint}\''
                f' but expected either an index, None, or an integer pair.')

        self.index[idx] = description

    def arg_unchecked(self, arg_name):
        """
        Declare {arg_name} to be an argument unchecked by OpGrind 
        """
        pass

    # TODO: add validation to restrict extra_args to prevent graph cycles
    def computed_index(self, comp_index, comp_func, tem_func, input_indexes,
            min_val, *extra_args):
        """
        Registers {comp_func} to compute the dimensions of {comp_index}.
        Registers {tem_func} which produces a text string explaining how
        the index is computed.

        Adds an index predicate to ensure all components of the computed index
        are >= {min_val}

        {extra_args} can be one of: LAYOUT or a parameter registered with
        arg_option.  

        The following calls are made:

        comp_func(*index_dims, *extra_vals)
        (index_dims are the resolved dimensions of {input_indexes})
        (extra_vals are the runtime values of {extra_args})

        If a downstream constraint (registered with add_index_predicate) fails,
        then OpGrind makes these calls:

        tem_func(*index_desc, *extra_vals)
        tem_func(*index_dims, *extra_vals)
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

        extra_node_names = []
        arg_muts = self._gen_node(ge.ArgMutations)
        # index_dims = self._gen_node(ge.IndexDims)
        for arg in extra_args:
            if arg == base.LAYOUT:
                node = self._gen_node(ge.Layout, base.LAYOUT)
            else:
                node = self.arg_gen_nodes[arg]
            extra_node_names.append(node.name)
            arg_muts.maybe_append_parent_sn(node)

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
                *extra_args)

        # add a predicate to ensure the computed index is >= some minimum value
        bounds_pobj = flib.PredAbove(min_val)
        self.add_index_predicate(f'{comp_index} >= {min_val}', bounds_pobj,
                comp_index)  

    def _gen_nodes_map(self):
        # get a map of idx => primary rank node (from the gen_graph) for all
        # indexes
        nodes = {} 
        for idx in self.index.keys():
            pri_idx = self.equiv_index[idx]
            node = self.gen_graph[pri_idx]
            nodes[idx] = node
        return nodes

    def limit_ranks(self, sig, min_val, max_val):
        """
        Declare that the rank of {sig} be in [{min_val}, {max_val}]
        """
        self._check_sig(sig, 'rank limits')
        node_map = self._gen_nodes_map()
        for idx in sig:
            if idx not in node_map:
                raise SchemaError(
                    f'Index \'{idx}\' mentioned in signature \'{sig}\' was '
                    f'not registered with add_index.  All indices must first '
                    f'be registered before being used in a limit_ranks call')

        # add constraint to each node in the sig
        pri_sig = ''.join(sorted(self.equiv_index[idx] for idx in sig))
        cons = base.SumRangeConstraint(pri_sig, min_val, max_val)
        for idx in sig:
            node = node_map[idx]
            node.func.add_schema_constraint(cons)

        # self.rank_candidates.add_rank_limits(sig, min_val, max_val)


    def _is_data_tensor(self, name):
        node = self.arg_pred_nodes.get(name, None)
        return node is not None and isinstance(node.func, pr.DataTensor)

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
        """
        if self.dtype_cons.has_valid_dtypes(tensor_name):
            raise SchemaError(
                f'{self.__qualname__}: Tensor \'{tensor_name}\' is already '
                f'registered with valid dtypes')
        """

        dtypes = [ t for ex in type_list for t in base.dtype_expr(ex) ]
        # self.dtype_cons.add_valid(tensor_name, dtypes)

        G.set_registry(self.gen_graph)
        obj = ge.DTypeIndiv(self, tensor_name, dtypes)
        dtype_node = G.add_node(obj, self.obs_dtypes)
        self.dtypes_filt.append_parent_sn(dtype_node)

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

        G.set_registry(self.gen_graph)
        obj = ge.DTypeEquiv(self, trg_tensor, src_tensor)
        src_dtype = self._gen_node(ge.DTypeIndiv, src_tensor)
        trg_dtype = G.add_node(obj, self.obs_dtypes, src_dtype)
        self.dtypes_filt.append_parent_sn(trg_dtype)

    def exclude_combos(self, *field_value_pairs):
        """
        Allows to mark combinations of dtypes, ranks, and layout as excluded
        from the valid set.  This is useful for those cases that are not
        implemented by the framework.

        {field_val_pairs} is an even-length list of field, val, field, val, ...
        field is one of: 
        - data tensors registered in init_fields
        - one-letter index names registered in init_fields
        - the constant LAYOUT, if has_layout

        val is one of:
        - dtype string, such as 'int32' for data tensor fields
        - integer specifying a rank of an index field
        - the LAYOUT field has an integer in [0, num_layouts), as defined
          by the call to arg_layout.
        """
        if not self.excluded_combos.initialized:
            self.excluded_combos.init_fields(self.data_tensors,
                    self.index.keys())
        try: 
            self.excluded_combos.add_combo(*field_value_pairs)
        except RuntimeError as ex:
            raise SchemaError(ex)

    def arg_int(self, arg_name, lo=None, hi=None):
        """
        Declare {arg_name} to be an integer that can take on values in a range.
        If {lo} is None, it is sys.maxint
        If {hi} is None, it is -sys.maxint-1 
        """
        G.set_registry(self.gen_graph)
        pred_obj = pr.ArgInt(arg_name, lo, hi)
        gen_obj = ge.Int(lo, hi)
        schema = self._pred_node(pr.Schema)
        p_arg = P.add_node(pred_obj, schema)
        g_arg = G.add_node(gen_obj)
        self.arg_gen_nodes[arg_name] = g_arg
        self.arg_pred_nodes[arg_name] = p_arg
        self.args_node.append_parent_sn(g_arg)

    def arg_option(self, arg_name, options):
        """
        Expect {arg_name} to take on one of the values in {options}
        """
        G.set_registry(self.gen_graph)
        P.set_registry(self.pred_graph)
        options_gobj = ge.Options(self, arg_name, options)
        g_arg = G.add_node(options_gobj, self.obs_args)
        options_pobj = pr.Options(arg_name, options_gobj, options)
        schema = self._pred_node(pr.Schema)
        p_arg = P.add_node(options_pobj, schema)
        arg_node = self._pred_node(pr.ArgMap)
        arg_node.append_parent_sn(p_arg)
        self.arg_pred_nodes[arg_name] = p_arg
        self.arg_gen_nodes[arg_name] = g_arg
        self.args_node.append_parent_sn(g_arg)
        self.fixes.append_parent_sn(g_arg)

    def arg_layout(self, arg_name, formats, rank_idx):
        """
        Declares {arg_name} to control layout-dependent signatures for tensors. 
        {layouts} is an array, where each element is a map of: rank => code
        The rank of {rank_idx} determines which layout is mapped.
        """
        G.set_registry(self.gen_graph)
        P.set_registry(self.pred_graph)
        self.data_formats = base.DataFormats(arg_name, formats, rank_idx)
        
        # define the real arg 
        layout = self._gen_node(ge.Layout, base.LAYOUT)
        ranks = self._gen_node(ge.IndexRanks)
        df_gobj = ge.DataFormat(self, self.data_formats, arg_name, rank_idx)
        g_arg = G.add_node(df_gobj, ranks, layout, self.obs_args) 
        self.data_format_gobj = df_gobj
        self.fixes.append_parent_sn(g_arg)

        if arg_name is not None:
            self.arg_gen_nodes[arg_name] = g_arg
            self.args_node.append_parent_sn(g_arg)
            # hash_node = self._gen_node(ge.ArgRankHash)
            # hash_node.append_parent(g_arg)

        schema = self._pred_node(pr.Schema)
        data_format_obj = pr.DataFormat(self.data_formats, df_gobj, arg_name)
        p_arg = P.add_node(data_format_obj, schema) 
        self.arg_pred_nodes[arg_name] = p_arg

        arg_node = self._pred_node(pr.ArgMap)
        arg_node.append_parent_sn(p_arg)

    def _check_sigs_layout(self, arg_name, sigs_list):
        if self.data_formats is None:
            # arg_layout is implicitly 1
            num_layouts = 1
        else:
            num_layouts = self.data_formats.num_layouts()
        if len(sigs_list) == 1:
            sigs_list = sigs_list * num_layouts

        if len(sigs_list) != num_layouts:
            raise SchemaError(
                f'{type(self).__qualname__}: registering \'{arg_name}\' '
                f'there are {self.num_layouts} '
                f'layouts (as established by the call to \'arg_layout\') but '
                f'{len(sigs_list)} elements of \'sigs\' argument.')
        return sigs_list 

    def _add_definite_rank(self, *sigs):
        """
        Add the indices in sigs to the set of so-called 'definite-rank
        indices'.  Such an index has the property that a change in rank always
        affects generated output.  Most indices have this property, but indices
        exclusively registered with arg_shape_bcast_list do not.  This is
        because such an index could produce the same output with rank != 1 and
        rank 1, since any rank != 1 could be broadcasted.
        """
        self.definite_rank_indices.update(idx for sig in sigs for idx in sig)

    def _arg_shape_func(self, arg_name, sigs_list, shape_pnode, arg_gobj, kind): 
        """
        Backend function for arg_shape_* API functions.
        sigs_list must be a list of either 1 or num_layout elements.  If 1, it
        is implicitly broadcasted to num_layouts
        """
        sigs_list = self._check_sigs_layout(arg_name, sigs_list)
        P.set_registry(self.pred_graph)
        G.set_registry(self.gen_graph)
        # node: ge.Sig 
        # node: one of ge.DataTensor, ge.ShapeList, ge.ShapeInt, ge.ShapeTensor    
        # edges: ge.SigMap -> ge.Sig, [newnode] -> ge.RankStatusArgShape 
        # arg_shapes = self._gen_node(ge.ArgShapes)
        arg_shapes = self._gen_node(ge.ArgMutations)
        dtypes = self._gen_node(ge.DTypesFilter)
        if isinstance(arg_gobj, ge.DataTensor):
            arg_node = G.add_node(arg_gobj, arg_shapes, dtypes)
        else:
            arg_node = G.add_node(arg_gobj, arg_shapes)
        self.args_node.append_parent_sn(arg_node)
        self.arg_gen_nodes[arg_name] = arg_node
        
        G.set_registry(self.gen_graph)
        sigmap = self._gen_node(ge.SigMap)
        layout = self._gen_node(ge.Layout, base.LAYOUT)
        sig_obj = ge.Sig(self, arg_name, sigs_list)
        sig_node = G.add_node(sig_obj, layout)
        sigmap.append_parent_sn(sig_node)

        shape_map = self._pred_node(pr.ShapeMap)
        shape_map.append_parent_sn(shape_pnode)
        self.shape_args.append(arg_name)

    def arg_tensor(self, arg_name, *sigs):
        """
        Register {arg_name} as a tensor.  

        sigs are all strings of signatures.  If len(sigs) == 1, then it
        specifies a static signature regardless of whether 'arg_layout' was
        called.  If len(sigs) > 1, then arg_layout is required to be called
        before this call.
        """
        schema = self._pred_node(pr.Schema)
        shp_pobj = pr.TensorShape(arg_name)
        arg_gobj = ge.DataTensor(arg_name)
        arg_pobj = pr.DataTensor(arg_name, arg_gobj)
        arg_p = P.add_node(arg_pobj, schema)
        shp_pobj = pr.TensorShape(arg_name)
        shp_p = P.add_node(shp_pobj, arg_p)
        kind = ShapeKind.DataTensor
        self._arg_shape_func(arg_name, sigs, shp_p, arg_gobj, kind)

        P.set_registry(self.pred_graph)
        dtypes = self._pred_node(pr.DTypes)
        tensor_dtype_obj = pr.TensorDType(arg_name)
        dtype = P.add_node(tensor_dtype_obj, arg_p)
        dtypes.append_parent_sn(dtype)
        self._add_definite_rank(*sigs)
        self.arg_pred_nodes[arg_name] = arg_p
        self.data_tensors.append(arg_name)

    def _arg_shape_list_base(self, arg_name, broadcast_mode=False, *sigs):
        """
        See arg_shape_bcast_list and arg_shape_list
        """
        P.set_registry(self.pred_graph)
        schema = self._pred_node(pr.Schema)
        arg_gobj = ge.ShapeList(arg_name)
        arg_pobj = pr.ShapeList(arg_name, arg_gobj, broadcast_mode)
        arg_p = P.add_node(arg_pobj, schema) 
        kind = ShapeKind.List
        self._arg_shape_func(arg_name, sigs, arg_p, arg_gobj, kind)
        self.arg_pred_nodes[arg_name] = arg_p

    def arg_shape_bcast_list(self, arg_name, *sigs):
        """
        Register {arg_name} as an integer list parameter which defines the
        shape of a signature.  

        Expect arg_name value to be list of non-negative integers.
        If arg_val is length 1, interpret it as a generic broadcasted shape of
        unspecified rank.
        """
        self._arg_shape_list_base(arg_name, True, *sigs)

    def arg_shape_list(self, arg_name, *sigs):
        """
        Register {arg_name} as an integer list parameter which defines the
        shape of a signature.  

        Expect arg_name value to be list of non-negative integers defining a
        shape.  In contrast to arg_shape_bcast_list, here there is no
        broadcasting interpretation.
        """
        self._add_definite_rank(*sigs)
        self._arg_shape_list_base(arg_name, False, *sigs)

    def arg_shape_int(self, arg_name, index):
        """
        Register {arg_name} as an integer parameter which defines the shape of
        an index.  The shape will be the broadcasted value of the argument if
        the index has rank greater than 1.
        """
        # TODO: currently uses arg_shape_func, which assumes the arg defines
        # the rank.  In this case, it only defines the shape as the integer 
        # value broadcasted {rank} times.  But, the rank is not determined from
        # this input
        P.set_registry(self.pred_graph)
        schema = self._pred_node(pr.Schema)
        gen_obj = ge.ShapeInt(arg_name)
        pred_obj = pr.ShapeInt(arg_name, gen_obj)
        arg_p = P.add_node(pred_obj, schema)
        kind = ShapeKind.Int
        self._arg_shape_func(arg_name, (index,), arg_p, gen_obj, kind)
        self.arg_pred_nodes[arg_name] = arg_p

    def arg_shape_tensor(self, arg_name, *sigs):
        """
        Register {arg_name} as a 1D integer tensor whose elements define the
        shape of a signature.  
        """
        P.set_registry(self.pred_graph)
        schema = self._pred_node(pr.Schema)
        gen_obj = ge.ShapeTensor(arg_name)
        pred_obj = pr.ShapeTensor(arg_name, gen_obj)
        arg_p = P.add_node(pred_obj, schema)
        kind = ShapeKind.Tensor
        self._arg_shape_func(arg_name, sigs, arg_p, gen_obj, kind)
        self._add_definite_rank(*sigs)
        self.arg_pred_nodes[arg_name] = arg_p

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
        # created nodes
        # pr.ShapeTensor2D, ge.ShapeTensor2D
        # ge.Sig(i)
        # pr.SliceShape(i)

        # created edges
        # pr.ShapeTensor2D -> pr.Schema
        # ge.ShapeTensor2D -> ge.GetArgShapes
        # pr.SliceShape(i) -> pr.ShapeTensor2D
        # ge.Sig(i) -> ge.Layout
        # ge.SigMap -> ge.Sig(i)
        P.set_registry(self.pred_graph)
        G.set_registry(self.gen_graph)
        schema = self._pred_node(pr.Schema)
        shape2d_gobj = ge.ShapeTensor2D(arg_name, len(sigs))
        shape2d_pobj = pr.ShapeTensor2D(arg_name, shape2d_gobj, len(sigs))
        p_shape2d = P.add_node(shape2d_pobj, schema)
        self.arg_pred_nodes[arg_name] = p_shape2d

        # arg_shapes = self._gen_node(ge.GetArgShapes)
        arg_shapes = self._gen_node(ge.ArgMutations)
        g_shape2d = G.add_node(shape2d_gobj, arg_shapes)
        self.arg_gen_nodes[arg_name] = g_shape2d
        self.args_node.append_parent_sn(g_shape2d)

        g_sig_map = self._gen_node(ge.SigMap)
        g_layout = self._gen_node(ge.Layout, base.LAYOUT)
        p_shape_map = self._pred_node(pr.ShapeMap)
        # p_layout = self._pred_node(pr.Layout, base.LAYOUT)

        for i, sig in enumerate(sigs):
            prefix = f'{arg_name}.{i}'

            # pr.ShapeMap -> pr.SliceShape
            shp_pobj = pr.SliceShape(arg_name, i)
            p_shp = P.add_node(shp_pobj, p_shape2d)
            p_shape_map.append_parent(p_shp)

            if isinstance(sig, str):
                sig = [sig]
            g_sig_obj = ge.Sig(self, prefix, sig)
            g_sig = G.add_node(g_sig_obj, g_layout)
            g_sig_map.append_parent(g_sig)
        self._add_definite_rank(*sigs)

    def arg_rank(self, arg_name, sig):
        """
        Register {arg_name} to be an integer argument which defines the rank of
        {sig}
        """
        cons_name = f'rank({sig}) == \'{arg_name}\''
        rank_pobj = pr.ArgInt(arg_name, 0, None)

        P.set_registry(self.pred_graph)
        G.set_registry(self.gen_graph)
        schema = self._pred_node(pr.Schema)
        p_rank = P.add_node(rank_pobj, schema)
        self.arg_pred_nodes[arg_name] = p_rank

        g_ranks = self._gen_node(ge.IndexRanks)
        g_rank = G.add_node(ge.Rank(self, sig), g_ranks)
        self.arg_gen_nodes[arg_name] = g_rank
        self.args_node.append_parent_sn(g_rank)

        self._add_definite_rank(sig)

    def rank_dims_constraint(self, constraint_name, get_dims, rank_sig,
            dims_index, shape_arg):
        """
        Creates a constraint called {constraint_name} with the logic:
        RANK(rank_sig) == get_dims(shape_arg).

        Creates a generated index dimension:
        DIMS(dims_index) <- RANK(rank_sig)
        """
        # 'sum' simply sums up the individual ranks of indices in rank_sig 
        def gen_single_index(ranks_list):
            val = sum(ranks_list)
            return [([val],)]

        self.gen_indices.add_generator(gen_single_index, dims_index, rank_sig)
        self.dims_graph.maybe_add_input_index(dims_index)
        self._add_definite_rank(rank_sig)

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
        # TODO: Find the appropriate place for this constraint
        # P.set_registry(self.pred_graph)
        # id_cons_obj = pr.IndexDimsConstraint(pred_name, status_func)
        # ids = (pr.GetRanks, pr.GetArgSigs, pr.ShapeMap, pr.Schema)
        # parents = self._pred_nodes(*ids)
        # id_cons = P.add_node(id_cons_obj, *parents)
        # self.pending_index_edges[id_cons.name] = indices

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

    # TODO: should I clone the graph, or simply set the parents to nodes in the
    # gen graph?
    def return_tensor(self, *sigs):
        """
        Append a return tensor to the list of expected return tensors.

        *sigs may contain either one element, or {num_layout} elements.  If one
        element, it defines the static signature for the return tensor.  If
        multiple, they are defined by the provided layout as declared in
        'arg_layout'
        """
        index = self.num_returns
        ret_name = f'return[{index}]'
        sigs_list = self._check_sigs_layout(ret_name, sigs)

        P.set_registry(self.pred_graph)
        G.set_registry(self.gen_graph)

        g_sig_obj = ge.Sig(self, ret_name, sigs_list)
        # p_sig_obj = pr.Sig(ret_name, sigs_list)

        rten_pobj = pr.GetReturnTensor(ret_name)
        rvalid_pobj = pr.ValidReturnShape(ret_name)
        # pred_shape_pobj = pr.PredictedShape(ret_name)

        schema = self._pred_node(pr.Schema)
        # layout = self._pred_node(pr.Layout, base.LAYOUT)
        rten = P.add_node(rten_pobj, schema)

        # sig_inds = { idx for sig in sigs for idx in sig }
        # self.pending_index_edges[pred_shape.name] = list(sig_inds)
        shapes = self._pred_node(pr.GetShapes)
        rval = P.add_node(rvalid_pobj, rten, shapes)
        self.return_nodes.extend((rten, rval))
        self.num_returns += 1

