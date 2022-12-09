import itertools
import copy
import numpy as np
from contextlib import contextmanager
from collections import namedtuple
from .fgraph import NodeFunc
from .base import ALL_DTYPES, INDEX_RANKS, LAYOUT
from . import generators as ge
from . import base
from . import fgraph
from . import oparg

"""
The inference graph (inf_graph) is constructed using nodes in this file.  Its
job is to ingest the arguments provided by the op, which are delivered via the
ObservedValue nodes.

"""
class ReportNodeFunc(NodeFunc):
    """
    NodeFunc which further implements user-facing reporting functions
    """
    def __init__(self, op, name=None):
        super().__init__(name)
        self.op = op

    @contextmanager
    def reserve_edit(self, dist):
        doit = (dist <= self.op.avail_edits)
        if doit:
            self.op.avail_edits -= dist
        try:
            yield doit
        finally:
            if doit:
                self.op.avail_edits += dist

class ObservedValue(NodeFunc):
    """
    Node for delivering inputs to any individual rank nodes.
    This is the portal to connect the rank graph to its environment
    """
    def __init__(self, name):
        super().__init__(name)

    def __call__(self):
        return [{}]

class Layout(NodeFunc):
    def __init__(self, op):
        super().__init__(None)
        self.op = op

    def __call__(self):
        num_layouts = self.op.data_formats.num_layouts()
        for i, layout in enumerate(range(num_layouts)):
            yield layout

class RankRange(ReportNodeFunc):
    """
    Produce a range of all valid ranks of a primary index.  'Valid' means
    obeying all schema constraints.
    """
    def __init__(self, op, name):
        super().__init__(op, name)
        self.schema_cons = []
        self.obs_shapes_cons = [] # constraint based on shapes
        self.obs_args_cons = []

    def add_schema_constraint(self, cons):
        # these functions are called with **index_ranks
        self.schema_cons.append(cons)

    def add_shapes_constraint(self, cons):
        # these functions are called with obs_shapes, **index_ranks
        self.obs_shapes_cons.append(cons)

    def add_args_constraint(self, cons):
        # these functions are called with obs_args, **index_ranks
        self.obs_args_cons.append(cons)

    def __call__(self, obs_shapes, obs_args, **index_ranks):
        # Get the initial bounds consistent with the schema
        sch_lo, sch_hi = 0, 100000
        for cons in self.schema_cons:
            clo, chi = cons(**index_ranks)
            sch_lo = max(sch_lo, clo)
            sch_hi = min(sch_hi, chi)

        for cons in self.obs_shapes_cons:
            clo, chi = cons(obs_shapes, **index_ranks)
            sch_lo = max(sch_lo, clo)
            sch_hi = min(sch_hi, chi)

        for cons in self.obs_args_cons:
            clo, chi = cons(obs_args, **index_ranks)
            sch_lo = max(sch_lo, clo)
            sch_hi = min(sch_hi, chi)

        for i in range(sch_lo, sch_hi+1):
            yield i

class RankEquiv(NodeFunc):
    """
    Produce a range identical to the primary index
    """
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, rank):
        yield rank

class IndexRanks(NodeFunc):
    """
    Gather ranks together index ranks into one map
    Parents:  RankRange and RankEquiv nodes
    """
    def __init__(self):
        super().__init__()

    def __call__(self, **ranks):
        yield ranks

class ArgIndels(ReportNodeFunc):
    """
    Implicitly calculates the expected (arg => exp_rank) from the current
    index_ranks and sigs.  Then, computes the (arg => delta) map as: arg =>
    (exp_rank - obs_rank) and yields it.  If an observed shape is an integer,
    this indicates a 'rank agnostic' shape.  delta is always zero in this case.
     """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, index_ranks, sigs, obs_shapes, layout):
        arg_ranks = {}
        for arg, sig in sigs.items():
            rank = sum(index_ranks[idx] for idx in sig)
            arg_ranks[arg] = rank
        """
        Produces instructions to insert part of an index's dimensions, or
        delete a subrange from a shape.  
        """
        arg_delta = {}
        edit = base.ShapeEdit(self.op, index_ranks, sigs, layout)

        for arg, rank in arg_ranks.items():
            if arg not in obs_shapes:
                continue
            obs_shape = obs_shapes[arg]
            sig = sigs[arg]
            if isinstance(obs_shape, int):
                obs_rank = None
                delta = 0 # rank-agnostic shape cannot have rank violation
            else:
                obs_rank = len(obs_shape)
                delta = rank - obs_rank

            if delta == 0:
                continue

            else:
                arg_delta[arg] = delta

        edit.add_indels(arg_delta)
        with self.reserve_edit(edit.cost()) as avail: 
            if avail:
                yield edit

class IndexUsage(ReportNodeFunc):
    """
    Construct the usage map idx => (dims => [arg1, ...]), and add it to the
    received shape_edit object.
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, index_ranks, shape_edit, obs_shapes):
        # compute idx usage
        # if indels are present, pass-through
        if shape_edit.indel_cost() != 0:
            yield shape_edit
            return

        usage_map = {} # idx => (dims => [arg1, ...]) 
        sigs = shape_edit.arg_sigs
        for arg, obs_shape in obs_shapes.items():
            sig = sigs[arg]
            if isinstance(obs_shape, int):
                assert len(sig) == 1, f'obs_shape was integer but sig was {sig}'
                idx = sig[0]
                usage = usage_map.setdefault(idx, {})
                args = usage.setdefault(obs_shape, set())
                args.add(arg)
            else:
                off = 0
                for idx in sig:
                    usage = usage_map.setdefault(idx, {})
                    dims = tuple(obs_shape[off:off+index_ranks[idx]])
                    args = usage.setdefault(dims, set())
                    args.add(arg)
                    off += index_ranks[idx]
        shape_edit.add_idx_usage(usage_map)
        with self.reserve_edit(shape_edit.cost()) as avail:
            if avail:
                yield shape_edit

Formula = namedtuple('Formula', ['dims', 'code_path', 'desc_path', 'dims_path'])

class IndexConstraints(ReportNodeFunc):
    """
    Add results of evaluating the index constraints onto the shape_edit object
    """
    def __init__(self, op):
        super().__init__(op)
        self.index_preds = op.index_preds

    def get_comp_order(self):
        comp_nodes = [ n for n in self.op.dims_graph.values() if 
                isinstance(n.func, ge.CompDims) ]
        comp_order = [ c.sub_name for c in comp_nodes ]
        return comp_order

    def init_dims_graph(self, idx_info):
        # initialize input nodes
        for sig, node in self.op.dims_graph.items():
            if not isinstance(node.func, ge.GenDims):
                continue
            elif all(idx in idx_info for idx in sig):
                if len(sig) == 1:
                    val = idx_info[sig]
                else:
                    val = tuple(idx_info[idx] for idx in sig)
                node.set_cached(val)
            else:
                pass

    def get_template(self, idx_info):
        """
        Compute right-hand-side formulas using idx_info, which is a map of
        idx => info.  info may be one of:
        - index one-letter code, e.g.  'i'
        - snake-cased description, e.g. 'input_spatial'
        - string representation of dimensions, e.g. '[2,4,6]'

        Return [formula, ...], where 
        """
        self.op.comp_dims_mode = False
        self.init_dims_graph(idx_info)
        comp_order = self.get_comp_order()
        comp_nodes = [ self.op.dims_graph[idx] for idx in comp_order ]
        gen = fgraph.gen_graph_iterate(comp_nodes)
        items = list(gen)
        assert len(items) == 1, 'Internal Error with comp graph'
        templ_map = items[0]
        templ_list = [ templ_map[idx] for idx in comp_order ]
        self.op.comp_dims_mode = True
        return templ_list 

    def get_comp_dims(self, index_dims):
        """
        Compute dims from input index dims
        """
        self.op.comp_dims_mode = True
        self.init_dims_graph(index_dims)
        comp_order = self.get_comp_order()
        comp_nodes = [ self.op.dims_graph[idx] for idx in comp_order ]
        gen = fgraph.gen_graph_iterate(comp_nodes)
        comp_dims_list = list(gen)
        assert len(comp_dims_list) == 1, 'Internal Error with comp graph'
        comp_dims = comp_dims_list[0]
        return comp_dims

    def __call__(self, shape_edit, obs_args):
        if shape_edit.cost() != 0:
            yield shape_edit
            return

        # each usage should have a single entry
        self.op.dims_graph_input.update(obs_args)
        self.op.dims_graph_input[INDEX_RANKS] = shape_edit.index_ranks
        self.op.dims_graph_input[LAYOUT] = shape_edit.layout

        input_dims = shape_edit.get_input_dims()
        comp_dims = self.get_comp_dims(input_dims)
        shape_edit.add_comp_dims(comp_dims)

        idx_codes = {} # e.g. 'i'
        idx_descs = {} # e.g. 'input_spatial'
        idx_sdims = {} # e.g. '[2,3,5]'
        all_dims = { **input_dims, **comp_dims }
        for idx, dims in all_dims.items():
            idx_codes[idx] = idx
            idx_descs[idx] = base.snake_case(self.op.index[idx].desc)
            idx_sdims[idx] = base.dims_string(dims)
        
        frm_codes = self.get_template(idx_codes)
        frm_descs = self.get_template(idx_descs)
        frm_sdims = self.get_template(idx_sdims)

        comp_order = self.get_comp_order() 
        formulas = {}
        for p, idx in enumerate(comp_order):
            code_path = idx_codes[idx] + ' = ' + frm_codes[p]
            desc_path = idx_descs[idx] + ' = ' + frm_descs[p]
            dims_path = idx_sdims[idx] + ' = ' + frm_sdims[p]
            dims = comp_dims[idx] 
            formulas[idx] = Formula(dims, code_path, desc_path, dims_path)
        
        # formulas = self.op.dims_graph.templates(input_dims, **comp) 
        index_dims = { **input_dims, **comp_dims }

        for pred in self.index_preds:
            # skip any predicates if any input indices are missing - this can
            # happen when the predicate only applies to specific layouts
            if not all(idx in index_dims for idx in pred.indices):
                continue
            pred_input_dims = [ index_dims[idx] for idx in pred.indices ]
            if not pred(*pred_input_dims):
                # collect the predecessor formulas
                comp_order = self.get_comp_order()
                en = enumerate(comp_order)
                max_pos = max((p for p, i in en if i in pred.indices), default=-1)
                source_formulas = []
                for idx in comp_order[:max_pos+1]:
                    source_formulas.append(formulas[idx])
                shape_edit.add_constraint_error(pred, source_formulas)

        with self.reserve_edit(shape_edit.cost()) as avail:
            if avail:
                yield shape_edit

class DataFormat(ReportNodeFunc):
    """
    Generate the special data_format argument, defined by the 'layout' API call
    Inference: yields None or ValueEdit
    """
    def __init__(self, op, formats, arg_name):
        super().__init__(op, arg_name)
        self.formats = formats

    def __call__(self, ranks, layout, obs_args):
        imp_fmt = self.formats.data_format(layout, ranks)
        obs_fmt = self.formats.observed_format(obs_args)
        if obs_fmt is None:
            # this will only occur if the schema permits None for data_format
            used_fmt = self.formats.default_format(ranks)
        else:
            used_fmt = obs_fmt

        arg_name = self.formats.arg_name
        edit = base.DataFormatEdit(arg_name, obs_fmt, used_fmt, imp_fmt)
        with self.reserve_edit(edit.cost()) as avail:
            if avail:
                yield edit

class Options(ReportNodeFunc):
    """
    Represent a specific set of options known at construction time
    """
    def __init__(self, op, name, options):
        super().__init__(op, name)
        self.arg_name = name
        try:
            iter(options)
        except TypeError:
            raise SchemaError(
                f'{type(self).__qualname__}: \'options\' argument must be '
                f'iterable.  Got {type(options)}')
        self.options = options

    def __call__(self, obs_args):
        obs_option = obs_args[self.arg_name]
        for imp_option in self.options:
            edit = base.ValueEdit(self.arg_name, obs_option, imp_option)
            with self.reserve_edit(edit.cost()) as avail:
                if avail:
                    yield edit

class DTypes(ReportNodeFunc):
    def __init__(self, op):
        super().__init__(op)
        self.rules = op.dtype_rules

    def __call__(self, obs_dtypes, index_ranks, layout):
        edit = self.rules.edit(obs_dtypes, index_ranks, layout)
        with self.reserve_edit(edit.cost()) as avail:
            if avail:
                yield edit

class Report(NodeFunc):
    def __init__(self, op):
        super().__init__(None)
        self.op = op

    def __call__(self, dtype_edit, shape_edit, **kwargs):
        df_node_name = self.op.data_format_inode.used_name()
        df_edit = kwargs.pop(df_node_name)
        res = base.Fix(df_edit, dtype_edit, shape_edit, **kwargs)
        yield res

