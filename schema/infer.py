import itertools
import copy
from contextlib import contextmanager
from .fgraph import NodeFunc
from .base import ALL_DTYPES
from . import base
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

    def add_schema_constraint(self, cons):
        self.schema_cons.append(cons)

    def __call__(self, _, sigs, **index_ranks):
        # Get the initial bounds consistent with the schema
        sch_lo, sch_hi = 0, 1e10
        for cons in self.schema_cons:
            clo, chi = cons(**index_ranks)
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
        edit = base.ShapeEdit(self, index_ranks, sigs, layout)

        for arg, rank in arg_ranks.items():
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

class IndexConstraints(ReportNodeFunc):
    """
    Add results of evaluating the index constraints onto the shape_edit object
    """
    def __init__(self, op):
        super().__init__(op)
        self.cons = op.index_preds

    def __call__(self, shape_edit, **comp):
        if shape_edit.cost() != 0:
            yield shape_edit
            return

        # each usage should have a single entry
        index_dims = shape_edit.get_index_dims()
        self.op.dims_graph.template_mode = True
        index_templ = self.op.dims_graph(index_dims, **comp) 

        for pred in self.cons.preds:
            input_templs = [ index_templ[i] for i in pred.indices ]
            input_dims = [ t.dims for t in input_templs ]
            if not pred.func(*input_dims):
                shape_edit.add_constraint_error(pred.name, input_templs)
                break

        with self.reserve_edit(shape_edit.cost()) as avail:
            if avail:
                yield shape_edit

class DataFormat(ReportNodeFunc):
    """
    Generate the special data_format argument, defined by the 'layout' API call
    Inference: yields None or ValueEdit
    """
    def __init__(self, op, formats, arg_name, rank_idx):
        super().__init__(op, arg_name)
        self.formats = formats
        self.arg_name = arg_name
        self.rank_idx = rank_idx

    def __call__(self, ranks, layout, obs_args):
        inf_format = self.formats.data_format(layout, ranks)
        obs_format = obs_args.get(self.arg_name, base.DEFAULT_FORMAT)
        edit = base.ValueEdit(obs_format, inf_format)
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
            edit = base.ValueEdit(obs_option, imp_option)
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
    def __init__(self):
        super().__init__(None)

    def __call__(self, dtypes, index_usage, **kwargs):
        out_map = {
                'dtypes': dtypes,
                'shape': index_usage,
                **kwargs
                }
        yield out_map

