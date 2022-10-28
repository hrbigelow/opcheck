import itertools
from copy import copy
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

    def user_msg(self):
        """
        A message describing the constraint(s) defined
        """
        raise NotImplementedError

    def edit(self, op_arg, *edit_info):
        """
        Edit op_arg using edit_info to a valid state.  
        """
        raise NotImplementedError

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
    Produce a range of ranks for a given primary index.
    """
    def __init__(self, op, name):
        super().__init__(op, name)
        self.schema_cons = []

    def add_schema_constraint(self, cons):
        self.schema_cons.append(cons)

    def __call__(self, obs_shapes, sigs, **index_ranks):
        # Get the initial bounds consistent with the schema
        sch_lo, sch_hi = 0, 1e10
        for cons in self.schema_cons:
            clo, chi = cons(**index_ranks)
            sch_lo = max(sch_lo, clo)
            sch_hi = min(sch_hi, chi)

        idx = self.sub_name

        # Narrow the schema sch_lo, sch_hi interval based on observed shapes
        test_lo, test_hi = sch_lo, sch_hi
        cost = [0] * (sch_hi+1)
        final_shapes = {}
        for arg, obs_shape in obs_shapes.items():
            if isinstance(obs_shape, int):
                # an integer shape is rank-agnostic, so doesn't define any
                # rank-constraint
                continue
            sig = sigs[arg]
            pri_sig = sorted(self.op.equiv_index[idx] for idx in sig)
            if idx not in pri_sig:
                continue
            prev_rank = sum(index_ranks.get(i, 0) for i in pri_sig)
            obs_rank = len(obs_shape)
            todo_inds = tuple(k for k in pri_sig if k not in index_ranks)
            target = obs_rank - prev_rank
            if len(todo_inds) == 1:
                clo, chi = target, target
                final_shapes[arg] = prev_rank
            else:
                clo, chi = 0, target
            for i in range(sch_lo, sch_hi+1):
                dist = max(max(clo - i, 0), max(i - chi, 0))
                cost[i] += dist
            # print('cost: ', cost)

        for i in range(sch_lo, sch_hi+1):
            c = cost[i]
            if c == 0:
                yield i
            else:
                if len(final_shapes) == 0:
                    with self.reserve_edit(c) as avail:
                        if avail:
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
    In Test mode:
    In Inference mode: arg => InsertEdit or DeleteEdit
    """
    def __init__(self, op):
        super().__init__(op)

    def user_msg(self, *info):
        pass

    def edit(self, shape, *info):
        kind, rest = info[0], info[1:]
        if kind == Indel.Insert:
            spos, size = rest
            shape[spos:spos] = [None] * size 
        elif kind == Indel.Delete:
            sbeg, send = rest
            del shape[sbeg:send]
        return shape 

    def __call__(self, index_ranks, sigs, obs_shapes):
        arg_ranks = {}
        for arg, sig in sigs.items():
            rank = sum(index_ranks[idx] for idx in sig)
            arg_ranks[arg] = rank

        """
        Produces instructions to insert part of an index's dimensions, or
        delete a subrange from a shape.  
        """
        indels = {} # arg => [edit, ...]
        total_edit = 0
        for arg, rank in arg_ranks.items():
            edits = indels.setdefault(arg, [])
            obs_shape = obs_shapes[arg]
            sig = sigs[arg]
            if isinstance(obs_shape, int):
                continue
            obs_rank = len(obs_shape)
            delta = rank - obs_rank
            if delta == 0:
                edit = base.ShapeEdit(obs_shape, sig, index_ranks)
                edits.append(edit)

            elif delta > 0:
                spos = 0 # shape position coordinate
                for idx in sig:
                    idx_rank = index_ranks[idx]
                    for b in range(idx_rank - delta + 1):
                        edit = base.ShapeEdit(obs_shape, sig, index_ranks)
                        args = (Indel.Insert, spos+b, idx, b, b+delta)
                        edit.add_indel(self, delta, args)
                        edits.append(edit)
                    spos += idx_rank

            else:
                for b in range(obs_rank + delta):
                    edit = base.ShapeEdit(obs_shape, sig, index_ranks)
                    args = (Indel.Delete, b, b-delta)
                    edit.add_indel(self, abs(delta), args)
                    edits.append(edit)
            total_edit += abs(delta)

        with self.reserve_edit(total_edit) as avail: 
            if not avail:
                return
            indel_args = list(indels.keys())
            for indel_combo in itertools.product(*indels.values()):
                arg_edits = dict(zip(indel_args, indel_combo))
                yield arg_edits # arg => ShapeEdit

class ArgMutations(ReportNodeFunc):
    """
    Test: arg => shape  (shapes are mutated or not)
    Inference: index_dims, arg => MutateEdit
    """
    def __init__(self, op):
        super().__init__(op)

    def user_msg(self, point_muts):
        pass

    def edit(self, shape, point_muts):
        for pos, dim in point_muts.items():
            shape[pos] = dim
        return shape

    def __call__(self, arg_edits):
        # gather index versions from index_ranks, sigs, arg_indels
        # align the imputed index template with the observed shapes
        # insertions and deletions are in the direction from obs_shape ->
        # imputed template.
        idx_versions = {} # idx_versions[idx] = { dims, ... }

        for arg, edit in arg_edits.items():
            usage = edit.idx_dims()
            for idx, dims in usage.items():
                ver = idx_versions.setdefault(idx, set())
                ver.add(tuple(dims))

        idxs = list(idx_versions.keys())
        for dims_combo in itertools.product(*idx_versions.values()):
            imp_index_dims = dict(zip(idxs, dims_combo))
            mut_arg_edits = copy(arg_edits)
            total_edit = 0
            for arg, edit in mut_arg_edits.items():
                usage = edit.idx_dims()
                mutations = {}
                for idx, obs_dims in usage.items():
                    imp_dims = imp_index_dims[idx]
                    muts = {}
                    for i, (obs, imp) in enumerate(zip(obs_dims, imp_dims)):
                        if obs != imp:
                            muts[i] = imp
                    if len(muts) != 0:
                        mutations[idx] = muts
                        total_edit += len(muts)
                edit.add_point_mut(self, mutations)
            with self.reserve_edit(total_edit) as avail:
                if avail:
                    yield mut_arg_edits

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

    def user_msg(self, obs_format, rank):
        idx_desc = self.op.index[self.rank_idx]
        msg =  f'{self.arg_name} ({obs_format}) not compatible with '
        msg += f'{idx_desc} dimensions ({rank}).  '
        msg += f'For {rank} {idx_desc} dimensions, {self.arg_name} can be '
        msg += 'TODO'
        return msg

    def edit(self, op_arg, new_val):
        op_arg.val = new_val
        return op_arg

    def __call__(self, ranks, layout, obs_args):
        inferred_fmt = self.formats.data_format(layout, ranks)

        obs_format = obs_args.get(self.arg_name, base.DEFAULT_FORMAT)
        if inferred_fmt == obs_format:
            yield base.ValueEdit(self.arg_name, obs_format, obs_format)
        else:
            with self.reserve_edit(1) as avail:
                if avail:
                    yield base.ValueEdit(self.arg_name, obs_format,
                            inferred_fmt)

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

    def edit(self, op_arg, new_val):
        op_arg.val = new_val
        return op_arg

    def __call__(self, obs_args):
        option = obs_args[self.arg_name]
        if option in self.options: 
            yield base.ValueEdit(self.arg_name, option, option)
        else:
            with self.reserve_edit(1) as avail:
                if avail:
                    for val in self.options:
                        yield base.ValueEdit(self.arg_name, option, val)

class DTypeIndiv(ReportNodeFunc):
    """
    A Dtype with an individual valid set.
    Test mode yields a dtype or symbolic
    Inference:  yields None or a DTypesEdit
    """
    def __init__(self, op, arg_name, valid_dtypes):
        super().__init__(op, arg_name)
        self.arg_name = arg_name
        self.valid_dtypes = valid_dtypes
        self.invalid_dtypes = tuple(t for t in ALL_DTYPES if t not in
                valid_dtypes)

    def user_msg(self, obs_dtype):
        valid_str = ', '.join(d.name for d in self.valid_dtypes)
        msg =  f'{self.arg_name}.dtype was {obs_dtype.name} but must be '
        msg += f'one of {valid_str}'
        return msg

    def __call__(self, obs_dtypes):
        obs_dtype = obs_dtypes[self.arg_name]
        if obs_dtype in self.valid_dtypes:
            yield base.DTypesEdit(obs_dtype, 0)
        else:
            with self.reserve_edit(1) as avail:
                if avail:
                    yield base.DTypesEdit(obs_dtype, 1)

class DTypeEquiv(ReportNodeFunc):
    """
    A DType which is declared equal to another using equate_dtypes 
    Inference: yields None or a DTypesEdit
    """
    def __init__(self, op, arg_name, src_arg_name):
        super().__init__(op, arg_name)
        self.arg_name = arg_name
        self.src_arg_name = src_arg_name
        self.all_dtypes = ALL_DTYPES

    def user_msg(self, obs_dtype, obs_src_dtype):
        msg =  f'{self.arg_name}.dtype ({obs_dtype.name}) not equal to '
        msg += f'{self.src_arg_name}.dtype ({obs_src_dtype.name}).  '
        msg += f'dtypes of \'{self.arg_name}\' and \'{self.src_arg_name}\' '
        msg += f'must be equal.'
        return msg

    def __call__(self, obs_dtypes, src_edit):
        obs_dtype = obs_dtypes[self.arg_name]
        obs_src_dtype = obs_dtypes[self.src_arg_name]
        if obs_dtype == obs_src_dtype:
            yield base.DTypesEdit(obs_dtype, 0)
        else:
            with self.reserve_edit(1) as avail:
                if avail:
                    yield base.DTypesEdit(obs_dtype, 1)

class DTypesNotImpl(ReportNodeFunc):
    """
    Represents configurations that are not implemented, as declared with API
    function exclude_combos
    Inference: yields None or DTypesNotImpl
    """
    def __init__(self, op):
        super().__init__(op)
        self.exc = self.op.excluded_combos

    def user_msg(self):
        # highlight all dtypes, the rank-bearing index, and data_format
        pass

    def __call__(self, ranks, layout, **edits):
        total_edit = sum(e.cost() for e in edits.values())
        if total_edit == 0:
            dtypes = { arg: e.orig_value for arg, e in edits.items() }
            excluded = self.exc.excluded(dtypes, ranks, layout)
        else:
            excluded = True

        if not excluded:
            yield base.NotImplEdit(self, 0)
        else:
            # use a cost of 1, because there is no notion of a set of dtypes
            # being "more wrong" than another.  they are either right or wrong
            with self.reserve_edit(1) as avail:
                if avail:
                    yield base.NotImplEdit(self, 1)

class DataTensor(ReportNodeFunc):
    """
    Produces oparg.DataTensorReport
    """
    def __init__(self, op, arg_name):
        super().__init__(op, arg_name)
        self.arg_name = arg_name

    def edit(self, shape_edit, dtype_edit):
        shape = shape_edit.apply()
        dtype = dtype_edit.apply()
        return oparg.DataTensorArg(shape, dtype)

    def __call__(self, shape_edits, *dtype_parent):
        shape_edit = shape_edits.get(self.arg_name, None)
        dtype_edit = dtype_parent[0] # hack due to awkward graph construction
        rep = oparg.DataTensorReport(self, shape_edit, dtype_edit) 
        yield rep

class ShapeList(ReportNodeFunc):
    """
    Generate the current shape of the input signature
    """
    def __init__(self, op, arg_name):
        super().__init__(op, arg_name)
        self.arg_name = arg_name

    def edit(self, shape_edit):
        shape = shape_edit.apply()
        return oparg.ShapeListArg(shape)

    def __call__(self, shape_edits):
        shape_edit = shape_edits.get(self.arg_name, None)
        rep = oparg.ShapeListReport(self, shape_edit)
        yield rep

class Args(NodeFunc):
    """
    Collects all arguments
    """
    def __init__(self, arg_order):
        super().__init__(None)
        self.arg_order = arg_order

    def __call__(self, not_impl, **kwargs):
        tups = []
        for arg in self.arg_order:
            if arg in kwargs:
                tups.append((arg, kwargs[arg]))
        fix = base.Fix(tups, not_impl)
        yield fix

