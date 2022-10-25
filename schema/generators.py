import sys
import math
import enum
from contextlib import contextmanager
from collections import Counter
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import itertools
from random import randint, choice, sample
from functools import partial
from collections import defaultdict
from .fgraph import FuncNode as F, func_graph_evaluate, NodeFunc
from .base import GenMode, ErrorInfo, GenKind
from .error import *
from . import oparg, util, base, fgraph

"""
The generation graph (gen_graph) is constructed using NodeFuncs in this file.
Each node in the graph is run in Test or Inference mode, as controlled by
op.generation_mode.

GenMode.Test: produces arg sets that are within a threshold of edit distance
set by op.avail_edits, from correct arg sets.

GenMode.Inference: produce a set of configurations which satisfy all schema
constraints, and come within op.max_edit_dist edit distance of satisfying all
'observed' quantities.  The observed quantities are supplied by the node type
ObservedValue. 

GenMode.Inference will be run starting with 0, 1, etc setting for
op.max_edit_dist.  If exactly one configuration is found with zero edit
distance from the observations, it means the operation succeeded.  Otherwise,
the graph produces a set of suggestions for inputs.  Each suggested edit will
fix the problem, but may require more than one 'edit'.

In Inference mode, the yielded values are either suggested fixes, or None, if
there is nothing to fix.  The fix can either be a specific instruction on how
to fix the input, or a symbolic entity.  The set of fixes is used to generate
instructions to the user.

In Generation mode, the yielded values are used to construct op arguments.

"""
def get_max_dimsize(target_nelem, arg_ranks):
    ranks = dict(arg_ranks)
    max_rank = max(ranks.values())
    if max_rank == 0:
        return 1
    dimsize = math.ceil(math.pow(target_nelem, 1.0 / max_rank))
    # print(arg_ranks, dimsize)
    return dimsize

def compute_dims(op, mut_arg_ranks, index_ranks, arg_sigs, **comp):
    max_dimsize = get_max_dimsize(op.target_nelem, mut_arg_ranks)
    """
    Resolve a set of all index dims consistent with {index_ranks}.  First,
    any indexes registered with add_index_generator or rank_dims_constraint
    will be computed.  Then, remaining indexes not registered with
    computed_index will be randomly generated in [1, max_dimsize].
    Finally, the computed index dims are created.  The system iterates
    until all computed index dims are non-negative.
    """
    gen_dims_list = op.gen_indices(index_ranks)
    # indexes appearing in at least one data tensor signature.  (both input
    # and return signatures) (some indexes are merely used as intermediate
    # quantities to simplify computation)
    # create deterministic order 
    sig_indexes = { idx for sig in arg_sigs.values() for idx in sig }
    sig_indexes = list(sorted(sig_indexes))

    for gen_index_dims in gen_dims_list:
        gen_indexes = ''.join(gen_index_dims.keys())

        # generated dims will not change during the below iteration
        input_dims = dict(gen_index_dims) 
        for idx in sig_indexes:
            if idx in input_dims:
                continue
            if idx in op.dims_graph.computed_indexes():
                continue
            dims = [ randint(1, max_dimsize) for _ in
                    range(index_ranks[idx]) ]
            input_dims[idx] = dims

        while True:
            comp_dims = op.dims_graph(input_dims, **comp) 

            # fix any visible computed dims which are negative
            # TODO: zero could need to be 1 for some dims.
            todo = next(((idx, c, dim) 
                for idx, dims in comp_dims.items()
                for c, dim in enumerate(dims)
                if dim < 0), None)
            if todo is None:
                index_dims = { **comp_dims, **input_dims }
                break
            comp_idx, c, dim = todo
            comp_inputs = op.dims_graph.get_index_inputs(comp_idx)
            # apply the assumption that computed indexes are either
            # component-wise or broadcasting.  secondly, assume that the
            # computed index is monotonically increasing in the values of
            # the input indices
            comp_rank = index_ranks[comp_idx]
            for input_idx in comp_inputs:
                if input_idx in gen_indexes:
                    continue
                if input_idx not in input_dims:
                    continue
                input_rank = index_ranks[input_idx]
                if input_rank == comp_rank:
                    inc = c
                elif input_rank == 1:
                    inc = 0
                else:
                    raise SchemaError(
                        f'Computed index \'{comp_idx}\' has rank '
                        f'{comp_rank} but has input index \'{input_idx}\' '
                        f'with rank {input_rank}.\n'
                        f'Computed indices must either be component-wise '
                        f'or broadcasting.')
                input_dims[input_idx][inc] += 1

        for idx, dims in index_dims.items():
            if all(d >= 0 for d in dims):
                continue
            assert False, f'Index {idx} had negative dims {dims}'
        # this is irrelevant since it has not anticipated indels yet
        # check_sizes(arg_sigs, index_dims, op.target_nelem * 10)
        return index_dims

ALL_DTYPES = (
        tf.int8, tf.int16, tf.int32, tf.int64,
        tf.uint8, tf.uint16, tf.uint32, tf.uint64,
        tf.float16, tf.float32, tf.float64,
        tf.qint8, tf.qint16, tf.qint32,
        tf.bfloat16, 
        tf.bool,
        tf.complex64, tf.complex128
        )

class Symbolic(enum.Enum):
    ValidDType = 0
    InvalidDType = 1
    ValidIndexDims = 2
    ValidArgShapes = 3
    UserFix = 4

class Unused:
    pass

class Indel(enum.Enum):
    Insert = 0
    Delete = 1


ALL_KINDS = (GenKind.InferLive, GenKind.InferShow, GenKind.TestLive,
        GenKind.TestShow)
LIVE_KINDS = (GenKind.InferLive, GenKind.TestLive)
TEST_KINDS = (GenKind.TestLive, GenKind.TestShow)


class GenFunc(NodeFunc):
    def __init__(self, kinds, name):
        super().__init__(name)
        self.kinds = kinds

class ReportNodeFunc(GenFunc):
    """
    NodeFunc which further implements user-facing reporting functions
    """
    def __init__(self, op, kinds, name=None):
        super().__init__(kinds, name)
        self.op = op

    def constraint_msg(self):
        """
        A message describing the constraint(s) defined
        """
        raise NotImplementedError

    def edit(self, op_arg, *edit_info):
        """
        Edit op_arg using edit_info to a valid state
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

class ObservedValue(GenFunc):
    """
    Node for delivering inputs to any individual rank nodes.
    This is the portal to connect the rank graph to its environment
    """
    def __init__(self, name):
        super().__init__((), name)

    def __call__(self):
        return [{}]

class Layout(ReportNodeFunc):
    def __init__(self, op, name):
        super().__init__(op, LIVE_KINDS, name)

    def __call__(self):
        all_layouts = list(range(self.op.data_formats.num_layouts()))
        yield from all_layouts

class Sig(ReportNodeFunc):
    """
    Represent a set of signatures for argument {name} corresponding to the
    available layouts. 
    """
    def __init__(self, op, name, options):
        super().__init__(op, LIVE_KINDS, name)
        self.options = options

    def __call__(self, layout):
        yield self.options[layout]

class SigMap(GenFunc):
    """
    Aggregate all of the :sig nodes into a map of arg_name => sig
    """
    def __init__(self):
        super().__init__(LIVE_KINDS, None)

    def __call__(self, **kwargs):
        sig_map = kwargs
        yield sig_map

class RankRange(ReportNodeFunc):
    """
    Produce a range of ranks for a given primary index.
    """
    def __init__(self, op, name):
        super().__init__(op, LIVE_KINDS, name)
        # self.indel_func = indel_node.func
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

        if self.op.generation_mode == GenMode.Test:
            yield from range(sch_lo, sch_hi+1)

        elif self.op.generation_mode == GenMode.Inference:
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
        else:
            raise RuntimeError('generation_mode not set')

class EquivRange(ReportNodeFunc):
    """
    Produce a range identical to the primary index
    """
    def __init__(self, op, name):
        super().__init__(op, LIVE_KINDS, name)

    def __call__(self, rank):
        yield rank

class IndexRanks(GenFunc):
    """
    Gather ranks together index ranks into one map
    Parents:  RankRange and EquivRange nodes
    """
    def __init__(self):
        super().__init__(LIVE_KINDS, None)

    def __call__(self, **ranks):
        yield ranks

class ArgIndels(ReportNodeFunc):
    """
    Produce a map of arg_name => indel, where indel is one of:
    (Insert, position, length)
    (Delete, start, end)
    """
    def __init__(self, op):
        super().__init__(op, LIVE_KINDS)

    def user_msg(self, *info):
        pass

    def edit(self, op_arg, index_dims, *info):
        kind, rest = info[0], info[1:]
        allowed_op_args = (DataTensorArg, ShapeTensorArg, ShapeListArg)
        if not isinstance(op_arg, allowed_op_args):
            raise RuntimeError(
                f'{type(self).__qualname__}: cannot edit a non-shape op_arg')

        if kind == Indel.Insert:
            spos, idx, ibeg, iend = rest
            op_arg.shape[spos:spos] = index_dims[idx][ibeg:iend]
        elif kind == Indel.Delete:
            sbeg, send = rest
            del op_arg.shape[sbeg:send]
        return op_arg

    def __call__(self, index_ranks, sigs, obs_shapes):
        arg_ranks = {}
        for arg, sig in sigs.items():
            rank = sum(index_ranks[idx] for idx in sig)
            arg_ranks[arg] = rank

        if self.op.generation_mode == GenMode.Test:
            yield {}
            # produce each type of indel up to a limit
            for arg, rank in arg_ranks.items():
                for pos in range(rank+1):
                    yield { arg: (Indel.Insert, pos, 1) }
                for pos in range(rank):
                    yield { arg: (Indel.Delete, pos, pos+1) }

        elif self.op.generation_mode == GenMode.Inference:
            """
            Produces instructions to insert part of an index's dimensions, or
            delete a subrange from a shape.
            """
            indels = {}
            total_edit = 0
            for arg, rank in arg_ranks.items():
                obs_shape = obs_shapes[arg]
                if isinstance(obs_shape, int):
                    continue
                obs_rank = len(obs_shape)
                delta = rank - obs_rank
                if delta > 0:
                    sig = sigs[arg]
                    spos = 0 # shape position coordinate
                    for idx in sig:
                        idx_rank = index_ranks[idx]
                        indels[arg] = []
                        for b in range(idx_rank - delta + 1):
                            ed = base.InsertEdit(self, spos+b, idx, b, b+delta)
                            indels[arg].append(ed)
                        spos += idx_rank

                elif delta < 0:
                    indels[arg] = []
                    for b in range(obs_rank + delta):
                        edit = base.DeleteEdit(self, b, b-delta)
                        indels[arg].append(edit)
                total_edit += abs(delta)

            with self.reserve_edit(total_edit) as avail: 
                if not avail:
                    return
                indel_args = list(indels.keys())
                for indel_combo in itertools.product(*indels.values()):
                    indel_map = dict(zip(*indel_args, *indel_combo))
                    yield indel_map

class ArgMutations(ReportNodeFunc):
    """
    In inference mode, produces a map of arg_name => point_mutation
    In generative mode, produces arg_name => shape
    point_mutation is a map of { shape_index => dim }
    """
    def __init__(self, op):
        super().__init__(op, LIVE_KINDS)

    def user_msg(self, point_muts):
        pass

    def edit(self, op_arg, point_muts):
        allowed_op_args = (DataTensorArg, ShapeTensorArg, ShapeListArg)
        if not isinstance(op_arg, allowed_op_args):
            raise RuntimeError(
                f'{type(self).__qualname__}: cannot edit a non-shape op_arg')
        for pos, dim in point_muts.items():
            op_arg.shape[pos] = dim
        return op_arg

    def __call__(self, arg_indels, index_ranks, sigs, obs_shapes, **comp):
        if self.op.generation_mode == GenMode.Test:
            # compute arg_ranks from index_ranks, sigs, arg_indels
            arg_ranks = {}
            for arg, sig in sigs.items():
                arg_ranks[arg] = sum(index_ranks[idx] for idx in sig)
                indel = arg_indels.get(arg, None)
                if indel is not None:
                    kind, rest = indel[0], indel[1:]
                    if kind == Indel.Insert:
                        _, size = rest
                        arg_ranks[arg] += size
                    elif kind == Indel.Delete:
                        beg, end = rest
                        size = end - beg
                        arg_ranks[arg] -= size
            # compute max_dimsize from arg_ranks
            max_dimsize = get_max_dimsize(self.op.target_nelem, arg_ranks)
            index_dims = compute_dims(self.op, arg_ranks, index_ranks, sigs,
                    **comp)

            arg_shapes = {}
            for arg, sig in sigs.items():
                shape = [ dim for idx in sig for dim in index_dims[idx] ]
                indel = arg_indels.get(arg, None)
                if indel is not None:
                    kind, rest = indel[0], indel[1:]
                    if kind == Indel.Insert:
                        pos, size = rest
                        ins = [randint(1, max_dimsize) for _ in range(size)]
                        shape[pos:pos] = ins
                    elif kind == Indel.Delete:
                        beg, end = rest
                        del shape[beg:end]
                arg_shapes[arg] = shape
            yield arg_shapes

            # generate point mutations
            with self.reserve_edit(1) as avail:
                if not avail:
                    return
                for arg, shape in arg_shapes.items():
                    for i in range(len(shape)):
                        old_val = shape[i]
                        new_val, alt_val = sample(range(1, max_dimsize), 2)
                        val = new_val if new_val != shape[i] else alt_val
                        shape[i] = val
                        copy = { k: list(v) for k, v in arg_shapes.items() }
                        yield copy
                        shape[i] = old_val
            # compute index_dims
            # generate non-mutated and mutated arg_shapes

        elif self.op.generation_mode == GenMode.Inference:
            # gather index versions from index_ranks, sigs, arg_indels
            # align the imputed index template with the observed shapes
            # insertions and deletions are in the direction from obs_shape ->
            # imputed template.
            idx_versions = {} # idx_versions[idx] = { dims, ... }
            for arg, shape in obs_shapes.items():
                if isinstance(shape, int):
                    continue
                mut_shape = list(shape)
                sig = sigs[arg]

                if arg_indels is None:
                    indel = None
                else:
                    indel = arg_indels.get(arg, None)

                if indel is None:
                    pos = 0
                    for idx in sig:
                        rank = index_ranks[idx]
                        usage = idx_versions.setdefault(idx, set())
                        usage.add(tuple(mut_shape[pos:pos+rank]))
                        pos += rank
                else:
                    kind, rest = indel
                    if kind == Indel.Delete:
                        del mut_shape[slice(*rest)]
                        pos = 0
                        for idx in sig:
                            rank = index_ranks[idx]
                            usage = idx_versions.setdefault(idx, set())
                            usage.add(tuple(mut_shape[pos:pos+rank]))
                            pos += rank

                    elif kind == Indel.Insert:
                        ins_idx, beg, end = rest
                        pos = 0
                        for idx in sig:
                            rank = index_ranks[idx]
                            ver = idx_versions.setdefault(idx, set())
                            if idx == ins_idx:
                                span = end - beg
                                dims = mut_shape[pos:pos+rank-span]
                                dims[beg:end] = [None] * span
                                pos += (rank-span)
                            else:
                                dims = mut_shape[pos:pos+rank]
                                pos += rank
                            ver.add(tuple(dims))

            idxs = list(idx_versions.keys())
            for dims_combo in itertools.product(*idx_versions.values()):
                imp_index_dims = dict(zip(idxs, dims_combo))
                total_edit = 0
                mutations = {} # arg => (comp => dim)
                for arg, sig in sigs.items():
                    obs_shape = obs_shapes[arg]
                    if isinstance(obs_shape, int):
                        continue
                    inp_shape = [dim for idx in sig for dim in
                            imp_index_dims[idx]]
                    for i, (a, b) in enumerate(zip(inp_shape, obs_shape)):
                        if a != b:
                            muts = mutations.setdefault(arg, {})
                            muts[i] = a
                            total_edit += 1
                with self.reserve_edit(total_edit) as avail:
                    if avail:
                        edit_map = {}
                        for arg, muts in mutations.items():
                            edit_map[arg] = base.MutateEdit(self, muts)
                        yield imp_index_dims, edit_map 

class Fixes(ReportNodeFunc):
    """
    Gathers the set of edits generated by the graph during Inference phase.
    """
    def __init__(self, op):
        kinds = (GenKind.InferLive, GenKind.InferShow)
        super().__init__(op, kinds)

    def __call__(self, arg_muts, arg_indels, dtypes_filt, sigs, **kwargs):
        edit_map = {} 
        index_dims, mutations = arg_muts
        for arg, edit in mutations.items():
            edits = edit_map.setdefault(arg, [])
            edits.append(edit)
        for arg, edit in arg_indels.items():
            edits = edit_map.setdefault(arg, [])
            edits.append(edit)
        for arg, edit in kwargs.items():
            edits = edit_map.setdefault(arg, [])
            edits.append(edit)

        fix = base.Fix(index_dims, sigs, edit_map, dtypes_filt)
        yield fix

class ArgRanks(ReportNodeFunc):
    """
    Represent the induced ranks for arguments as determined by index ranks
    Parents: Ranks, Sigs
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, ranks, sigs):
        arg_ranks = {}
        for arg, sig in sigs.items():
            rank = sum(ranks[idx] for idx in sig)
            arg_ranks[arg] = rank
        if self.op.generation_mode == GenMode.Test:
            pass
        elif self.op.generation_mode == GenMode.Inference:
            pass

        yield arg_ranks

class DataFormat(ReportNodeFunc):
    """
    Generate the special data_format argument, defined by the 'layout' API call
    """
    def __init__(self, op, formats, arg_name, rank_idx):
        super().__init__(op, LIVE_KINDS, arg_name)
        self.formats = formats
        self.arg_name = arg_name
        self.rank_idx = rank_idx

    def constraint_msg(self, obs_format, rank):
        idx_desc = self.op.index[self.rank_idx]
        msg =  f'{self.arg_name} ({obs_format}) not compatible with '
        msg += f'{idx_desc} dimensions ({rank}).  '
        msg += f'For {rank} {idx_desc} dimensions, {self.arg_name} can be '
        msg += 'TODO'
        return msg

    def __call__(self, ranks, layout, obs_args):
        inferred_fmt = self.formats.data_format(layout, ranks)

        if self.op.generation_mode == GenMode.Test:
            rank = ranks[self.rank_idx]
            for alt_fmt in self.formats.all_formats():
                if alt_fmt == inferred_fmt:
                    yield oparg.ValueArg(alt_fmt)
                else:
                    with self.reserve_edit(1) as avail:
                        if avail:
                            yield oparg.ValueArg(alt_fmt) 

        elif self.op.generation_mode == GenMode.Inference:
            obs_format = obs_args.get(self.arg_name, base.DEFAULT_FORMAT)
            if inferred_fmt == obs_format:
                yield None
            else:
                with self.reserve_edit(1) as avail:
                    if avail:
                        yield base.ValueEdit(self, self.arg_name, inferred_fmt)
        else:
            raise RuntimeError('generation_mode not set')

IndelMutation = namedtuple('IndelMutation', ['arg', 'delta'])

class Indels(ReportNodeFunc):
    """
    Represent an Indel mutation on one of the argument shapes
    Parents: ArgRanks, Sigs
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, arg_ranks, sigs, obs_shapes):
        if self.op.generation_mode == GenMode.Test:
            yield None
            if not self.available_edit(1):
                return
            definite_inds = self.op.definite_rank_indices
            arg_names = list(arg_ranks.keys())
            arg_ranks_tup = tuple(arg_ranks[n] for n in arg_names)
            for t, arg in enumerate(arg_names):
                sig = sigs[arg]
                definite_sig = any(idx in definite_inds for idx in sig)
                for delta in (-1, 1):
                    z = enumerate(arg_ranks_tup)
                    mut = tuple(r + delta if i == t else r for i, r in z)
                    # collisions with valid ranks will be filtered out
                    # later
                    if any(r < 0 for r in mut):
                        continue
                    # Mutating a rank to 1 aliases with the broadcasting
                    # behavior of an indefinite sig.
                    if not definite_sig and mut[t] == 1:
                        continue

                    # the suggestion gives the original arg and rank
                    args = (arg, arg_ranks[arg])
                    with self.new_error(self, args, 1) as avail:
                        assert avail
                        indel = IndelMutation(arg, delta)
                        yield indel

        elif self.op.generation_mode == GenMode.Inference:
            """
            All observed ranks are accounted for in RankRange nodes, which set
            errors as Indel instances.  So, any rank discrepancies between
            implied and observed ranks are already accounted for. 
            """
            yield None
        else:
            raise RuntimeError('generation_mode not set')

class MutatedArgRanks(ReportNodeFunc):
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, arg_ranks, indel):
        if self.op.generation_mode == GenMode.Inference:
            assert indel is None, 'Invariant 1 violated'
        mut_ranks = dict(arg_ranks)
        if indel is not None:
            mut_ranks[indel.arg] += indel.delta
        yield mut_ranks

class IndexDims(ReportNodeFunc):
    """
    Generate dims for indexes of ranks defined by index_ranks.  
    close to a target value.
    Parents: MutatedArgRanks, Ranks, Sigs
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, mut_arg_ranks, index_ranks, sigs, obs_shapes, **comp):

        if self.op.generation_mode == GenMode.Test:
            dims = self.compute_dims(mut_arg_ranks, index_ranks, sigs, **comp)
            yield dims

        elif self.op.generation_mode == GenMode.Inference:
            # If ranks don't all match, we won't use this suggestion anyway
            for arg, shape in obs_shapes.items():
                if isinstance(shape, int):
                    continue
                if len(shape) != mut_arg_ranks[arg]:
                    yield Symbolic.ValidIndexDims
                    return

            # usage_dims[idx][arg] = dims
            usage_dims = get_usage_dims(index_ranks, obs_shapes, sigs)
            error_inds = []
            for idx, use in usage_dims.items():
                used_dims = { tuple(d) for d in use.values() }
                if len(used_dims) > 1:
                    error_inds.append(idx)

            dist = len(error_inds)
            if dist == 0:
                index_dims = { idx: next(iter(use.values())) for idx, use in
                        usage_dims.items() }
                yield index_dims
            else:
                args = tuple(error_inds)
                with self.new_error(self, args, dist) as avail:
                    if avail:
                        yield Symbolic.ValidIndexDims
        else:
            raise RuntimeError(f'generation mode not set')

class IndexUsage(ReportNodeFunc):
    """
    Computes Index usage - needed for determining where index usages can be
    mutated non-synonymously.
    Parents: Sigs
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, sigs):
        idx_usage = defaultdict(list)
        indexes = { idx for sig in sigs.values() for idx in sig }
        for arg, sig in sigs.items():
            for idx in sig:
                if idx not in indexes:
                    continue
                idx_usage[idx].append(arg) 
        yield dict(idx_usage)

class ArgShapes(ReportNodeFunc):
    """
    Create argument shapes from sigs, index_dims, and indels
    Parents: IndexDims, Sigs, IndexUsage, Indels, MutatedArgRanks
    """
    def __init__(self, op, index_dims_obj):
        super().__init__(op)
        self.index_dims_obj = index_dims_obj

    def constraint_msg(self, *error_inds):
        phr = []
        for idx in error_inds:
            idx_desc = self.op.index[idx]
            msg = f'Index {idx} ({idx_desc}) has unequal dimensions.'
            phr.append(msg)
        msg = '\n'.join(phr)
        return msg

    @staticmethod
    def shape_indels(arg_shapes, indel, max_dimsize):
        """
        Create one set of data tensor shapes according to the {indel}.
        The indel specifies index_ranks, a changed data tensor, and a delta.
        """
        if indel is None:
            return arg_shapes

        # create the augmented shapes  
        shape = arg_shapes[indel.arg]
        if indel.delta < 0:
            # delete dimensions from the shape
            for _ in range(abs(indel.delta)):
                p = choice(range(len(shape)))
                shape.pop(p)
        else:
            for _ in range(indel.delta):
                p = choice(range(len(shape) + 1))
                new_dim = randint(1, max_dimsize)
                shape.insert(p, new_dim) 
        return arg_shapes

    def shape_mutations(self, usage_dims, max_dimsize):
        """
        Yield non-synonymous mutations of the form idx, arg, comp, new_val.
        That is, index idx component comp used in arg is mutated to new_val
        """
        for idx, usage in usage_dims.items():
            if len(usage) == 1:
                # only used in one location so a mutation would not lead to
                # inconsistent usage
                continue
            for arg, dims in usage.items():
                if len(dims) == 0:
                    continue
                c = randint(0, len(dims)-1)
                new_val, alt_val = sample(range(1, max_dimsize), 2)
                val = new_val if new_val != dims[c] else alt_val
                yield idx, arg, c, new_val
        
    def __call__(self, index_dims, sigs, idx_usage, indel, mut_arg_ranks):
        max_dimsize = get_max_dimsize(self.op.target_nelem, mut_arg_ranks)
        if index_dims == Symbolic.ValidIndexDims:
            arg_shapes = Symbolic.ValidArgShapes
        else:
            arg_shapes = {}
            for arg, sig in sigs.items():
                shape = [d for idx in sig for d in index_dims[idx]]
                arg_shapes[arg] = shape

        if self.op.generation_mode == GenMode.Test:
            assert arg_shapes != Unused
            indel_shapes = self.shape_indels(arg_shapes, indel, max_dimsize)
            yield indel_shapes 

            usage_dims = make_usage_dims(index_dims, sigs)
            pmuts = self.shape_mutations(usage_dims, max_dimsize)
            for idx, arg, comp, val in pmuts:
                # construct mutation
                old_val = usage_dims[idx][arg][comp]
                usage_dims[idx][arg][comp] = val
                # args represents the piece of information which would correct
                # this error
                # TODO: update IndexDims to match this argument info
                # args = (idx, arg, comp, old_val)
                args = (idx,)
                with self.new_error(self.index_dims_obj, args, 1) as avail:
                    if not avail:
                        continue
                    mut_shapes = make_arg_shapes(usage_dims, sigs)
                    yield mut_shapes
                usage_dims[idx][arg][comp] = old_val

        elif self.op.generation_mode == GenMode.Inference:
            assert indel is None, 'Indel should be None in Inference mode'
            yield arg_shapes
        else:
            raise RuntimeError('generation_mode not set')

class DTypeIndiv(ReportNodeFunc):
    """
    A Dtype with an individual valid set
    """
    def __init__(self, op, arg_name, valid_dtypes):
        super().__init__(op, LIVE_KINDS, arg_name)
        self.arg_name = arg_name
        self.valid_dtypes = valid_dtypes
        self.invalid_dtypes = tuple(t for t in ALL_DTYPES if t not in
                valid_dtypes)

    def constraint_msg(self, obs_dtype):
        valid_str = ', '.join(d.name for d in self.valid_dtypes)
        msg =  f'{self.arg_name}.dtype was {obs_dtype.name} but must be '
        msg += f'one of {valid_str}'
        return msg

    def __call__(self, obs_dtypes):
        if self.op.generation_mode == GenMode.Test:
            yield from self.valid_dtypes
            with self.reserve_edit(1) as avail:
                if avail:
                    yield Symbolic.InvalidDType

        elif self.op.generation_mode == GenMode.Inference:
            obs_dtype = obs_dtypes[self.arg_name]
            if obs_dtype in self.valid_dtypes:
                yield None
            else:
                with self.reserve_edit(1) as avail:
                    if avail:
                        yield base.DTypesEdit(self, arg_name)
        else:
            raise RuntimeError(f'generation mode not set')

class DTypeEquiv(ReportNodeFunc):
    """
    A DType which is declared equal to another using equate_dtypes 
    """
    def __init__(self, op, arg_name, src_arg_name):
        super().__init__(op, LIVE_KINDS, arg_name)
        self.arg_name = arg_name
        self.src_arg_name = src_arg_name
        self.all_dtypes = ALL_DTYPES

    def constraint_msg(self, obs_dtype, obs_src_dtype):
        msg =  f'{self.arg_name}.dtype ({obs_dtype.name}) not equal to '
        msg += f'{self.src_arg_name}.dtype ({obs_src_dtype.name}).  '
        msg += f'dtypes of \'{self.arg_name}\' and \'{self.src_arg_name}\' '
        msg += f'must be equal.'
        return msg

    def __call__(self, obs_dtypes, src_dtype):
        if self.op.generation_mode == GenMode.Test:
            for dtype in self.all_dtypes:
                if dtype == src_dtype:
                    yield src_dtype
                else:
                    with self.reserve_edit(1) as avail:
                        if avail:
                            yield dtype

        elif self.op.generation_mode == GenMode.Inference:
            obs_dtype = obs_dtypes[self.arg_name]
            obs_src_dtype = obs_dtypes[self.src_arg_name]
            if obs_dtype == obs_src_dtype:
                yield None
            else:
                with self.reserve_edit(1) as avail:
                    if avail:
                        yield base.DTypesEdit(self, self.arg_name)
        else:
            raise RuntimeError('generation_mode not set')

class DTypesFilter(ReportNodeFunc):
    """
    Represents configurations that are not implemented, as declared with API
    function exclude_combos
    """
    def __init__(self, op):
        super().__init__(op, LIVE_KINDS)
        self.exc = self.op.excluded_combos

    def user_msg(self):
        # highlight all dtypes, the rank-bearing index, and data_format
        pass

    def __call__(self, ranks, layout, obs_dtypes, **dtypes):
        excluded = self.exc.excluded(dtypes, ranks, layout)

        if self.op.generation_mode == GenMode.Test:
            # filter dtypes generated from above
            edit = 1 if excluded else 0
            with self.reserve_edit(edit) as avail:
                if avail:
                    yield dtypes

        elif self.op.generation_mode == GenMode.Inference:
            excluded = self.exc.excluded(obs_dtypes, ranks, layout)
            if not excluded:
                yield None  
            else:
                with self.reserve_edit(1) as avail:
                    if avail:
                        yield base.DTypesFiltered(self)
        else:
            raise RuntimeError('generation_mode not set')

class Rank(ReportNodeFunc):
    """
    Generate the rank of a given signature
    """
    def __init__(self, op, sig):
        super().__init__(op, sig)
        self.sig = sig

    def __call__(self, ranks_map):
        rank = sum(ranks_map[s] for s in self.sig)
        yield rank

def make_usage_dims(index_dims, arg_sigs):
    usage_dims = {}
    for idx, dims in index_dims.items():
        usage_dims[idx] = {}
        for arg, sig in arg_sigs.items():
            if idx in sig:
                usage_dims[idx][arg] = list(dims)
    return usage_dims

# produces usage_dims[idx][arg] = dims
def get_usage_dims(index_ranks, arg_shapes, arg_sigs):
    usage_dims = {}
    for arg, shape in arg_shapes.items():
        sig = arg_sigs[arg]
        it = base.shape_iter(shape)
        for idx in sig:
            r = index_ranks[idx]
            dims = base.shape_nextn(it, r)
            usage = usage_dims.setdefault(idx, {})
            usage[arg] = dims
    return usage_dims

def make_arg_shapes(usage_dims, arg_sigs):
    arg_shapes = {}
    for arg, sig in arg_sigs.items():
        arg_shapes[arg] = [dim for idx in sig for dim in usage_dims[idx][arg]]
    return arg_shapes

def check_sizes(arg_sigs, index_dims, max_nelem):
    for arg, sig in arg_sigs.items():
        shape = [dim for idx in sig for dim in index_dims[idx]]
        nelem = np.prod(shape)
        if nelem < max_nelem:
            continue
        raise SchemaError(
            f'Generated shape for argument \'{arg}\' was {shape} with '
            f'{nelem} elements, exceeding the maximum allowed {max_nelem}')

class TupleElement(NodeFunc):
    """
    Expect a tuple and return a particular element
    """
    def __init__(self, index):
        super().__init__()
        self.index = index

    def __call__(self, tup):
        yield tup[self.index]

class GetArgShapes(TupleElement):
    def __init__(self):
        super().__init__(1)

class DataTensor(GenFunc):
    """
    Produce the (shape, dtype) combo needed to produce a tensor
    Parents: ArgShapes, DTypes
    """
    def __init__(self, arg_name):
        kinds = (GenKind.TestLive,)
        super().__init__(kinds, arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_shapes, dtypes):
        dtype = dtypes[self.arg_name]
        shape = arg_shapes[self.arg_name]
        arg = oparg.DataTensorArg(shape, dtype)
        yield arg

class ShapeInt(GenFunc):
    """
    Produce an integer value representing the shape of arg_name.  Returns the
    empty list if the shape is inconsistent with a non-broadcasted integer.
    """
    def __init__(self, arg_name):
        kinds = (GenKind.TestLive,)
        super().__init__(kinds, arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_shapes):
        shape = arg_shapes[self.arg_name]
        if len(shape) != 1:
            return []
        else:
            arg = oparg.ShapeIntArg(shape[0])
            yield arg

class ShapeList(GenFunc):
    """
    Generate the current shape of the input signature
    """
    def __init__(self, arg_name):
        kinds = (GenKind.TestLive,)
        super().__init__(kinds, arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_shapes):
        if not isinstance(arg_shapes, dict):
            raise RuntimeError
        shape = arg_shapes[self.arg_name]
        arg = oparg.ShapeListArg(shape)
        return [arg]

class ShapeTensor(GenFunc):
    """
    Generate the current shape of the input signature as a tensor
    """
    def __init__(self, arg_name):
        kinds = (GenKind.TestLive,)
        super().__init__(kinds, arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_shapes):
        shape = arg_shapes[self.arg_name]
        arg = oparg.ShapeTensorArg(shape)
        yield arg

class ShapeTensor2D(GenFunc):
    """
    Generate a 2D tensor from dims and a list of signatures.  Since it is
    impossible to have input with non-rectangular shape, this node will produce
    no output if shape is non-rectangular.
    """
    def __init__(self, arg_name, num_rows):
        kinds = (GenKind.TestLive,)
        super().__init__(kinds, arg_name)
        self.arg_name = arg_name
        self.num_rows = num_rows

    def __call__(self, arg_shapes):
        names = [ f'{self.arg_name}.{i}' for i in range(self.num_rows) ]
        rows = [ arg_shapes[n] for n in names ]
        if len({ len(r) for r in rows }) != 1:
            # unequal length rows
            return []
        arg = oparg.ShapeTensor2DArg(rows)
        yield arg


class Int(GenFunc):
    def __init__(self, lo, hi):
        kinds = (GenKind.TestLive,)
        super().__init__(kinds, f'{lo}-{hi}')
        if lo is None:
            self.lo = -sys.maxsize - 1
        else:
            self.lo = lo
        if hi is None:
            self.hi = sys.maxsize
        else:
            self.hi = hi

    def __call__(self):
        return [randint(self.lo, self.hi)]

class Options(ReportNodeFunc):
    """
    Represent a specific set of options known at construction time
    """
    def __init__(self, op, name, options):
        super().__init__(op, LIVE_KINDS, name)
        self.arg_name = name
        try:
            iter(options)
        except TypeError:
            raise SchemaError(
                f'{type(self).__qualname__}: \'options\' argument must be '
                f'iterable.  Got {type(options)}')
        self.options = options

    def edit(self, op_arg, new_val):
        if not isinstance(op_arg, ValueArg):
            raise RuntimeError(
                f'{type(self).__qualname__}: must be a ValueArg instance')
        op_arg.val = new_val
        return op_arg

    def __call__(self, argmap):
        if self.op.generation_mode == GenMode.Test:
            for val in self.options:
                yield oparg.ValueArg(val)
            with self.reserve_edit(1) as avail:
                if avail:
                    yield oparg.ValueArg('DUMMY')

        elif self.op.generation_mode == GenMode.Inference:
            option = argmap[self.arg_name]
            if option in self.options: 
                yield oparg.ValueArg(option)
            else:
                with self.reserve_edit(1) as avail:
                    if avail:
                        for val in self.options:
                            yield oparg.ValueArg(val)
        else:
            raise RuntimeError('generation_mode not set')

class Args(GenFunc):
    """
    Collects all arguments as an ordered dictionary
    Parents: DataTensor, ShapeInt, ShapeList, ShapeTensor, ShapeTensor2D,
    DataFormat (if non-default), Option.
    Expect each argument to use the sub-name
    """
    def __init__(self):
        super().__init__(TEST_KINDS, None)

    def __call__(self, **kwargs):
        args = kwargs
        yield args 

