import pdb
import sys
import math
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
from .base import GenMode, ErrorInfo, EditSuggestion
from .error import *
from . import oparg, util, base, fgraph

"""
The inventory graph (inv_graph) is constructed using NodeFuncs in this file.
Each node in the graph is run in one of three GenModes (Inventory, Test,
Inference) as controlled by op.generation_mode.

GenMode.Inventory:  produces a list of input configurations that satisfy all
schema constraints

GenMode.Test: produces the same set as GenMode.Inventory, but also an
additional set which violate exactly one type of schema constraint (though may
violate multiple constraints of the same type).  These are also marked with the
appropriate expected status.

GenMode.Inference: produce a set of configurations which satisfy all schema
constraints, and come within op.max_edit_dist edit distance of satisfying all
'observed' quantities.  The observed quantities are supplied by the node type
ObservedValue. 

GenMode.Inference will be run starting with 0, 1, etc setting for
op.max_edit_dist.  If exactly one configuration is found with zero edit
distance from the observations, it means the operation succeeded.  Otherwise,
the graph produces a set of suggestions for inputs.  Each suggested edit will
fix the problem, but may require more than one 'edit'.

During Inference mode, the actual node values are only intermediate
computations.  The only information collected after graph evaluation is the
TestErrorClass, plus any information associated with it.

Invariants:

1 - In Inventory mode, the yielded value must be valid
2 - In Inference mode, the yielded value must either be valid, or Unused
3 - In Inference mode, the difference between yielded value and observed value
    is pushed to an error stack
4 - value(op.errors) + op.avail_edits is a constant.

"""
ALL_DTYPES = (
        tf.int8, tf.int16, tf.int32, tf.int64,
        tf.uint8, tf.uint16, tf.uint32, tf.uint64,
        tf.float16, tf.float32, tf.float64,
        tf.qint8, tf.qint16, tf.qint32,
        tf.bfloat16, 
        tf.bool,
        tf.complex64, tf.complex128
        )

class Unused:
    pass

class ReportNodeFunc(NodeFunc):
    """
    NodeFunc which further implements user-facing reporting functions
    """
    def __init__(self, op, name=None):
        super().__init__(name)
        self.op = op

    def constraint_msg(self):
        """
        A message describing the constraint(s) defined
        """
        raise NotImplemented

    def pass_through(self):
        return self.op.avail_edits == -100

    @contextmanager
    def set_pass_through(self, args, dist):
        try:
            ei = ErrorInfo(self, args, dist)
            self.op.errors.append(ei)
            edits = self.op.avail_edits
            self.op.avail_edits = -100
            yield
        finally:
            self.op.errors.pop()
            self.op.avail_edits = edits

    @contextmanager
    def new_error(self, func_obj, args, dist):
        doit = (dist <= self.op.avail_edits)
        if doit: 
            ei = ErrorInfo(func_obj, args, dist)
            self.op.errors.append(ei)
            self.op.avail_edits -= dist
        try:
            yield doit
        finally:
            if doit:
                self.op.errors.pop()
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

class RankRange(ReportNodeFunc):
    """
    Produce a range of ranks for a given primary index.
    """
    def __init__(self, op, name, indel_node):
        super().__init__(op, name)
        self.indel_func = indel_node.func
        self.schema_cons = []

    def add_schema_constraint(self, cons):
        self.schema_cons.append(cons)

    def __call__(self, obs_shapes, sigs, **index_ranks):
        if self.pass_through():
            yield
            return
        # Get the initial bounds consistent with the schema
        sch_lo, sch_hi = 0, 1e10
        for cons in self.schema_cons:
            clo, chi = cons(**index_ranks)
            sch_lo = max(sch_lo, clo)
            sch_hi = min(sch_hi, chi)

        if self.op.generation_mode == GenMode.Inventory:
            yield from range(sch_lo, sch_hi+1)

        elif self.op.generation_mode == GenMode.Test:
            yield from range(sch_lo, sch_hi+1)

        elif self.op.generation_mode == GenMode.Inference:
            idx = self.sub_name

            # Narrow the schema sch_lo, sch_hi interval based on observed shapes
            test_lo, test_hi = sch_lo, sch_hi
            cost = [0] * (sch_hi+1)
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
                else:
                    clo, chi = 0, target
                for i in range(sch_lo, sch_hi+1):
                    if i not in range(clo, chi+1):
                        cost[i] += 1

            for i in range(sch_lo, sch_hi+1):
                c = cost[i]
                if c == 0:
                    yield i
                else:
                    with self.new_error(self.indel_func, (), c) as avail:
                        if avail:
                            yield i
        else:
            raise RuntimeError('generation_mode not set')

class EquivRange(ReportNodeFunc):
    """
    Produce a range identical to the primary index
    """
    def __init__(self, op, name):
        super().__init__(op, name)

    def __call__(self, rank):
        if self.pass_through():
            yield
            return
        else:
            yield rank

class ArgRanks(ReportNodeFunc):
    """
    Represent the induced ranks for arguments as determined by index ranks
    Parents: Ranks, Sigs
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, ranks, sigs):
        if self.pass_through():
            yield
            return
        arg_ranks = {}
        for arg, sig in sigs.items():
            rank = sum(ranks[idx] for idx in sig)
            arg_ranks[arg] = rank
        yield arg_ranks

IndelMutation = namedtuple('IndelMutation', ['arg', 'delta'])

class Indels(ReportNodeFunc):
    """
    Represent an Indel mutation on one of the argument shapes
    Parents: ArgRanks, Sigs
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, arg_ranks, sigs, obs_shapes):
        if self.pass_through():
            yield
            return
        if self.op.generation_mode == GenMode.Inventory:
            yield None
        elif self.op.generation_mode == GenMode.Test:
            yield None
            with self.new_error(self, (), 1) as avail:
                if not avail:
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
                        # Mutating a rank to 1 aliases with the
                        # broadcasting behavior of an indefinite sig.
                        if not definite_sig and mut[t] == 1:
                            continue
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
        if self.pass_through():
            yield
            return
        if self.op.generation_mode == GenMode.Inference:
            assert indel is None, 'Invariant 1 violated'
        mut_ranks = dict(arg_ranks)
        if indel is not None:
            mut_ranks[indel.arg] += indel.delta
        yield mut_ranks

class ArgRankHash(ReportNodeFunc):
    """
    Identifies a specific configuration of final ranks.  Used to remove
    synonymous rank mutations which would collide with a non-mutated test case.

    Parents: MutatedArgRanks, Layout
    """
    def __init__(self, op):
        super().__init__(op)
        self.arg_names = []

    def add_arg_name(self, arg_name):
        self.arg_names.append(arg_name)

    def __call__(self, mut_arg_ranks, layout):
        if self.pass_through():
            yield
            return
        ranks = (mut_arg_ranks[k] for k in self.arg_names)
        yield hash((*ranks, layout))

def get_max_dimsize(target_nelem, arg_ranks):
    ranks = dict(arg_ranks)
    max_rank = max(ranks.values())
    if max_rank == 0:
        return 1
    dimsize = math.ceil(math.pow(target_nelem, 1.0 / max_rank))
    # print(arg_ranks, dimsize)
    return dimsize

class IndexDims(ReportNodeFunc):
    """
    Generate dims for indexes of ranks defined by index_ranks.  
    close to a target value.
    Parents: MutatedArgRanks, Ranks, Sigs
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, mut_arg_ranks, index_ranks, sigs, obs_shapes, **comp):
        if self.pass_through():
            yield
            return

        if self.op.generation_mode in (GenMode.Inventory, GenMode.Test):
            dims = self.compute_dims(mut_arg_ranks, index_ranks, sigs, **comp)
            yield dims

        elif self.op.generation_mode == GenMode.Inference:
            # If ranks don't all match, we won't use this suggestion anyway
            for arg, shape in obs_shapes.items():
                if isinstance(shape, int):
                    continue
                if len(shape) != mut_arg_ranks[arg]:
                    yield Unused
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
            elif dist <= self.op.avail_edits:
                args = tuple(error_inds)
                with self.set_pass_through(args, dist):
                    yield
                    return
            else:
                return
        else:
            raise RuntimeError(f'generation mode not set')

    def compute_dims(self, mut_arg_ranks, index_ranks, arg_sigs, **comp):
        max_dimsize = get_max_dimsize(self.op.target_nelem, mut_arg_ranks)
        """
        Resolve a set of all index dims consistent with {index_ranks}.  First,
        any indexes registered with add_index_generator or rank_dims_constraint
        will be computed.  Then, remaining indexes not registered with
        computed_index will be randomly generated in [1, max_dimsize].
        Finally, the computed index dims are created.  The system iterates
        until all computed index dims are non-negative.
        """
        gen_dims_list = self.op.gen_indices(index_ranks)
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
                if idx in self.op.dims_graph.computed_indexes():
                    continue
                dims = [ randint(1, max_dimsize) for _ in
                        range(index_ranks[idx]) ]
                input_dims[idx] = dims

            while True:
                comp_dims = self.op.dims_graph(input_dims, **comp) 

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
                comp_inputs = self.op.dims_graph.get_index_inputs(comp_idx)
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
            # check_sizes(arg_sigs, index_dims, self.op.target_nelem * 10)
            return index_dims

class IndexUsage(ReportNodeFunc):
    """
    Computes Index usage - needed for determining where index usages can be
    mutated non-synonymously.
    Parents: Sigs
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, sigs):
        if self.pass_through():
            yield
            return

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
        if self.pass_through():
            yield
            return

        max_dimsize = get_max_dimsize(self.op.target_nelem, mut_arg_ranks)
        if index_dims == Unused:
            arg_shapes = Unused
        else:
            arg_shapes = {}
            for arg, sig in sigs.items():
                shape = [d for idx in sig for d in index_dims[idx]]
                arg_shapes[arg] = shape

        if self.op.generation_mode == GenMode.Inventory:
            assert indel is None, 'Indel should be none in Inventory mode'
            yield arg_shapes

        elif self.op.generation_mode == GenMode.Test:
            assert arg_shapes != Unused
            indel_shapes = self.shape_indels(arg_shapes, indel, max_dimsize)
            yield indel_shapes 

            usage_dims = make_usage_dims(index_dims, sigs)
            pmuts = self.shape_mutations(usage_dims, max_dimsize)
            for idx, arg, comp, val in pmuts:
                # construct mutation
                old_val = usage_dims[idx][arg][comp]
                usage_dims[idx][arg][comp] = val
                args = (usage_dims, idx)
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
        super().__init__(op, arg_name)
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
        if self.pass_through():
            yield
            return

        if self.op.generation_mode == GenMode.Inventory:
            yield from self.valid_dtypes

        elif self.op.generation_mode == GenMode.Test:
            yield from self.valid_dtypes
            for dtype in self.invalid_dtypes:
                args = (dtype,)
                with self.new_error(self, args, 1) as avail:
                    if avail:
                        yield dtype

        elif self.op.generation_mode == GenMode.Inference:
            obs_dtype = obs_dtypes[self.arg_name]
            if obs_dtype in self.valid_dtypes:
                yield obs_dtype
                return
            if self.op.avail_edits == 0:
                return
            args = (obs_dtype,)
            with self.set_pass_through(args, 1):
                yield
        else:
            raise RuntimeError(f'generation mode not set')

class DTypeEquiv(ReportNodeFunc):
    """
    A DType which is declared equal to another using equate_dtypes 
    """
    def __init__(self, op, arg_name, src_arg_name):
        super().__init__(op, arg_name)
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
        if self.pass_through():
            yield
            return

        if self.op.generation_mode == GenMode.Inventory:
            yield src_dtype

        elif self.op.generation_mode == GenMode.Test:
            for dtype in self.all_dtypes:
                if dtype == src_dtype:
                    yield src_dtype
                else:
                    args = (dtype, src_dtype)
                    with self.new_error(self, args, 1) as avail:
                        if avail:
                            yield dtype

        elif self.op.generation_mode == GenMode.Inference:
            obs_dtype = obs_dtypes[self.arg_name]
            obs_src_dtype = obs_dtypes[self.src_arg_name]
            if obs_dtype == obs_src_dtype:
                yield obs_dtype
            elif self.op.avail_edits == 0:
                return
            else:
                args = (obs_dtype, obs_src_dtype)
                with self.set_pass_through(args, 1):
                    yield
        else:
            raise RuntimeError('generation_mode not set')

class DTypesNotImplemented(ReportNodeFunc):
    """
    Represents configurations that are not implemented, as declared with API
    function exclude_combos
    """
    def __init__(self, op):
        super().__init__(op)
        self.exc = self.op.excluded_combos

    def constraint_msg(self):
        pass

    def __call__(self, ranks, layout, **dtypes):
        if self.pass_through():
            yield
            return

        excluded = self.exc.excluded(dtypes, ranks, layout)

        if self.op.generation_mode == GenMode.Inventory:
            if not excluded:
                yield dtypes

        elif self.op.generation_mode == GenMode.Test:
            if not excluded:
                yield dtypes
            else:
                with self.new_error(self, (), 1) as avail:
                    if avail:
                        yield dtypes

        elif self.op.generation_mode == GenMode.Inference:
            if not excluded:
                yield dtypes
            elif self.op.avail_edits > 0: 
                # TODO: create meaningful arguments
                args = ()
                with self.set_pass_through(args, 1):
                    yield
                    return
            else:
                return
        else:
            raise RuntimeError('generation_mode not set')

class TestErrorClass(NodeFunc):
    """
    Retrieve the current test error class from the schema
    Parents: none
    """
    def __init__(self, op):
        super().__init__()
        self.op = op

    def __call__(self):
        yield EditSuggestion(*self.op.errors)

class IndexRanks(NodeFunc):
    """
    Gather ranks together index ranks into one map
    Parents:  RankRange and EquivRange nodes
    """
    def __init__(self):
        super().__init__()

    def __call__(self, **ranks):
        yield ranks

class Rank(ReportNodeFunc):
    """
    Generate the rank of a given signature
    """
    def __init__(self, op, sig):
        super().__init__(op, sig)
        self.sig = sig

    def __call__(self, ranks_map):
        if self.pass_through():
            yield
            return
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

class DataTensor(NodeFunc):
    """
    Produce the (shape, dtype) combo needed to produce a tensor
    Parents: ArgShapes, DTypes
    """
    def __init__(self, arg_name):
        super().__init__(arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_shapes, dtypes):
        dtype = dtypes[self.arg_name]
        shape = arg_shapes[self.arg_name]
        arg = oparg.DataTensorArg(shape, dtype)
        yield arg

class ShapeInt(NodeFunc):
    """
    Produce an integer value representing the shape of arg_name.  Returns the
    empty list if the shape is inconsistent with a non-broadcasted integer.
    """
    def __init__(self, arg_name):
        super().__init__(arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_shapes):
        shape = arg_shapes[self.arg_name]
        if len(shape) != 1:
            return []
        else:
            arg = oparg.ShapeIntArg(shape[0])
            yield arg

class ShapeList(NodeFunc):
    """
    Generate the current shape of the input signature
    """
    def __init__(self, arg_name):
        super().__init__(arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_shapes):
        if not isinstance(arg_shapes, dict):
            raise RuntimeError
        shape = arg_shapes[self.arg_name]
        arg = oparg.ShapeListArg(shape)
        return [arg]

class ShapeTensor(NodeFunc):
    """
    Generate the current shape of the input signature as a tensor
    """
    def __init__(self, arg_name):
        super().__init__(arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_shapes):
        shape = arg_shapes[self.arg_name]
        arg = oparg.ShapeTensorArg(shape)
        yield arg

class ShapeTensor2D(NodeFunc):
    """
    Generate a 2D tensor from dims and a list of signatures.  Since it is
    impossible to have input with non-rectangular shape, this node will produce
    no output if shape is non-rectangular.
    """
    def __init__(self, arg_name, num_rows):
        super().__init__(arg_name)
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

class SigMap(NodeFunc):
    """
    Aggregate all of the :sig nodes into a map of arg_name => sig
    """
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, **kwargs):
        sig_map = kwargs
        return [sig_map]

class Layout(ReportNodeFunc):
    def __init__(self, op, name):
        super().__init__(op, name)

    def __call__(self):
        if self.pass_through():
            yield
            return
        all_layouts = list(range(self.op.data_formats.num_layouts()))
        yield from all_layouts

class DataFormat(ReportNodeFunc):
    """
    Generate the special data_format argument, defined by the 'layout' API call
    """
    def __init__(self, op, formats, arg_name, rank_idx):
        super().__init__(op, arg_name)
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
        if self.pass_through():
            yield
            return

        inferred_df = self.formats.data_format(layout, ranks)
        if self.op.generation_mode == GenMode.Inventory:
            yield inferred_df

        elif self.op.generation_mode == GenMode.Test:
            rank = ranks[self.rank_idx]
            for alt_fmt in self.formats.all_formats():
                if alt_fmt == inferred_df:
                    yield alt_fmt
                else:
                    args = (alt_fmt, rank)
                    with self.new_error(self, args, 1) as avail:
                        if avail:
                            yield alt_fmt

        elif self.op.generation_mode == GenMode.Inference:
            obs_format = obs_args.get(self.arg_name, base.DEFAULT_FORMAT)
            if inferred_df == obs_format:
                yield obs_format
            else:
                args = (obs_format, ranks[self.rank_idx])
                with self.new_error(self, args, 1) as avail:
                    if avail:
                        yield inferred_df
        else:
            raise RuntimeError('generation_mode not set')

class Sig(ReportNodeFunc):
    """
    Represent a set of signatures for argument {name} corresponding to the
    available layouts. 
    """
    def __init__(self, op, name, options):
        super().__init__(op, name)
        self.options = options

    def __call__(self, layout):
        if self.pass_through():
            yield
            return
        else:
            yield self.options[layout]

class Int(NodeFunc):
    def __init__(self, lo, hi):
        super().__init__(f'{lo}-{hi}')
        if lo is None:
            self.lo = -sys.maxsize - 1
        else:
            self.lo = lo
        if hi is None:
            self.hi = sys.maxsize
        else:
            self.hi = hi

    def __call__(self):
        if self.pass_through():
            yield
            return
        else:
            return [randint(self.lo, self.hi)]

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

    def __call__(self, argmap):
        if self.pass_through():
            yield
            return

        if self.op.generation_mode == GenMode.Inventory:
            yield from self.options

        elif self.op.generation_mode == GenMode.Test:
            yield from self.options
            with self.new_error(self, (), 1) as avail:
                if not avail:
                    return
                yield 'DUMMY'
        elif self.op.generation_mode == GenMode.Inference:
            option = argmap[self.arg_name]
            if option in self.options: 
                yield option
            else:
                with self.new_error(self, (), 1) as avail:
                    if avail:
                        yield from self.options
        else:
            raise RuntimeError('generation_mode not set')

class Args(NodeFunc):
    """
    Collects all arguments as an ordered dictionary
    Parents: DataTensor, ShapeInt, ShapeList, ShapeTensor, ShapeTensor2D,
    DataFormat (if non-default), Option.
    Expect each argument to use the sub-name
    """
    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        args = kwargs
        yield args 

