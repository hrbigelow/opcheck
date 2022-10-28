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
from .base import ALL_DTYPES
from .oparg import *
from .error import *
from . import oparg, util, base, fgraph

"""
The generation graph (gen_graph) is constructed using NodeFuncs in this file.
Its job is to generate test examples for the op.  Will generate a set of
examples within a certain maximum edit distance of a valid example.  While all
nodes in the gen_graph produce the full set of valid values for their inputs,
certain nodes generate additional values which violate a schema constraint.
While yielding these invalid values, the node deducts from op.avail_test_edits.
and then resets it after the yield.

At the beginning of example generation, op.avail_test_edits is set by the user.
and determines the maximum edit distance that an emitted example can be from a
valid example.
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

class Unused:
    pass

class Indel(enum.Enum):
    Insert = 0
    Delete = 1

class GenFunc(NodeFunc):
    """
    A NodeFunc outfitted with 'kinds' to signal which of four roles it plays
    """
    def __init__(self, op, name=None):
        super().__init__(name)
        self.op = op

    @contextmanager
    def max_yield(self, max_val):
        old_val = self.op.max_yield_count
        self.op.max_yield_count = max_val
        try:
            yield
        finally:
            self.op.max_yield_count = old_val

    @contextmanager
    def reserve_edit(self, dist):
        doit = (dist <= self.op.avail_test_edits)
        if doit:
            self.op.avail_test_edits -= dist
        try:
            yield doit
        finally:
            if doit:
                self.op.avail_test_edits += dist

class Layout(GenFunc):
    def __init__(self, op, name):
        super().__init__(op, name)

    def __call__(self):
        num_layouts = self.op.data_formats.num_layouts()
        for i, layout in enumerate(range(num_layouts)):
            if i == self.op.max_yield_count:
                break
            yield layout

class Sig(GenFunc):
    """
    Represent a set of signatures for argument {name} corresponding to the
    available layouts. 
    """
    def __init__(self, op, name, options):
        super().__init__(op, name)
        self.options = options

    def __call__(self, layout):
        yield self.options[layout]

class SigMap(GenFunc):
    """
    Aggregate all of the :sig nodes into a map of arg_name => sig
    """
    def __init__(self):
        super().__init__(None)

    def __call__(self, **kwargs):
        sig_map = kwargs
        yield sig_map

class RankRange(GenFunc):
    """
    Produce a range of ranks for a given primary index.
    """
    def __init__(self, op, name):
        super().__init__(op, name)
        self.schema_cons = []

    def add_schema_constraint(self, cons):
        self.schema_cons.append(cons)

    def __call__(self, **index_ranks):
        # Get the initial bounds consistent with the schema
        sch_lo, sch_hi = 0, 1e10
        for cons in self.schema_cons:
            clo, chi = cons(**index_ranks)
            sch_lo = max(sch_lo, clo)
            sch_hi = min(sch_hi, chi)

        for i, rank in enumerate(range(sch_lo, sch_hi+1)):
            if i == self.op.max_yield_count:
                break
            yield rank

class RankEquiv(GenFunc):
    """
    Produce a range identical to the primary index
    """
    def __init__(self, op, name):
        super().__init__(op, name)

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

class ArgRanks(GenFunc):
    """
    Represent the induced ranks for arguments as determined by index ranks
    Parents: Ranks, Sigs
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, index_ranks, sigs):
        arg_ranks = {}
        for arg, sig in sigs.items():
            rank = sum(index_ranks[idx] for idx in sig)
            arg_ranks[arg] = rank
        yield arg_ranks

class ArgIndels(GenFunc):
    """
    In Test mode:
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, arg_ranks):
        yield {}
        num_yielded = 1
        # produce each type of indel up to a limit
        with self.reserve_edit(1) as avail:
            if not avail:
                return
            for arg, rank in arg_ranks.items():
                pos = choice(range(rank+1))
                yield { arg: (Indel.Insert, pos, 1) }
                num_yielded += 1
                if num_yielded == self.op.max_yield_count:
                    break
                pos = choice(range(rank))
                yield { arg: (Indel.Delete, pos, pos+1) }
                num_yielded += 1
                if num_yielded == self.op.max_yield_count:
                    break

class ArgMutations(GenFunc):
    """
    Test: arg => shape  (shapes are mutated or not)
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, arg_indels, index_ranks, sigs, **comp):
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
        index_dims = compute_dims(self.op, arg_ranks, index_ranks, sigs, **comp)

        num_yielded = 0
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
        num_yielded += 1

        # generate point mutations
        with self.reserve_edit(1) as avail:
            if not avail:
                return
            for arg, shape in arg_shapes.items():
                if num_yielded == self.op.max_yield_count:
                    break
                if len(shape) == 0:
                    continue
                i = choice(range(len(shape)))
                old_val = shape[i]
                new_val, alt_val = sample(range(1, max_dimsize), 2)
                val = new_val if new_val != shape[i] else alt_val
                shape[i] = val
                copy = { k: list(v) for k, v in arg_shapes.items() }

                yield copy
                num_yielded += 1
                shape[i] = old_val

class DataFormat(GenFunc):
    """
    Generate the special data_format argument, defined by the 'layout' API call
    Inference: yields None or ValueEdit
    """
    def __init__(self, op, formats, arg_name, rank_idx):
        super().__init__(op, arg_name)
        self.formats = formats
        self.arg_name = arg_name
        self.rank_idx = rank_idx

    def __call__(self, ranks, layout):
        inferred_fmt = self.formats.data_format(layout, ranks)
        num_yielded = 0
        rank = ranks[self.rank_idx]
        for alt_fmt in self.formats.all_formats():
            if num_yielded == self.op.max_yield_count:
                break
            if alt_fmt == inferred_fmt:
                yield oparg.ValueArg(alt_fmt)
                num_yielded += 1
            else:
                with self.reserve_edit(1) as avail:
                    if avail:
                        yield oparg.ValueArg(alt_fmt) 
                        num_yielded += 1

IndelMutation = namedtuple('IndelMutation', ['arg', 'delta'])

class Indels(GenFunc):
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

class MutatedArgRanks(GenFunc):
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, arg_ranks, indel):
        if self.op.generation_mode == GenMode.Inference:
            assert indel is None, 'Invariant 1 violated'
        mut_ranks = dict(arg_ranks)
        if indel is not None:
            mut_ranks[indel.arg] += indel.delta
        yield mut_ranks

class ArgShapes(GenFunc):
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

class DTypeIndiv(GenFunc):
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

    def __call__(self):
        num_yielded = 0
        for i, y in enumerate(self.valid_dtypes):
            if num_yielded == self.op.max_yield_count:
                break
            yield y
            num_yielded += 1
        with self.reserve_edit(1) as avail:
            if avail:
                if num_yielded == self.op.max_yield_count:
                    return 
                y = choice(self.invalid_dtypes)
                yield y
                num_yielded += 1

class DTypeEquiv(GenFunc):
    """
    A DType which is declared equal to another using equate_dtypes 
    Inference: yields None or a DTypesEdit
    """
    def __init__(self, op, arg_name, src_arg_name):
        super().__init__(op, arg_name)
        self.arg_name = arg_name
        self.src_arg_name = src_arg_name
        self.all_dtypes = ALL_DTYPES

    def __call__(self, src_dtype):
        num_yielded = 0
        for dtype in self.all_dtypes:
            if dtype == src_dtype:
                yield src_dtype
                num_yielded += 1
            else:
                with self.reserve_edit(1) as avail:
                    if avail:
                        yield dtype
                        num_yielded += 1
            if num_yielded == self.op.max_yield_count:
                break

class DTypesNotImpl(GenFunc):
    """
    Represents configurations that are not implemented, as declared with API
    function exclude_combos
    Inference: yields None or DTypesNotImpl
    """
    def __init__(self, op):
        super().__init__(op)
        self.exc = self.op.excluded_combos

    def __call__(self, ranks, layout, **dtypes):
        excluded = self.exc.excluded(dtypes, ranks, layout)
        # filter dtypes generated from above
        edit = 1 if excluded else 0
        with self.reserve_edit(edit) as avail:
            if avail:
                yield dtypes

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

class DataTensor(GenFunc):
    """
    Produce the (shape, dtype) combo needed to produce a tensor
    Parents: ArgShapes, DTypes
    """
    def __init__(self, op, arg_name):
        super().__init__(op, arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_shapes, dtypes):
        shape = arg_shapes[self.arg_name]
        dtype = dtypes[self.arg_name]
        arg = oparg.DataTensorArg(shape, dtype)
        yield arg

class ShapeInt(GenFunc):
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

class ShapeList(GenFunc):
    """
    Generate the current shape of the input signature
    """
    def __init__(self, op, arg_name):
        super().__init__(op, arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_shapes):
        if not isinstance(arg_shapes, dict):
            raise RuntimeError
        shape = arg_shapes[self.arg_name]
        arg = oparg.ShapeListArg(shape)
        yield arg


class ShapeTensor(GenFunc):
    """
    Generate the current shape of the input signature as a tensor
    """
    def __init__(self, op, arg_name):
        super().__init__(op, arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_muts, arg_indels, sigs, obs_shapes):
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

class Int(GenFunc):
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
        return [randint(self.lo, self.hi)]

class Options(GenFunc):
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

    def __call__(self):
        for val in self.options:
            yield oparg.ValueArg(val)
        with self.reserve_edit(1) as avail:
            if avail:
                with self.max_yield(1):
                    yield oparg.ValueArg('DUMMY')

class Args(GenFunc):
    """
    Collects all arguments as an ordered dictionary
    Parents: DataTensor, ShapeInt, ShapeList, ShapeTensor, ShapeTensor2D,
    DataFormat (if non-default), Option.
    Expect each argument to use the sub-name
    """
    def __init__(self):
        super().__init__(None)

    def __call__(self, **kwargs):
        args = kwargs
        yield args 

