import sys
import math
import enum
from contextlib import contextmanager
from collections import namedtuple
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
from random import randint, choice, sample
from .fgraph import FuncNode as F, func_graph_evaluate, NodeFunc
from .base import ALL_DTYPES
from .oparg import *
from .error import *
from . import oparg, base, fgraph

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

def compute_dims(op, mut_arg_ranks, index_ranks, arg_sigs, positive_mode,
        **comp):
    max_dimsize = get_max_dimsize(op.target_nelem, mut_arg_ranks)
    """
    Resolve a set of all index dims consistent with {index_ranks}.  First,
    any indexes registered with add_index_generator or rank_dims_constraint
    will be computed.  Then, remaining indexes not registered with
    computed_index will be randomly generated in [1, max_dimsize].
    Finally, the computed index dims are created.  The system iterates
    until all computed index dims are non-negative.
    """

    # [ (idx => dims), (idx => dims), ... ] 
    gen_dims_list = op.gen_indices(index_ranks)
    # indexes appearing in at least one data tensor signature.  (both input
    # and return signatures) (some indexes are merely used as intermediate
    # quantities to simplify computation)
    # create deterministic order 
    sig_indexes = { idx for sig in arg_sigs.values() for idx in sig }
    sig_indexes = list(sorted(sig_indexes))

    # e.g. gen_dims_list = 
    # [ {'s': [1,1,1], 'd': [2,3,5]}, { 's': [2,3,1], 'd': [1,1,1] } ]
    for gen_index_dims in gen_dims_list:
        gen_indexes = ''.join(gen_index_dims.keys())

        # generated dims will not change during the below iteration
        input_dims = dict(gen_index_dims) 
        for idx in sig_indexes:
            if idx in input_dims:
                continue
            if idx in op.dims_graph.computed_indexes():
                continue
            dims = [ randint(1, max_dimsize) for _ in range(index_ranks[idx]) ]
            input_dims[idx] = dims

        comp_idxs = op.dims_graph.computed_indexes()
        if positive_mode:
            comp_rngs = { idx: range(randint(1, max_dimsize), 10000) for idx in
                    comp_idxs }
        else:
            comp_rngs = { idx: range(-10000, -1) for idx in comp_idxs }

        while True:
            comp_dims = op.dims_graph.dims(input_dims, **comp) 

            # fix any visible computed dims which are negative
            # TODO: zero could need to be 1 for some dims.
            todo = next(((idx, c, dim) 
                for idx, dims in comp_dims.items()
                for c, dim in enumerate(dims)
                if dim not in comp_rngs[idx]), None)
            if todo is None:
                index_dims = { **comp_dims, **input_dims }
                break

            # the first detected out-of-bounds index, and the component
            # which is out of bounds
            oob_idx, oob_comp, oob_dim = todo
            delta = 1 if oob_dim < comp_rngs[oob_idx].start else -1

            comp_inputs = op.dims_graph.get_index_inputs(oob_idx)
            # apply the assumption that computed indexes are either
            # component-wise or broadcasting.  secondly, assume that the
            # computed index is monotonically increasing in the values of
            # the input indices
            comp_rank = index_ranks[oob_idx]
            for input_idx in comp_inputs:
                if input_idx in gen_indexes:
                    continue
                if input_idx not in input_dims:
                    continue
                input_rank = index_ranks[input_idx]
                if input_rank == comp_rank:
                    inc = oob_comp
                elif input_rank == 1:
                    inc = 0
                else:
                    raise SchemaError(
                        f'Computed index \'{oob_idx}\' has rank '
                        f'{comp_rank} but has input index \'{input_idx}\' '
                        f'with rank {input_rank}.\n'
                        f'Computed indices must either be component-wise '
                        f'or broadcasting.')
                input_dims[input_idx][inc] += delta
                """
                if not positive_mode:
                    print(f'{input_idx} {inc} '
                            f'filter: {input_dims["f"]} '
                            f'{input_dims[input_idx][inc]} '
                            f'{comp_dims[oob_idx][oob_comp]}')
                """

                # it's possible this approach fails and we cannot find any
                # inputs which yield the desired computed output dimensions
                if input_dims[input_idx][inc] < 0:
                    return None

        for idx, dims in comp_dims.items():
            if any(d not in comp_rngs[idx] for d in dims):
                assert False, (
                        f'Index {idx} had out-of-range dims {dims}. '
                        f'valid range is {comp_rngs[idx]}')
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
        sch_lo, sch_hi = 0, 100000
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

        # compute negative dims
        index_dims = compute_dims(self.op, arg_ranks, index_ranks, sigs, False,
                **comp)
        if index_dims is not None:
            neg_arg_shapes = {}
            for arg, sig in sigs.items():
                shape = [ dim for idx in sig for dim in index_dims[idx] ]
                neg_arg_shapes[arg] = shape
            yield neg_arg_shapes

        # incorporate the indel
        index_dims = compute_dims(self.op, arg_ranks, index_ranks, sigs, True,
                **comp)
        assert index_dims is not None

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

class DTypeIndiv(GenFunc):
    """
    A Dtype with an individual valid set.
    Test mode yields a dtype or symbolic
    Inference:  yields None or a DTypesEdit
    """
    def __init__(self, op, arg_name):
        super().__init__(op, arg_name)
        self.arg_name = arg_name
        self.valid_dtypes = op.dtype_rules.indiv_rules[arg_name]
        self.invalid_dtypes = tuple(t for t in ALL_DTYPES if t not in
                self.valid_dtypes)

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
    def __init__(self, op, arg_name):
        super().__init__(op, arg_name)
        self.arg_name = arg_name
        self.src_arg_name = op.dtype_rules.equate_rules[arg_name]
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
        self.rules = self.op.dtype_rules

    def __call__(self, ranks, layout, **dtypes):
        matched_rule = self.rules.matched_rule(dtypes, ranks, layout)
        # filter dtypes generated from above
        edit = 0 if matched_rule is None else 1
        with self.reserve_edit(edit) as avail:
            if avail:
                yield dtypes

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
            arg = oparg.IntArg(shape[0])
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

class RankInt(NodeFunc):
    """
    Generate an argument which is an integer defining the rank of a signature
    """
    def __init__(self, arg_name, sig):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.sig = sig

    def __call__(self, index_ranks):
        rank = sum(index_ranks[idx] for idx in self.sig)
        arg = oparg.IntArg(rank)
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

