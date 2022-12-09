import sys
import math
import enum
from copy import copy
from contextlib import contextmanager
from collections import namedtuple
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
from random import randint, choice, sample
from .fgraph import FuncNode as F, func_graph_evaluate, NodeFunc
from .base import ALL_DTYPES, INDEX_RANKS
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

class GenDims(NodeFunc):
    """
    Yield one or more sets of dimensions for {sig}.  If sig is a single index,
    each set will be an integer list of rank {rank_idx}.  If {yield_scalar}, an
    additional integer will be yielded, which represents a rank-agnostic, broadcasted
    dimension.

    {func} is expected to yield one or more items.  Each item is either: a (lo, hi) pair,
    if {sig} is a single index, or a tuple ((lo, hi), (lo, hi), ...), with one member for
    each index in {sig}.

    GenDims ensures that each set of dimensions total number of elements (the product)
    does not exceed {max_prod}. 

    func is called as list(func(*input_dims, *pars)) to collect all of its yields.
    It is called once per component in the rank of {rank_idx}.
    """
    def __init__(self, sig, in_sig, func, rank_idx, yield_scalar, max_prod,
            pars):
        super().__init__(sig)
        self.in_sig = in_sig
        self.func = func
        self.rank_idx = rank_idx
        self.yield_scalar = yield_scalar
        self.max_prod = max_prod
        self.pars = pars
        self.num_indexes = len(sig)

    @staticmethod
    def fill(gen):
        lo, hi = next(gen)
        lo = 0 if lo is None else lo
        hi = int(1e10) if hi is None else hi
        yield lo, hi

    def gen_dims(self, gens):
        # gens is one generator per component of the dims.  total of
        # RANK(rank_idx) generate the next batch of dims.  if single index, an
        # integer list. if multiple index, a tuple of integer lists
        comp_ranges = tuple(next(g) for g in gens)
        if self.num_indexes == 1:
            yield base.range_under_size(comp_ranges, self.max_prod) 
        else:
            dims_list = []
            for idx_range in list(zip(*comp_ranges)):
                dims = base.range_under_size(idx_range, self.max_prod)
                dims_list.append(dims)
            yield tuple(dims_list)

    def gen_dims_scalar(self, gen):
        comp_range = next(gen) 
        dims = base.range_under_size([comp_range], self.max_prod)
        if self.num_indexes == 1:
            yield dims[0]
        else:
            yield tuple(d[0] for d in dims)

    def __call__(self, index_ranks, **grouped_dims_map):
        dims_map = base.ungroup_dims(grouped_dims_map)
        input_dims = [ dims_map[idx] for idx in self.in_sig ]

        if self.yield_scalar:
            if not all(isinstance(d, int) for d in input_dims):
                raise SchemaError(
                    f'{type(self).__name__}: yield_scalar flag only supported '
                    f'for integer index inputs'
                )
            gen = self.func(*input_dims, *self.pars)
            fgen = self.fill(gen)
            yield from self.gen_dims_scalar(fgen)

        rank = index_ranks[self.rank_idx]
        gens = []
        for c in range(rank):
            ins = tuple(base.bcast_dim(dims, c) for dims in input_dims)
            gen = self.func(*ins, *self.pars)
            fgen = self.fill(gen)
            gens.append(fgen)
        
        yield from self.gen_dims(gens)

class CompDims(NodeFunc):
    """
    Wrap a dims computation, performing scalar broadcasting as necessary
    """
    def __init__(self, op, idx, in_sig, cwise, func, tfunc, rank_idx, arg_names):
        super().__init__(idx)
        self.idx = idx
        self.op = op
        self.in_sig = in_sig
        self.cwise = cwise
        self.func = func
        self.tfunc = tfunc
        self.rank_idx = rank_idx
        self.arg_names = arg_names
        self.nargs = len(arg_names)

    def safe_func(self, *args):
        try:
            return self.func(*args)
        except BaseException as ex:
            raise SchemaError(
                f'The function registered for computed index {self.sub_name} '
                f'failed on the call: func{args}\n{ex}')

    def __call__(self, index_ranks, **dims_and_args):
        plist = [ (k,v) for k,v in dims_and_args.items() ]
        rit = reversed([plist.pop() for _ in range(self.nargs)])
        arg_vals = [v for _,v in rit]
        grouped_dims_map = { k:v for k,v in plist }
        dims_map = base.ungroup_dims(grouped_dims_map)
        input_dims = [ dims_map[idx] for idx in self.in_sig ]
        # check that it is rank-consistent
        if self.op.comp_dims_mode:
            if self.cwise:
                result = []
                rank = index_ranks[self.rank_idx]
                if not base.broadcastable_to(input_dims, rank):
                    raise OpCheckInternalError(
                        f'non-broadcastable dims: {input_dims}, {self.in_sig}'
                        f', {rank}')
                for c in range(rank):
                    ins = tuple(base.bcast_dim(dims, c) for dims in input_dims)
                    res = self.safe_func(*ins, *arg_vals)
                    if isinstance(res, tuple):
                        res = list(res)
                    result.append(res)
                yield result
            else:
                result = self.safe_func(*input_dims, *arg_vals)
                yield result
        else:
            templ = self.tfunc(*input_dims, *arg_vals)
            if self.nargs > 0:
                z = zip(self.arg_names, arg_vals)
                cfg = ', '.join(f'{arg} = {val}' for arg, val in z) 
                templ += f'   [{cfg}]'
            yield templ

class DimsInput(GenFunc):
    """
    Supply a value to the dims_graph via the schema.  name can be one of:
    base.INDEX_RANKS, base.LAYOUT, or one of the arg_names
    """
    def __init__(self, op, name):
        super().__init__(op, name)
        if name == base.LAYOUT:
            self.gen_node = op._gen_node(Layout, base.LAYOUT)
        elif name == base.INDEX_RANKS:
            self.gen_node = None
        else:
            self.gen_node = op.arg_gen_nodes[name]

    def __call__(self):
        yield self.op.dims_graph_input[self.sub_name] 

class Layout(GenFunc):
    def __init__(self, op):
        super().__init__(op, base.LAYOUT)

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
    def __init__(self, op, name, sigs):
        super().__init__(op, name)
        self.sigs = sigs

    def __call__(self, layout):
        yield self.sigs[layout]

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
        sch_lo, sch_hi = 0, 10000
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
    Compute dimensions of all indices given index_ranks using the schema's
    generative dims_graph.  Yield arg_shapes, a map of arg => shape.  For
    single-index argument signatures, shape may be an integer, indicating
    rank-agnostic broadcasting shape.  Otherwise, shape is an integer list.
    """
    def __init__(self, op):
        super().__init__(op)

    def sig_shape(self, sig, index_dims, index_ranks):
        if len(sig) == 1:
            return copy(index_dims[sig])
        else:
            shape = []
            for idx in sig:
                dims = index_dims[idx]
                if isinstance(dims, int):
                    dims = [dims] * index_ranks[idx]
                shape.extend(dims)
        return shape

    def __call__(self, arg_indels, index_ranks, sigs, **comp):
        # yield negative dims version
        arg_ranks = {}
        for arg, sig in sigs.items():
            arg_ranks[arg] = sum(index_ranks[idx] for idx in sig)

        for k, v in comp.items():
            if k == base.LAYOUT:
                val = v
            else:
                val = v.value()
            self.op.dims_graph_input[k] = val

        self.op.dims_graph_input[INDEX_RANKS] = index_ranks
        all_nodes = set(self.op.dims_graph.values())
        dims_kinds = (GenDims, CompDims)
        dims_nodes = [n for n in all_nodes if isinstance(n.func, dims_kinds)]
        index_gen = fgraph.gen_graph_map(all_nodes, dims_nodes)
        index_dims_list = []
        for tup_map in index_gen:
            dims_map = {}
            for sig, tup in tup_map.items():
                if len(sig) == 1:
                    dims_map[sig] = tup
                else:
                    dims_map.update(dict(zip(sig, tup)))
            index_dims_list.append(dims_map)

        mut_arg_ranks = {}
        for arg, sig in sigs.items():
            mut_arg_ranks[arg] = arg_ranks[arg]
            indel = arg_indels.get(arg, None)
            if indel is not None:
                kind, rest = indel[0], indel[1:]
                if kind == Indel.Insert:
                    _, size = rest
                    mut_arg_ranks[arg] += size
                elif kind == Indel.Delete:
                    beg, end = rest
                    size = end - beg
                    mut_arg_ranks[arg] -= size

        # incorporate the indel
        # max_dimsize = get_max_dimsize(self.op.target_nelem, mut_arg_ranks)
        max_dimsize = 2 # very conservative, so that an insertion of 2 can only
        # increase memory by a factor of 4
        assert len(index_dims_list) > 0

        for index_dims in index_dims_list:
            num_yielded = 0
            arg_shapes = {}
            for arg, sig in sigs.items():
                shape = self.sig_shape(sig, index_dims, index_ranks)
                indel = arg_indels.get(arg, None)
                # if shape is int, it is broadcastable and cannot be indel'ed
                if indel is not None and isinstance(shape, list):
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
            max_mut_size = 5 
            with self.reserve_edit(1) as avail:
                if not avail:
                    return
                for arg, shape in arg_shapes.items():
                    if num_yielded == self.op.max_yield_count:
                        break
                    if isinstance(shape, int) or len(shape) == 0:
                        continue
                    i = choice(range(len(shape)))
                    old_val = shape[i]
                    new_val, alt_val = sample(range(1, max_mut_size), 2)
                    val = new_val if new_val != shape[i] else alt_val
                    shape[i] = val
                    mut_shape = { k: copy(v) for k, v in arg_shapes.items() }
                    yield mut_shape
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
    Generates all valid dtypes for {arg_name}, which has been declared with 
    API call valid_dtypes.  Generates up to op.max_gen_invalid_dtypes ones
    """
    def __init__(self, op, arg_name):
        super().__init__(op, arg_name)
        self.arg_name = arg_name
        self.valid_dtypes = op.dtype_rules.indiv_rules[arg_name]
        self.invalid_dtypes = tuple(t for t in ALL_DTYPES if t not in
                self.valid_dtypes)

    def __call__(self):
        yield from self.valid_dtypes
        with self.reserve_edit(1) as avail:
            if avail:
                tot = min(len(self.invalid_dtypes), self.op.dtype_err_quota)
                ys = sample(self.invalid_dtypes, tot)
                yield from ys

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
        num_err = 0
        for dtype in self.all_dtypes:
            if dtype == src_dtype:
                yield src_dtype
            else:
                with self.reserve_edit(1) as avail:
                    if avail and num_err != self.op.dtype_err_quota:
                        yield dtype
                        num_err += 1

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

