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

def comp_index_clusters(op):
    """
    Produce a list of clusters of computed indices.  Each cluster itself is a
    list.  cluster assignment is based on single-linkage clustering.  Two
    computed indices are linked iff they share an input index
    """
    graph = op.dims_graph
    comp_idxs = graph.computed_indexes()
    input_idxs = graph.input_indexes()
    nidx = len(input_idxs)
    adj = np.identity(nidx).astype(np.bool)
    for cidx in comp_idxs:
        inputs = graph.get_index_inputs(cidx)
        if len(inputs) == 0:
            continue
        pos = [input_idxs.index(i) for i in inputs]
        sp = pos[0]
        for p in pos[1:]:
            adj[sp,p] = True
    while True:
        nadj = np.matmul(adj, adj.transpose())
        if np.all(nadj == adj):
            break
        adj = nadj
    # iclust[p] = cluster, where p is the index into input_idxs
    iclust = list(range(nidx))
    for p in range(nidx):
        # find position of first True
        ft = adj[p].tolist().index(True)
        cl = iclust[ft]
        iclust[p] = cl

    nclust = max(iclust) + 1
    cclust = [[] for _ in range(nclust)]
    for cidx in comp_idxs:
        inputs = graph.get_index_inputs(cidx)
        if len(inputs) == 0:
            continue
        p = input_idxs.index(inputs[0])
        cl = iclust[p]
        cclust[cl].append(cidx)
    return cclust

def less_lb(op, comp_dims, target_ival):
    """
    Returns True if any computed index values are less than the target range.
    """
    for cidx, dims in comp_dims.items():
        lo = target_ival[cidx][0]
        if any(d < lo for d in dims):
            return True
    return False

def less_ub(op, comp_dims, target_ival):
    """
    Computes comp_idxs using input_dims.  Returns True if all computed index
    values are <= target range
    """
    for cidx, dims in comp_dims.items():
        hi = target_ival[cidx][1]
        if any(hi < d for d in dims):
            return False
    # all dims are <= target range
    return True

def ival_bsearch(op, ranks, max_dimsize, comp_idxs, gen_dims, less_fn,
        pos_mode, **comp):
    """
    Find the acceptable input range for all free index inputs 
    """
    graph = op.dims_graph
    free_idxs = { i for c in comp_idxs for i in graph.get_index_inputs(c) if i
            not in gen_dims }
    ranges = {}
    for idx in free_idxs:
        rng = op.index[idx].dims_range()
        lo = max(rng.start, 1)
        hi = min(rng.stop - 1, max_dimsize)
        ranges[idx] = [lo, hi]

    free_dims = {}
    target_ival = {}
    for cidx in comp_idxs:
        if pos_mode:
            rng = op.index[cidx].dims_range()
            lo = max(rng.start, 1)
            hi = min(rng.stop - 1, max_dimsize)
        else:
            lo, hi = (-max_dimsize, -1)
        target_ival[cidx] = (lo, hi)

    while any(r[0] != r[1] for r in ranges.values()):
        for fidx in free_idxs:
            r = ranks[fidx]
            lo, hi = ranges[fidx]
            mid = (lo + hi) // 2
            free_dims[fidx] = [mid] * r
        input_dims = { **free_dims, **gen_dims }
        all_comp_dims = graph.dims(input_dims, **comp)
        comp_dims = { cidx: all_comp_dims[cidx] for cidx in comp_idxs }

        is_less = less_fn(op, comp_dims, target_ival)
        print('pos_mode: ', pos_mode, 
                ', is_less: ', is_less, 
                ', comp_dims: ', comp_dims,
                ', target_ival: ', target_ival,
                # ', gen_dims: ', gen_dims,
                ', ranges: ', ranges)
        for fidx in free_idxs:
            lo, hi = ranges[fidx]
            mid = (lo + hi) // 2
            if is_less:
                ranges[fidx][0] = mid + 1
            else:
                ranges[fidx][1] = mid

    found = { idx: r[0] for idx, r in ranges.items() }
    return found

def compute_dims(op, mut_arg_ranks, index_ranks, arg_sigs, pos_mode, **comp):
    """
    Resolve a set of all index dims consistent with {index_ranks}.  First, any
    indexes registered with add_index_generator or rank_dims_constraint will be
    computed.  The system iterates until all computed index dims are
    non-negative.
    """
    # results
    final_dims_list = []
    gen_dims_list = op.gen_indices(index_ranks)
    clusters = comp_index_clusters(op)
    max_dimsize = get_max_dimsize(op.target_nelem, mut_arg_ranks)

    # [ (idx => dims), (idx => dims), ... ] 
    # [ {'s': [1,1,1], 'd': [2,3,5]}, { 's': [2,3,1], 'd': [1,1,1] } ]
    for gen_dims in gen_dims_list:
        free_dims = {}
        for cl in clusters:
            # find the equal_range [lo, hi) such that any input values in this
            # range yield outputs in the target range
            lo = ival_bsearch(op, index_ranks, max_dimsize, cl, gen_dims,
                    less_lb, pos_mode, **comp)
            hi = ival_bsearch(op, index_ranks, max_dimsize, cl, gen_dims,
                    less_ub, pos_mode, **comp)
            print('lo: ', lo, ', hi: ', hi)
            for fidx in lo.keys():
                lb = lo[fidx]
                ub = hi[fidx]
                if lb == ub:
                    free_dims[fidx] = None
                    continue
                r = index_ranks[fidx]
                free_dims[fidx] = [randint(lb,ub) for _ in range(r)]
        if any(fdims is None for fdims in free_dims.values()):
            continue

        input_dims = { **free_dims, **gen_dims }
        comp_dims = op.dims_graph.dims(input_dims, **comp) 
        final_dims = { **input_dims, **comp_dims }
        print('final_dims: ', final_dims)

        for idx, rank in index_ranks.items():
            if idx not in final_dims:
                rng = op.index[idx].dims_range()
                lo = max(rng.start, 1)
                hi = min(rng.stop - 1, max_dimsize)
                final_dims[idx] = [randint(lo, hi) for _ in range(rank)] 
        final_dims_list.append(final_dims)

        # TODO: handle broadcasting logic
        # apply the assumption that computed indexes are either
        # component-wise or broadcasting.  secondly, assume that the
        # computed index is monotonically increasing in the values of
        # the input indices
        """
        for idx, rng in comp_ranges.items():
            dims = comp_dims[idx]
            if any(d not in rng for d in dims):
                assert False, (
                        f'Index {idx} had out-of-range dims {dims}. '
                        f'valid range is {rng}')
        """
    print(final_dims_list)
    return final_dims_list

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

def plex(res):
    """
    convert a list of individual results into one multiplexed result.
    rlist := [res, ...]
    res   := [item, ...]
    item  := int or (int, ...)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]] => [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

    [[(1,5,3),(2,6,3)], [(6,8,3),(2,6,4)], [(9,5,3),(2,6,7)], [(5,1,0),(2,1,3)]] => 
    [([1,6,9,5], [5,8,5,1], [3,3,3,0]), ([2,2,2,2], [6,6,6,1], [3,4,7,3])]
    """
    if len(res) == 0:
        return res
    item = res[0][0]
    if isinstance(item, int):
        return list(list(z) for z in zip(*res))
    elif isinstance(item, tuple):
        return [tuple(list(z) for z in zip(*c)) for c in zip(*res)]

class GenDims(NodeFunc):
    """

    """
    def __init__(self, name, func, rank_idx, yield_scalar, max_prod, pars):
        super().__init__(name)
        """
        func(*inputs, *extra_vals, max_val) yields a list of ints or a list of
        tuples
        """
        self.func = func
        self.rank_idx = rank_idx
        self.yield_scalar = yield_scalar
        self.max_prod = max_prod
        self.pars = pars

    def max_result(self, vals):
        return max(np.prod(v) if isinstance(v, tuple) else v for v in vals)

    def __call__(self, kwnode, *input_dims):
        index_ranks = kwnode[INDEX_RANKS]
        if self.yield_scalar:
            yield from self.func(*input_dims, *self.pars, self.max_prod)

        rank = index_ranks[self.rank_idx]
        res = []
        max_val = self.max_prod
        for r in range(rank):
            ins = tuple(d[r] if isinstance(d, list) else d for d in input_dims)
            vals = list(self.func(*ins, *self.pars, max_val))
            res.append(vals)
            max_item = self.max_result(vals)
            max_val //= max_item
        pres = plex(res)
        yield from pres 

class CompDims(NodeFunc):
    """
    Wrap a dims computation, performing scalar broadcasting as necessary
    """
    def __init__(self, name, func, tfunc, rank_idx, arg_names):
        super().__init__(name)
        self.func = func
        self.tfunc = tfunc
        self.rank_idx = rank_idx
        self.arg_names = arg_names

    def __call__(self, kwnode, *input_dims):
        arg_vals = tuple(kwnode[k] for k in self.arg_names)
        index_ranks = kwnode[INDEX_RANKS]
        rank = index_ranks[self.rank_idx]
        result = []
        for r in range(rank):
            ins = tuple(d[r] if isinstance(d, list) else d for d in input_dims)
            res = self.func(*ins, *arg_vals)
            result.append(res)
        yield result

class DimsInput(NodeFunc):
    # A dummy node for supplying input to the Dims graph via set_cached()
    def __init__(self):
        super().__init__()

    def __call__(self):
        return None

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
    generative dims_graph.
    """
    def __init__(self, op):
        super().__init__(op)

    def __call__(self, arg_indels, index_ranks, sigs, **comp):
        # yield negative dims version
        arg_ranks = {}
        for arg, sig in sigs.items():
            arg_ranks[arg] = sum(index_ranks[idx] for idx in sig)

        dims_input = { INDEX_RANKS: index_ranks, **comp } 
        self.op.dims_input_node.set_cached(dims_input)
        all_nodes = set(self.op.dims_graph.values())
        live_nodes = all_nodes.difference([self.op.dims_input_node])
        index_dims_list = fgraph.gen_graph_values(live_nodes, live_nodes)
        # index_dims_list = compute_dims(self.op, arg_ranks, index_ranks, sigs, False, **comp)
        for index_dims in index_dims_list:
            neg_arg_shapes = {}
            for arg, sig in sigs.items():
                shape = [ dim for idx in sig for dim in index_dims[idx] ]
                neg_arg_shapes[arg] = shape
            yield neg_arg_shapes

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
        max_dimsize = get_max_dimsize(self.op.target_nelem, mut_arg_ranks)
        index_dims_list = compute_dims(self.op, mut_arg_ranks, index_ranks,
                sigs, True, **comp)
        assert len(index_dims_list) > 0

        for index_dims in index_dims_list:
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

