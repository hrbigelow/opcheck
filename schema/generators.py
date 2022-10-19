import pdb
import sys
import math
from contextlib import contextmanager
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import itertools
from random import randint, choice, sample
from functools import partial
from collections import defaultdict
from .fgraph import FuncNode as F, func_graph_evaluate, NodeFunc
from .base import GenMode
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

    @contextmanager
    def new_error(self, edits, node_instance=None):
        if node_instance is None:
            node_instance = self
        doit = (edits <= self.op.avail_edits)
        if doit: 
            self.op.errors.append(node_instance)
            self.op.avail_edits -= edits
        try:
            yield doit
        finally:
            if doit:
                self.op.errors.pop()
                self.op.avail_edits += edits

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
        self.indel_node = indel_node
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

        if self.op.generation_mode == GenMode.Inventory:
            yield from range(sch_lo, sch_hi+1)
        elif self.op.generation_mode == GenMode.Test:
            yield from range(sch_lo, sch_hi+1)
            # print(f'{self.sub_name}: [{sch_lo}, {sch_hi}]')
            """
            with self.new_error(1) as avail:
                # TODO: maybe add additional cost?
                if not avail:
                    return
                yield sch_hi+1
                if sch_lo > 0:
                    yield sch_lo-1
            """
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
                    with self.new_error(c, self.indel_node.func) as avail:
                        if avail:
                            yield i
        else:
            raise RuntimeError('generation_mode not set')

class RankRangeBck(NodeFunc):
    """
    Produce a range of ranks for a given index.  Takes the intersection of
    legal values for each constraint
    """
    def __init__(self, op, name):
        super().__init__(name)
        self.op = op
        self.schema_cons = []
        self.observ_cons = []

    def add_schema_constraint(self, cons):
        self.schema_cons.append(cons)

    def add_observ_constraint(self, cons):
        self.observ_cons.append(cons)

    def __call__(self, obs_shapes, sigs, **index_ranks):
        """
        1. Produces the "schema interval" [lo, hi] which is the intersection of
        all schema constraint intervals.

        2. If enabled, each observational constraint is evaluated to yield an
        "error-padded" interval [max(0, clo - err), chi + err] where [clo, chi]
        is the observational constraint interval, and err is the maximum
        tolerated error.

        The final interval becomes the intersection of the schema interval with
        all error-padded intervals.  Each position inside the padding incurs a
        cost.  Any non-zero cost causes the op.error_status to be set while
        yielding that value.
        """
        constraint_args = { 'shapes': obs_shapes, 'sigs': sigs }

        # Get the initial bounds consistent with the schema
        lo, hi = 0, 1e10
        for cons in self.schema_cons:
            args = tuple(constraint_args[arg] for arg in cons.get_argnames())
            clo, chi = cons(*args, **index_ranks)
            lo = max(lo, clo)
            hi = min(hi, chi)

        if self.op.generation_mode == GenMode.Inventory:
            yield from range(lo, hi+1)
        elif self.op.generation_mode == GenMode.Test:
            yield from range(lo, hi+1)
            if self.op.test_error_cls is None:
                # TODO: maybe fine-grain this error?
                self.op.test_error_cls = NoMatchingRanks
                yield hi+1
                if lo > 0:
                    yield lo-1
                self.op.test_error_cls = None
        elif self.op.generation_mode == GenMode.Inference:
            # each observation will further limit the range to [clo, chi]
            cost = [0] * (hi+1)
            margin = 1 # this is enough to discover an under-specified range
            for cons in self.observ_cons:
                args = tuple(constraint_args[arg] for arg in cons.get_argnames())
                clo, chi = cons(*args, **index_ranks)
                lpad = clo - 1
                if lpad in range(lo, hi+1):
                    cost[lpad] += 1
                rpad = chi + 1
                if rpad in range(lo, hi+1):
                    cost[rpad] += 1
            for i in range(lo, hi+1):
                if self.op.cur_edit_dist + cost[i] <= self.op.max_edit_dist:
                    self.op.cur_edit_dist += cost[i]
                    yield i
                    self.op.cur_edit_dist -= cost[i]
        else:
            raise RuntimeError('generation_mode not set')

class EquivRange(NodeFunc):
    """
    Produce a range identical to the primary index
    """
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, rank):
        yield rank

class ArgRanks(NodeFunc):
    """
    Represent the induced ranks for arguments as determined by index ranks
    Parents: Ranks, Sigs
    """
    def __init__(self):
        super().__init__()

    def __call__(self, ranks, sigs):
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
        if self.op.generation_mode == GenMode.Inventory:
            yield None
        elif self.op.generation_mode == GenMode.Test:
            yield None
            with self.new_error(1) as avail:
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
        if self.op.generation_mode == GenMode.Inference:
            assert indel is None, 'Invariant 1 violated'
        mut_ranks = dict(arg_ranks)
        if indel is not None:
            mut_ranks[indel.arg] += indel.delta
        yield mut_ranks

class ArgRankHash(NodeFunc):
    """
    Identifies a specific configuration of final ranks.  Used to remove
    synonymous rank mutations which would collide with a non-mutated test case.

    Parents: MutatedArgRanks, Layout
    """
    def __init__(self):
        super().__init__()
        self.arg_names = []

    def add_arg_name(self, arg_name):
        self.arg_names.append(arg_name)

    def __call__(self, mut_arg_ranks, layout):
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

def check_sizes(arg_shapes, max_nelem):
    for arg, shape in arg_shapes.items():
        nelem = np.prod(shape)
        if nelem < max_nelem:
            continue
        raise SchemaError(
            f'Generated shape for argument \'{arg}\' was {shape} with '
            f'{nelem} elements, exceeding the maximum allowed {max_nelem}')

class IndexDims(ReportNodeFunc):
    """
    Generate dims for indexes of ranks defined by index_ranks.  
    close to a target value.
    Parents: MutatedArgRanks, Ranks, Sigs
    """
    def __init__(self, op):
        super().__init__(op)

    def shape_mutations(self, mut_arg_ranks, index_dims, sigs):
        """
        Mutate individual dims of correct shapes to create IndexDims errors.
        Only mutate indices which have multiple usage.
        """
        idx_usage = defaultdict(list)
        indexes = { idx for sig in sigs.values() for idx in sig }
        for arg, sig in sigs.items():
            for idx in sig:
                if idx not in indexes:
                    continue

        max_dimsize = get_max_dimsize(self.op.target_nelem, mut_arg_ranks)
        results = []

        for idx, args in idx_usage.items():
            if len(index_dims[idx]) == 0:
                continue
            if len(args) == 1:
                continue
            for mut_arg in args:
                arg_shapes = defaultdict(list)
                for arg, sig in sigs.items():
                    # create a shape for arg, possibly mutated
                    for usage in sig:
                        snip = list(index_dims[usage])
                        if arg == mut_arg and idx == usage:
                            c = randint(0, len(snip)-1)
                            new_val, alt_val = sample(range(1, max_dimsize), 2)
                            val = new_val if new_val != snip[c] else alt_val
                            yield (arg, c, val)
                        arg_shapes[arg].extend(snip)
                results.append(dict(arg_shapes))
        return results 

    def __call__(self, mut_arg_ranks, index_ranks, sigs, obs_shapes, **comp):
        if self.op.generation_mode == GenMode.Inventory:
            dims = self.compute_dims(mut_arg_ranks, index_ranks, sigs, **comp)
            yield dims
        elif self.op.generation_mode == GenMode.Test:
            dims = self.compute_dims(mut_arg_ranks, index_ranks, sigs, **comp)
            yield dims
            with self.new_error(1) as avail:
                if not avail:
                    return
                point_muts = self.shape_mutations(mut_arg_ranks, dims, sigs)
                for arg, comp, val in point_muts:
                    point_mut_shapes = dict(arg_shapes)
                    point_mut_shapes[arg][comp] = val
                    yield point_mut_shapes

        elif self.op.generation_mode == GenMode.Inference:
            # If ranks don't all match, we won't use this suggestion anyway
            """
            Produce a set of suggested index_dims such that the argument shapes
            implied by the suggestion are within  available edit distance from
            the observed shapes.
            """
            for arg, shape in obs_shapes.items():
                if isinstance(shape, int):
                    continue
                if len(shape) != mut_arg_ranks[arg]:
                    yield Unused
                    return

            # idx => { dims, dims, ... } distinct 
            idx_shapes = {} # 
            idx_count = defaultdict(int)
            for arg, shape in obs_shapes.items():
                sig = sigs[arg]
                it = base.shape_iter(shape)
                for idx in sig:
                    r = index_ranks[idx]
                    dims = base.shape_nextn(it, r)
                    tmap = idx_shapes.setdefault(idx, {})
                    tmap.setdefault(tuple(dims), 0)
                    tmap[tuple(dims)] += 1
                    idx_count[idx] += 1

            inds, map_list = zip(*idx_shapes.items())
            list_list = [ list(ml.items()) for ml in map_list ]
            totals = tuple(idx_count[idx] for idx in inds)
            # one dims_set is: ( (dims, ct), (dims, ct), ... )
            # inds is ( idx, idx, ... )
            for dims_set in itertools.product(*list_list):
                dist = sum(t - ds[1] for t, ds in zip(totals, dims_set))
                z = zip(inds, dims_set)
                index_dims = { i: list(ds[0]) for i, ds in z }
                comp_dims = self.op.dims_graph(index_dims, **comp) 
                # TODO: check comp_dims are within bounds
                index_dims.update(comp_dims)
                if dist == 0:
                    yield index_dims
                else:
                    with self.new_error(dist) as avail:
                        if not avail:
                            return
                        yield index_dims
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

class IndexUsage(NodeFunc):
    """
    Computes Index usage - needed for determining where index usages can be
    mutated non-synonymously.
    Parents: Sigs
    """
    def __init__(self):
        super().__init__()

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
    def __init__(self, op):
        super().__init__(op)

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

    def __call__(self, index_dims, sigs, idx_usage, indel, mut_arg_ranks):
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

    def __call__(self, obs_dtypes):
        if self.op.generation_mode == GenMode.Inventory:
            yield from self.valid_dtypes
        elif self.op.generation_mode == GenMode.Test:
            yield from self.valid_dtypes
            with self.new_error(1) as avail:
                if avail:
                    yield from self.invalid_dtypes
        elif self.op.generation_mode == GenMode.Inference:
            # get observation
            # print(obs_dtypes)
            obs_dtype = obs_dtypes[self.arg_name]
            if obs_dtype in self.valid_dtypes:
                yield obs_dtype
            else:
                with self.new_error(1) as avail:
                    if not avail:
                        return
                    yield from self.valid_dtypes
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

    def __call__(self, obs_dtypes, src_dtype):
        if self.op.generation_mode == GenMode.Inventory:
            yield src_dtype
        elif self.op.generation_mode == GenMode.Test:
            yield src_dtype # leave error alone
            with self.new_error(1) as avail:
                if not avail:
                    return
                for dtype in self.all_dtypes:
                    if dtype != src_dtype:
                        yield dtype
        elif self.op.generation_mode == GenMode.Inference:
            # get the observation.  
            eq = (obs_dtypes[self.arg_name] == obs_dtypes[self.src_arg_name])
            if eq:
                yield src_dtype
            else:
                with self.new_error(1) as avail:
                    if avail:
                        yield src_dtype
        else:
            raise RuntimeError('generation_mode not set')

class DTypesNotImplemented(ReportNodeFunc):
    """
    Represents configurations that are not implemented, declared with API
    function exclude_dtypes.  {fields} is a tuple containing members of:
    - names of data tensors
    - one-letter index names
    - the base.LAYOUT constant

    See api::exclude_dtypes for more detail
    """
    def __init__(self, op):
        super().__init__(op)
        self.dtypes = []
        self.dtype_names = []
        self.indexes = []
        self.layout = None
        self.exclude = []

    def add_dtype_node(self, node_name, tensor_name):
        self.dtypes.append(node_name)
        self.dtype_names.append(tensor_name)

    def add_index(self, idx):
        self.indexes.append(idx)

    def add_layout(self):
        self.layout = fgraph.node_name(Layout, base.LAYOUT)

    def add_config(self, dtypes, index_ranks, layout):
        self.exclude.append((dtypes, index_ranks, layout))

    def is_excluded(self, dtypes, ranks, layout):
        for exc_dtypes, exc_ranks, exc_layout in self.exclude:
            zd = zip(exc_dtypes, dtypes)
            zr = zip(exc_ranks, ranks)
            if (all(sd is None or sd == d for sd, d in zd) and
                    all(sr is None or sr == r for sr, r in zr) and
                    (exc_layout is None or exc_layout == layout)):
                return True
        return False

    def get_dtype_map(self, dtypes):
        return { n: t for n, t in zip(self.dtype_names, dtypes) }

    def __call__(self, obs_dtypes, **gen_input):
        # gen_input should include:
        # ge.DTypeIndiv, ge.DTypeEquiv
        # ge.RankRange(idx) for idx in self.indexes
        # ge.Layout 
        # print(gen_input)
        dtypes = tuple(gen_input[tn] for tn in self.dtypes)
        ranks = tuple(gen_input[idx] for idx in self.indexes)
        if self.layout is not None:
            layout = gen_input[self.layout]
        else:
            layout = None

        is_ex = self.is_excluded(dtypes, ranks, layout)
        dtypes_map = self.get_dtype_map(dtypes)

        if self.op.generation_mode == GenMode.Inventory:
            if is_ex:
                return
            else:
                yield dtypes_map 
        elif self.op.generation_mode == GenMode.Test:
            if is_ex:
                with self.new_error(1) as avail:
                    if not avail:
                        return
                    yield dtypes_map 
            else:
                yield dtypes_map 
        elif self.op.generation_mode == GenMode.Inference:
            # TODO: Fix this
            if is_ex:
                return
            else:
                # get observed dtypes
                obs_map = dict(zip(self.dtype_names, obs_dtypes.values()))
                yield obs_map
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
        yield list(self.op.errors)

class IndexRanks(NodeFunc):
    """
    Gather ranks together index ranks into one map
    Parents:  RankRange and EquivRange nodes
    """
    def __init__(self):
        super().__init__()

    def __call__(self, **ranks):
        # print(repr(ranks))
        yield ranks

class Rank(NodeFunc):
    """
    Generate the rank of a given signature
    """
    def __init__(self, sig):
        super().__init__(sig)
        self.sig = sig

    def __call__(self, ranks_map):
        rank = sum(ranks_map[s] for s in self.sig)
        yield rank

class Inventory(NodeFunc):
    """
    Generate inventory and test cases for the op.
    """
    def __init__(self, op):
        super().__init__()
        self.op = op

    def __call__(self):
        self.op.obs_dtypes.set_cached(None)
        self.op.obs_shapes.set_cached(None)
        self.op.obs_layout.set_cached(None)
        yield from fgraph.gen_graph_values(
                self.op.inv_live_nodes,
                self.op.inv_output_nodes)

# Specifies a set of data tensor shapes, in which one tensor has an insertion
# or deletion of a dimension relative to a valid set.

def make_indel_mutations(index_ranks_list, arg_sigs, definite_inds):
    """
    returns a list of IndelMutations.  These are used to generate
    NoMatchingRanks errors 
    """
    arg_names = list(arg_sigs.keys())
    num_arg = len(arg_names)
    # arg rank values tuple => (index_ranks, changed_arg, delta)
    valid_ranks = set()
    for index_ranks in index_ranks_list:
        arg_ranks = get_arg_ranks(index_ranks, arg_sigs)
        arg_ranks_tup = tuple(arg_ranks[n] for n in arg_names)
        valid_ranks.add(arg_ranks_tup)

    indels = []
    for index_ranks in index_ranks_list:
        arg_ranks = get_arg_ranks(index_ranks, arg_sigs)
        arg_ranks_tup = tuple(arg_ranks[n] for n in arg_names)
        for t, arg in enumerate(arg_names):
            sig = arg_sigs[arg]
            definite_sig = any(idx in definite_inds for idx in sig)
            for delta in (-1, 1):
                z = enumerate(arg_ranks_tup)
                mut_arg_ranks = tuple(r + delta if i == t else r for i, r in z)
                # Sometimes, a mutation can collide with a valid rank.
                # These would manifest as an IndexUsageError rather than
                # NoMatchingRanks.  So, skip them here.
                if mut_arg_ranks in valid_ranks:
                    continue
                if any(r < 0 for r in mut_arg_ranks):
                    continue

                # Mutating a rank to 1 aliases with the broadcasting behavior
                # of an indefinite sig.
                if not definite_sig and mut_arg_ranks[t] == 1:
                    continue

                indel = IndelMutation(index_ranks, arg_names[t], delta)
                indels.append(indel)
    return indels 

def get_arg_shapes(index_dims, arg_sigs):
    arg_shapes = defaultdict(list)
    for arg, sig in arg_sigs.items():
        for idx in sig:
            dims = index_dims[idx]
            arg_shapes[arg].extend(dims)
    return dict(arg_shapes)

def shape_indels(index_dims, arg_sigs, indel, max_dimsize):
    """
    Create one set of data tensor shapes according to the {indel}.
    The indel specifies index_ranks, a changed data tensor, and a delta.

    Dimension sizes are found such that the highest-rank (post-mutated) tensor
    will have in the ball-park of {target_nelem} elements.
    """
    # create the augmented shapes.  if changed_arg and rank_delta are None,
    # no augmentation is performed.
    arg_shapes = get_arg_shapes(index_dims, arg_sigs)
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

def shape_mutations(index_dims, arg_sigs, idx_usage, max_dimsize):
    """
    Mutate individual dims of correct shapes to create IndexUsageErrors.
    Only mutate indices which have multiple usage.
    """
    results = []
    for idx, args in idx_usage.items():
        if len(index_dims[idx]) == 0:
            continue
        if len(args) == 1:
            continue
        for mut_arg in args:
            arg_shapes = defaultdict(list)
            for arg, sig in arg_sigs.items():
                # create a shape for arg, possibly mutated
                for usage in sig:
                    snip = list(index_dims[usage])
                    if arg == mut_arg and idx == usage:
                        c = randint(0, len(snip)-1)
                        new_val, alt_val = sample(range(1, max_dimsize), 2)
                        snip[c] = new_val if new_val != snip[c] else alt_val
                    arg_shapes[arg].extend(snip)
            results.append(dict(arg_shapes))
    return results 

def check_sizes(arg_sigs, index_dims, max_nelem):
    for arg, sig in arg_sigs.items():
        shape = [dim for idx in sig for dim in index_dims[idx]]
        nelem = np.prod(shape)
        if nelem < max_nelem:
            continue
        raise SchemaError(
            f'Generated shape for argument \'{arg}\' was {shape} with '
            f'{nelem} elements, exceeding the maximum allowed {max_nelem}')

class RankStatusArgShape(NodeFunc):
    """
    Generate shapes for all registered input signatures, plus point-mutated
    shapes, plus indel-mutated shapes.  Returns a list of items of:

    (index_ranks, (SchemaStatus, arg_shapes))
    """
    def __init__(self, dims_graph, index_dims_gen, op, nelem):
        super().__init__()
        self.dims_graph = dims_graph
        self.dims_gen = index_dims_gen
        self.op = op
        self.target_nelem = nelem

    def max_dimsize(self, arg_ranks, indel):
        ranks = dict(arg_ranks)
        if indel is not None:
            ranks[indel.arg] += indel.delta
        max_rank = max(ranks.values())
        return math.ceil(math.pow(self.target_nelem, 1.0 / max_rank))

    def gen_configurations(self):
        self.op.obs_dtypes.set_cached(None)
        self.op.obs_shapes.set_cached(None)
        self.op.obs_layout.set_cached(None)
        self.op.generation_mode = GenMode.Test
        self.op.test_error_cls = None
        inv_list = list(fgraph.gen_graph_values(self.op.inv_live_nodes,
            self.op.inv_output_nodes))
        return inv_list

    def filter_synonymous(self, inv_list):
        item = inv_list[0]
        akeys = list(item[1].keys())
        dkeys = list(item[2].keys())

        def item_hash(item):
            mut_arg_ranks = item[0]
            dtypes = item[2]
            layout = item[4]
            rt = tuple(mut_arg_ranks[k] for k in akeys)
            dt = tuple(dtypes[k] for k in dkeys)
            return hash(*rt, *dt, layout)

        success = set()
        for item in inv_list:
            status = item[6]
        if status is None:
            h = item_hash(item)
            success.add(h)

        skip = set()
        for i, item in enumerate(inv_list):
            status = item[6]
            if status is None:
                continue
            h = item_hash(item)
            if h in success:
                skip.add(i)

        filtered = []
        for i, item in enumerate(inv_list):
            if i not in skip:
                filtered.append(item)
        return filtered

    def index_dims(self, index_ranks, arg_sigs, gen_index_dims, max_dimsize,
            **dims_comp_args):
        """
        Resolve a set of all index dims consistent with {index_ranks}.  First,
        any indexes registered with add_index_generator or rank_dims_constraint
        will be computed.  Then, remaining indexes not registered with
        computed_index will be randomly generated in [1, max_dimsize].
        Finally, the computed index dims are created.  The system iterates
        until all computed index dims are non-negative.
        """
        # indexes appearing in at least one data tensor signature.  (both input
        # and return signatures) (some indexes are merely used as intermediate
        # quantities to simplify computation)
        sig_indexes = { idx for sig in arg_sigs.values() for idx in sig }

        # create deterministic order 
        sig_indexes = list(sorted(sig_indexes))
        gen_indexes = ''.join(gen_index_dims.keys())

        # generated dims will not change during the below iteration
        input_dims = dict(gen_index_dims) 
        for idx in sig_indexes:
            if idx in input_dims:
                continue
            if idx in self.dims_graph.computed_indexes():
                continue
            dims = [ randint(1, max_dimsize) for _ in range(index_ranks[idx]) ]
            input_dims[idx] = dims

        while True:
            comp_dims = self.dims_graph(input_dims, **dims_comp_args) 

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
            comp_inputs = self.dims_graph.get_index_inputs(comp_idx)
            # apply the assumption that computed indexes are either
            # component-wise or broadcasting.  secondly, assume that the
            # computed index is monotonically increasing in the values of the
            # input indices
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
                        f'Computed index \'{comp_idx}\' has rank {comp_rank} '
                        f'but has input index \'{input_idx}\' with rank '
                        f'{input_rank}.\n'
                        f'Computed indices must either be component-wise or '
                        f'broadcasting.')
                input_dims[input_idx][inc] += 1

        # TODO: move outside.  check this after applying indels

        for idx, dims in index_dims.items():
            if all(d >= 0 for d in dims):
                continue
            assert False, f'Index {idx} had negative dims {dims}'

        return index_dims

    def __call__(self, **kwargs):
        # idx_usage is a map of: idx => [arg1, arg2, ...] 
        # (arguments that the index appears in)
        inv_list = self.gen_configurations()
        inv_list = self.filter_synonymous(inv_list)

        idx_usage = defaultdict(list)
        indexes = { idx for sig in sigs.values() for idx in sig }
        for arg, sig in sigs.items():
            for idx in sig:
                if idx not in indexes:
                    continue
                idx_usage[idx].append(arg) 

            # run dims generation to create arg_shapes, then yield it
            max_dimsize = self.max_dimsize(arg_ranks, None)
            gen_dims_list = self.dims_gen(index_ranks)
            for gen_index_dims in gen_dims_list:
                index_dims = self.index_dims(index_ranks, sigs,
                        gen_index_dims, max_dimsize, **kwargs)
                check_sizes(arg_shapes, self.target_nelem * 5)
                item = (index_ranks, (Success, arg_shapes))
                yield item

                point_muts = shape_mutations(index_dims, sigs, idx_usage,
                        max_dimsize)
                for arg_shapes in point_muts:
                    check_sizes(arg_shapes, self.target_nelem)
                    item = (index_ranks, (IndexUsageError, arg_shapes))
                    yield item

        # Generated indel-mutated arg_shapes from an indel_list
        # the indel_list is a set of mutations at the arg level
        indel_list = make_indel_mutations(index_ranks_list, arg_sigs,
                self.op.definite_rank_indices)
        for indel in indel_list:
            max_dimsize = self.max_dimsize(indel.index_ranks, sigs, indel)
            gen_dims_list = self.dims_gen(indel.index_ranks)
            for gen_index_dims in gen_dims_list:
                index_dims = self.index_dims(indel.index_ranks, sigs,
                        gen_index_dims, max_dimsize, **kwargs)
                arg_shapes = shape_indels(index_dims, sigs, indel, max_dimsize)
                check_sizes(arg_shapes, self.target_nelem * 5)
                item = (indel.index_ranks, (NoMatchingRanks, arg_shapes))
                shapes_list.append(item)
        return shapes_list

class TupleElement(NodeFunc):
    """
    Expect a tuple and return a particular element
    """
    def __init__(self, index):
        super().__init__()
        self.index = index

    def __call__(self, tup):
        return [tup[self.index]]

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
        return [arg]

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
            return [arg]

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
        return [arg]

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
        return [arg]

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

    def __call__(self, obs_args):
        formats = self.op.data_formats
        all_layouts = list(range(formats.num_layouts()))
        if self.op.generation_mode == GenMode.Inventory:
            yield from all_layouts
        elif self.op.generation_mode == GenMode.Test:
            yield from all_layouts
        elif self.op.generation_mode == GenMode.Inference:
            obs_format = obs_args.get(formats.arg_name, base.DEFAULT_FORMAT)
            obs_layout = formats.layout(obs_format)
            yield obs_layout
            with self.new_error(1, self.op.data_format_gobj) as avail:
                if avail:
                    for alt_layout in all_layouts:
                        if alt_layout != obs_layout:
                            yield alt_layout

class DataFormat(ReportNodeFunc):
    """
    Generate the special data_format argument, defined by the 'layout' API call
    """
    def __init__(self, op, formats, arg_name):
        super().__init__(op, arg_name)
        self.formats = formats
        self.arg_name = arg_name

    def __call__(self, ranks, layout, obs_args):
        inferred_df = self.formats.data_format(layout, ranks)
        if self.op.generation_mode == GenMode.Inventory:
            yield inferred_df
        elif self.op.generation_mode == GenMode.Test:
            yield inferred_df
            with self.new_error(1) as avail:
                if not avail:
                    return
                for fmt in self.formats.all_formats():
                    if fmt != inferred_df:
                        yield fmt
        elif self.op.generation_mode == GenMode.Inference:
            obs_format = obs_args.get(self.arg_name, base.DEFAULT_FORMAT)
            if inferred_df == obs_format:
                yield obs_format
            else:
                with self.new_error(1) as avail:
                    if avail:
                        yield inferred_df
        else:
            raise RuntimeError('generation_mode not set')

class Sig(NodeFunc):
    """
    Represent a set of signatures for argument {name} corresponding to the
    available layouts. 
    """
    def __init__(self, name, options):
        super().__init__(name)
        self.options = options

    def __call__(self, layout):
        return [self.options[layout]]

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
        if self.op.generation_mode == GenMode.Inventory:
            yield from self.options
        elif self.op.generation_mode == GenMode.Test:
            yield from self.options
            with self.new_error(1) as avail:
                if not avail:
                    return
                yield 'DUMMY'
        elif self.op.generation_mode == GenMode.Inference:
            option = argmap[self.arg_name]
            if option in self.options: 
                yield option
            else:
                with self.new_error(1) as avail:
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

