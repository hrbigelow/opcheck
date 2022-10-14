import sys
import math
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
This file provides the complete collection of NodeFuncs for use in the GenNodes
of the generative graphs op.gen_graph and op.inv_graph (api.py).  

The goal of op.gen_graph is to produce a thorough set of test cases for the
predicate graph op.input_pred_graph, which produce either 'TP' or 'TN' (true
positive or true negative) results.

Each test is defined as the tuple (expected_status, config).  The test proceeds
as follows:

1. call arguments are produced from config
2. the wrapped framework op is called with the call arguments
3. the actual status is collected
4. the expected_status and actual_status are compared, generating TP, FP, TN,
or FN result.

In order to construct a generative graph that produces only TP and TN results,
the following notions are followed.

Each NodeFunc is one of two types.  A node's item type (the type of item in
the returned list) can either be a 'status tuple' consisting of (status,
value) or just a value.  The first kind produces a list containing tuples with
both Success as the status, and various test Status types.  The second kind of
node implicitly generates values that are expected to produce Success as a
result.

These implicit-success nodes should return an empty list if they receive
invalid inputs.


The inventory graph (inv_graph) will be run in three modes:

INVENTORY:  produces a list of input configurations that satisfy all schema
constraints

TEST_GEN: produces the same set as INVENTORY (each tagged with expected status
Success).  In addition, produces an additional set which violate exactly one
type of schema constraint (though may violate multiple constraints of the same
type).  These are also marked with the appropriate expected status.

INFERENCE: produce a set of configurations which satisfy all schema
constraints, and have up to op.max_observ_error observation constraint errors.

Inference will be run starting with 0, 1, etc setting for op.max_observ_error.
The majority of calls will find a configuration at 0 observed error.

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


class ObservedValue(NodeFunc):
    """
    Node for delivering inputs to any individual rank nodes.
    This is the portal to connect the rank graph to its environment
    """
    def __init__(self, name):
        super().__init__(name)

    def __call__(self):
        return [{}]

class RankRange(NodeFunc):
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

    def __call__(self, obs_dtypes, obs_shapes, obs_layout, sigs, **gen_inputs):
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
        obs_map = { 
                'dtypes': obs_dtypes, 
                'shapes': obs_shapes, 
                'layout': obs_layout,
                'sigs': sigs }

        # Get the initial bounds consistent with the schema
        lo, hi = 0, 1e10
        for cons in self.schema_cons:
            args = tuple(obs_map[arg] for arg in cons.get_argnames())
            clo, chi = cons(*args, **gen_inputs)
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
                args = tuple(obs_map[arg] for arg in cons.get_argnames())
                clo, chi = cons(*args, **gen_inputs)
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


"""
class ArgRank(NodeFunc):
    def __init__(self, op, arg_name):
        super().__init__(arg_name)
        self.op = op
        self.arg_name = arg_name

    def __call__(self, obs_shapes, sigs, **index_ranks):
        sig = sigs[self.arg_name]
        rank = sum(index_ranks[idx] for idx in sig)
        if self.op.generation_mode == GenMode.Inventory:
            yield rank
        elif self.op.generation_mode == GenMode.Test:
            yield rank
            if self.op.test_error_cls is None:
                # TODO: maybe fine-grain these errors
                self.op.test_error_cls = NoMatchingRanks
                yield rank + 1
                if rank > 0:
                    yield rank - 1
                self.op.test_error_cls = None
        elif self.op.generation_mode == GenMode.Inference:
            obs_shape = obs_shapes[self.arg_name]
            obs_rank = len(obs_shape)
            if rank == obs_rank:
                yield rank
            elif self.op.cur_edit_dist < self.op.max_edit_dist:
                self.op.cur_edit_dist += 1
                yield rank
                self.op.cur_edit_dist -= 1
        else:
            raise RuntimeError(f'generation mode not set')
"""

class DTypeIndiv(NodeFunc):
    """
    A Dtype with an individual valid set
    """
    def __init__(self, op, arg_name, valid_dtypes):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.op = op
        self.valid_dtypes = valid_dtypes
        self.invalid_dtypes = tuple(t for t in ALL_DTYPES if t not in
                valid_dtypes)

    def __call__(self, obs_dtypes):
        if self.op.generation_mode == GenMode.Inventory:
            yield from self.valid_dtypes
        elif self.op.generation_mode == GenMode.Test:
            yield from self.valid_dtypes
            if self.op.test_error_cls is None:
                self.op.test_error_cls = DTypeNotValid
                yield from self.invalid_dtypes
                self.op.test_error_cls = None
        elif self.op.generation_mode == GenMode.Inference:
            # get observation
            # print(obs_dtypes)
            obs_dtype = obs_dtypes[self.arg_name]
            if obs_dtype in self.valid_dtypes:
                yield obs_dtype
            if self.op.cur_edit_dist < self.op.max_edit_dist:
                self.op.cur_edit_dist += 1
                for dtype in self.valid_dtypes:
                    if dtype != obs_dtype:
                        yield dtype
                self.op.cur_edit_dist -= 1
        else:
            raise RuntimeError(f'generation mode not set')

class DTypeEquiv(NodeFunc):
    """
    A DType which is declared equal to another using equate_dtypes 
    """
    def __init__(self, op, arg_name):
        super().__init__(arg_name)
        self.op = op
        self.arg_name = arg_name
        self.all_dtypes = ALL_DTYPES

    def __call__(self, obs_dtypes, src_dtype):
        if self.op.generation_mode == GenMode.Inventory:
            yield src_dtype
        elif self.op.generation_mode == GenMode.Test:
            yield src_dtype # leave error alone
            if self.op.test_error_cls is not None:
                return
            self.op.test_error_cls = DTypeNotEqual
            for dtype in self.all_dtypes:
                if dtype != src_dtype:
                    yield dtype
            self.op.test_error_cls = None
        elif self.op.generation_mode == GenMode.Inference:
            # get the observation
            obs_dtype = obs_dtypes[self.arg_name]
            if src_dtype == obs_dtype:
                yield obs_dtype
            else:
                if self.op.cur_edit_dist < self.op.max_edit_dist:
                    self.op.cur_edit_dist += 1
                    yield src_dtype
                    self.op.cur_edit_dist -= 1
        else:
            raise RuntimeError('generation_mode not set')

class DTypesNotImplemented(NodeFunc):
    """
    Represents configurations that are not implemented, declared with API
    function exclude_dtypes.  {fields} is a tuple containing members of:
    - names of data tensors
    - one-letter index names
    - the base.LAYOUT constant

    See api::exclude_dtypes for more detail
    """
    def __init__(self, op):
        super().__init__()
        self.op = op
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

        if self.op.generation_mode == GenMode.Inventory:
            if is_ex:
                return
            else:
                yield self.get_dtype_map(dtypes)
        elif self.op.generation_mode == GenMode.Test:
            if is_ex:
                if self.op.test_error_cls is None:
                    self.op.test_error_cls = DTypeComboExcluded
                    yield self.get_dtype_map(dtypes)
                    self.op.test_error_cls = None
            else:
                yield self.get_dtype_map(dtypes)

        elif self.op.generation_mode == GenMode.Inference:
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
    """
    def __init__(self, op):
        super().__init__()
        self.op = op

    def __call__(self):
        yield self.op.test_error_cls

class DTypesStatus(NodeFunc):
    def __init__(self, dtype_cons):
        super().__init__()
        self.dtype_cons = dtype_cons
        # double is alias for float64
        # half is alias for float16
        # tf.string, tf.variant, tf.resource are omitted
        self.all_dtypes = (
                tf.int8, tf.int16, tf.int32, tf.int64,
                tf.uint8, tf.uint16, tf.uint32, tf.uint64,
                tf.float16, tf.float32, tf.float64,
                tf.qint8, tf.qint16, tf.qint32,
                tf.bfloat16, 
                tf.bool,
                tf.complex64, tf.complex128
                )

    def __call__(self, ranks, layout):
        """
        Generate a list of all possible dtype combinations for data tensors.
        Each combination is expressed as a map of arg => dtype.

        Each item in the list is paired with the expected SchemaStatus object
        that it would produce when validated.
        """
        tests = self.dtype_cons.tests
        tensor_names = self.dtype_cons.tensors
        k = len(tensor_names)
        max_errors = 1

        # list of (type, dtype_tuple), where <type> is a SchemaStatus expected
        # to be produced from this tuple.
        combo_gen = util.dtype_combos(k, self.all_dtypes, tests, max_errors,
                ranks, layout)
        combos = list(combo_gen)
        combo_maps = [ (s, dict(zip(tensor_names, d))) for s, d in combos ]
        return combo_maps

class ValidDTypes(DTypesStatus):
    def __init__(self, dtype_cons):
        super().__init__(dtype_cons)

    def __call__(self, ranks, layout):
        combo_maps = super().__call__(ranks, layout)
        valid_maps = [ d for stat_cls, d in combo_maps if stat_cls == Success ]
        return valid_maps

class Ranks(NodeFunc):
    """
    Gather ranks together into one map
    """
    def __init__(self):
        super().__init__()

    def __call__(self, **ranks):
        return [ranks]

"""
class Ranks(NodeFunc):
    # Generates combinations of ranks according to the currently active
    # constraints in the nodes.
    def __init__(self, op):
        super().__init__()
        self.op = op

    def __call__(self, **kwargs):
        self.op.rank_input.set_cached(kwargs)
        ranks_list = fgraph.gen_graph_iterate(*self.op.rank_graph.values())
        return ranks_list
"""

"""
class Ranks(NodeFunc):
    def __init__(self, op, rank_candidates):
        super().__init__()
        self.op = op
        self.rcands = rank_candidates

    def __call__(self):
        # Generate all allowed rank combinations.  Generates a list of maps.
        # Each map has index => rank for each index in self.index
        idx_ranks_list = list(self.rcands.all_index_ranks())
        return idx_ranks_list 
"""

class Rank(NodeFunc):
    """
    Generate the rank of a given signature
    """
    def __init__(self, sig):
        super().__init__(sig)
        self.sig = sig

    def __call__(self, ranks_map):
        rank = sum(ranks_map[s] for s in self.sig)
        return [rank]

class Dims(NodeFunc):
    """
    Generate dimensions for {output_indices} using {gen_func}. 

    Calls gen_func(ranks_list, *gen_args).  ranks_list are the ranks of each
    index in {input_indices} in order.

    returns a list of shape tuples, one shape for each index in output_indices.
    A shape is an integer list.  

    For example, if output_indices has two indices, a return value could be:
    [ 
      ([1,2,3], [4,5]),
      ([6,4,2], [5,4]) 
    ]
    """
    def __init__(self, gen_func, output_indices, input_indices, gen_args):
        super().__init__(output_indices)
        self.output_indices = output_indices 
        self.input_indices = input_indices 
        self.func = gen_func
        self.gen_args = gen_args

    @staticmethod
    def valid_return(vals):
        return (
                isinstance(vals, list) and
                all(isinstance(v, tuple) for v in vals) and
                all(isinstance(s, list) for v in vals for s in v)
                )

    def __call__(self, ranks_map):
        ranks_list = [ ranks_map[i] for i in self.input_indices ]
        vals = self.func(ranks_list, *self.gen_args)
        if not self.valid_return(vals):
            raise SchemaError(
                f'{type(self).__qualname__}: Custom Dims generation function '
                f'\'{self.func.__name__}\' returned the wrong type.  Expected '
                f'a list of shape tuples, for example like: \n'
                f'[ \n'
                f'  ([1,2,3], [4,5]),\n'
                f'  ([6,4,2], [5,4]) \n'
                f'].\n'
                f'Got: {vals}\n')
        return [ dict(zip(self.output_indices,v)) for v in vals ]

# Specifies a set of data tensor shapes, in which one tensor has an insertion
# or deletion of a dimension relative to a valid set.
IndelMutation = namedtuple('IndelMutation', ['index_ranks', 'arg', 'delta'])

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

def get_arg_ranks(index_ranks, arg_sigs):
    arg_ranks = {}
    for arg, sig in arg_sigs.items():
        arg_ranks[arg] = sum(index_ranks[idx] for idx in sig)
    return arg_ranks

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

def check_sizes(arg_shapes, max_nelem):
    for arg, shape in arg_shapes.items():
        nelem = np.prod(shape)
        if nelem < max_nelem:
            continue
        raise SchemaError(
            f'Generated shape for argument \'{arg}\' was {shape} with '
            f'{nelem} elements, exceeding the maximum allowed {max_nelem}')

class Inventory(NodeFunc):
    """
    Generate inventory and test cases for the op.
    """
    def __init__(self, op):
        super().__init__()
        self.obs_nodes = (op.obs_dtypes, op.obs_shapes, op.obs_layout)
        self.outputs = op.inv_output_nodes

    def __call__(self, **kwargs):
        for n in self.obs_nodes:
            n.set_cached(None)
        yield from fgraph.all_values(*self.outputs)

class RankStatusArgShape(NodeFunc):
    """
    Generate shapes for all registered input signatures, plus point-mutated
    shapes, plus indel-mutated shapes.  Returns a list of items of:

    (index_ranks, (SchemaStatus, arg_shapes))
    """
    def __init__(self, dims_graph, index_ranks_gen, index_dims_gen, op, nelem):
        super().__init__()
        self.dims_graph = dims_graph
        self.ranks_gen = index_ranks_gen
        self.dims_gen = index_dims_gen
        self.op = op
        self.target_nelem = nelem

    def max_dimsize(self, index_ranks, arg_sigs, indel):
        ranks = get_arg_ranks(index_ranks, arg_sigs)
        if indel is not None:
            ranks[indel.arg] += indel.delta
        max_rank = max(ranks.values())
        return math.ceil(math.pow(self.target_nelem, 1.0 / max_rank))

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

    def __call__(self, arg_sigs, ret_sigs, **kwargs):
        # idx_usage is a map of: idx => [arg1, arg2, ...] 
        # (arguments that the index appears in)
        
        all_sigs = { **arg_sigs, **ret_sigs }
        idx_usage = defaultdict(list)
        indexes = { idx for sig in arg_sigs.values() for idx in sig }
        for arg, sig in arg_sigs.items():
            for idx in sig:
                if idx not in indexes:
                    continue
                idx_usage[idx].append(arg) 

        # TODO: replace this with the graph generation
        index_ranks_list = list(self.ranks_gen.all_index_ranks())
        shapes_list = []
        # Generate correct and point-mutated arg_shapes from each index_ranks
        for index_ranks in index_ranks_list:
            max_dimsize = self.max_dimsize(index_ranks, all_sigs, None)
            gen_dims_list = self.dims_gen(index_ranks)
            for gen_index_dims in gen_dims_list:
                index_dims = self.index_dims(index_ranks, all_sigs,
                        gen_index_dims, max_dimsize, **kwargs)
                arg_shapes = get_arg_shapes(index_dims, all_sigs)
                check_sizes(arg_shapes, self.target_nelem * 5)
                item = (index_ranks, (Success, arg_shapes))
                shapes_list.append(item)

                point_muts = shape_mutations(index_dims, all_sigs, idx_usage,
                        max_dimsize)
                for arg_shapes in point_muts:
                    check_sizes(arg_shapes, self.target_nelem)
                    item = (index_ranks, (IndexUsageError, arg_shapes))
                    shapes_list.append(item)

        # Generated indel-mutated arg_shapes from an indel_list
        # the indel_list is a set of mutations at the arg level
        indel_list = make_indel_mutations(index_ranks_list, arg_sigs,
                self.op.definite_rank_indices)
        for indel in indel_list:
            max_dimsize = self.max_dimsize(indel.index_ranks, all_sigs, indel)
            gen_dims_list = self.dims_gen(indel.index_ranks)
            for gen_index_dims in gen_dims_list:
                index_dims = self.index_dims(indel.index_ranks, all_sigs,
                        gen_index_dims, max_dimsize, **kwargs)
                arg_shapes = shape_indels(index_dims, all_sigs, indel,
                        max_dimsize)
                check_sizes(arg_shapes, self.target_nelem * 5)
                item = (indel.index_ranks, (NoMatchingRanks, arg_shapes))
                shapes_list.append(item)
        return shapes_list

class StatusAggregator(NodeFunc):
    """
    Collects the first element of every pair tuple input.
    Expects each to be a SchemaStatus type, and returns the first non-Success
    one.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        statuses = tuple(a[0] for a in args)
        return [statuses]

class TupleElement(NodeFunc):
    """
    Expect a tuple and return a particular element
    """
    def __init__(self, index):
        super().__init__()
        self.index = index

    def __call__(self, tup):
        return [tup[self.index]]

class GetRanks(TupleElement):
    def __init__(self):
        super().__init__(0)

class GetStatusArgShape(TupleElement):
    def __init__(self):
        super().__init__(1)

class GetSigs(TupleElement):
    def __init__(self):
        super().__init__(2)

class GetDTypes(TupleElement):
    def __init__(self):
        super().__init__(1)

class GetArgShapes(TupleElement):
    def __init__(self):
        super().__init__(1)

class DataTensor(NodeFunc):
    """
    Produce the (shape, dtype) combo needed to produce a tensor
    """
    def __init__(self, arg_name):
        super().__init__(arg_name)
        self.arg_name = arg_name

    def __call__(self, arg_shapes, dtypes, **kwargs):
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

class Layout(NodeFunc):
    def __init__(self, op, name):
        super().__init__(name)
        self.op = op

    def __call__(self):
        return list(range(self.op.data_formats.num_layouts()))

class DataFormat(NodeFunc):
    """
    Generate the special data_format argument, defined by the 'layout' API call
    """
    def __init__(self, formats):
        super().__init__()
        self.formats = formats

    # TODO: what is the logic for generating extra test values?
    def __call__(self, ranks, layout):
        df = self.formats.data_format(layout, ranks)
        if df is None:
            return [None]
        else:
            return [df]

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

class Options(NodeFunc):
    """
    Represent a specific set of options known at construction time
    """
    def __init__(self, name, options):
        super().__init__(name)
        try:
            iter(options)
        except TypeError:
            raise SchemaError(
                f'{type(self).__qualname__}: \'options\' argument must be '
                f'iterable.  Got {type(options)}')
        self.options = options

    def __call__(self):
        return self.options

