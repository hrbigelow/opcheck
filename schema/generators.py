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
from .base import Kind, kind, kpfx
from .fgraph import FuncNode as F, func_graph_evaluate
from .error import *
from . import util

class DTypes(object):
    def __init__(self, dtype_cons):
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

    def __call__(self, ranks, **kwargs):
        """
        Generate all valid dtype combinations.  Generates a list of maps.
        Each map has a full tensor_name => dtype for each input tensor
        """
        tests = self.dtype_cons.tests
        tensor_names = self.dtype_cons.tensors
        k = len(tensor_names)
        max_errors = 1

        layout = kwargs.get(Kind.LAYOUT, None)

        # list of (type, dtype_tuple), where <type> is a SchemaStatus expected
        # to be produced from this tuple.
        combo_gen = util.dtype_combos(k, self.all_dtypes, tests, max_errors,
                ranks, layout)
        combos = list(combo_gen)
        combo_maps = [ (s, dict(zip(tensor_names, d))) for s, d in combos ]
        return combo_maps

class Ranks(object):
    def __init__(self, op, rank_candidates):
        self.op = op
        self.rcands = rank_candidates

    def __call__(self):
        """
        Generate all allowed rank combinations.  Generates a list of maps.
        Each map has index => rank for each index in self.index
        """
        idx_ranks_list = list(self.rcands.all_index_ranks())
        return idx_ranks_list 

class Rank(object):
    """
    Generate the rank of a given signature
    """
    def __init__(self, sig):
        self.sig = sig

    def __call__(self, ranks_map):
        rank = sum(ranks_map[s] for s in self.sig)
        return [rank]

class Dims(object):
    """
    Generate dimensions for {output_indices} using {gen_func}.  Used in
    Kind.GEN_DIMS nodes.  Has parent Kind.RANKS

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

class IndexDimsGD(object):
    """
    Perform gradient descent on two objectives to find suitable testing index
    dimensions.
    """
    def __init__(self, op, min_nelem, max_nelem):
        self.op = op
        self.lr = 2.0
        self.min_nelem = min_nelem
        self.max_nelem = max_nelem
        self.nodes = []
        self.variables = {} # idx => tf.Variable
        self.gen_index_dims = {}  # idx => 
        self.computed_indexes = set()
        self.extra_args = {}

        # edges to add after node creation
        self.edges = {} # kname => [ parent_kname, parent_kname, ... ] 

    def nonfree_inds(self):
        return (*self.computed_indexes, *self.gen_index_dims.keys())

    def add_free_index(self, idx):
        fi_obj = FreeIndex(self, idx)
        node = F.add_node(idx, fi_obj)
        self.nodes.append(node)
        self.variables[idx] = None

    def add_gen_index(self, idx):
        gi_obj = GenIndex(self, idx)
        node = F.add_node(idx, gi_obj)
        self.nodes.append(node)
        self.gen_index_dims[idx] = None

    def add_comp_index(self, idx, comp_func, parent_indices, *const_args):
        # adds a computed index graph node.  const_args are non-index arguments
        # which are constant during the gradient descent.  
        # all names in const_args must be keys in __call__'s **kwargs
        ci_obj = CompIndex(self, comp_func, const_args)
        node = F.add_node(idx, ci_obj)
        self.edges[idx] = parent_indices
        self.nodes.append(node)
        self.computed_indexes.add(idx)

    def finalize(self):
        for n, pnodes in self.edges.items():
            node = F.get_node(n)
            for pn in pnodes:
                pnode = F.get_node(pn)
                node.append_parent(pnode)

    def prepare(self, ranks, arg_sigs, **kwargs):
        # split kwargs into index and non-index
        self.extra_args.clear()

        for k, v in kwargs.items():
            if kind(k) == Kind.GEN_DIMS:
                dims_map = v
                for idx, dims in dims_map.items():
                    self.gen_index_dims[idx] = dims
            else:
                self.extra_args[k] = v
        self.arg_sigs = arg_sigs
        for idx in self.variables.keys():
            rank = ranks[idx]
            # hard-coded reasonable choices for initialization
            val = tf.random.uniform([rank], 1.0, 4.0)
            cons_fun = lambda v: tf.clip_by_value(tf.round(v), 1.0, 1e10)
            # var = tf.Variable(val, constraint = cons_fun, dtype=tf.float32)
            var = tf.Variable(val, dtype=tf.float32)
            self.variables[idx] = var

    def all_dims(self):
        dims = func_graph_evaluate(self.nodes)
        return dims

    def num_elem(self, sig):
        # computes the number of elements for sig, given the current index
        # dimensions.  if one or more indices are negative, returns the
        # negative of absolute number
        dims = self.all_dims()
        vals = [dims[s] for s in sig]
        # this allows returning '1' for rank 0 sig (this is incorrect but
        # good enough)
        vals.append(tf.constant([1.0]))
        st = tf.concat(vals, axis=0)
        nelem = tf.reduce_min(tf.sign(st)) * tf.abs(tf.reduce_prod(st))
        return nelem

    def index_loss(self):
        # Ensures all computed dims are >= 1
        # print('in index loss')
        dims = self.all_dims()
        comp_dims = [ dim for idx, dim in dims.items() if idx in
                self.computed_indexes ]
        st = tf.concat(comp_dims, axis=0)
        loss = tf.reduce_sum(tf.nn.relu(1.0 - st))
        return loss

    def max_product_degree(self):
        # compute the largest degree (number of non-constant terms) of any
        # product in the elem_loss.
        all_vars = self.all_dims().items()
        live_vars = [(k,v) for k,v in all_vars if k not in self.gen_index_dims]
        ranks = { k: v.numpy().size for k, v in live_vars } 
        sigs = self.arg_sigs.values()
        max_degree = max(sum(ranks.get(s, 0) for s in sig) for sig in sigs)
        return max_degree

    def elem_loss(self):
        """
        The objective is just a sum of product terms (the numbers of elements),
        with the added complication that some of the products may be functions
        of others.

        The max_degree is the largest degree of any product term, counting only
        the variables in the products (not the generated indices, which are
        constant during gradient descent).

        The objective is then the d'th root of the sum, which roughly makes it
        a constant gradient.  If all inputs are zero, the objective is the d'th
        root of min_nelem and has a slope of -1 (hitting zero when each
        variable is equal to the d'th root of min_nelem as well).

        Therefore, to ensure about the same number of gradient steps regardless
        of degree, we scale the objective (and hence the gradient) by this
        the d'th root of min_nelem.

        The objective is basically monotonically decreasing in every free
        variable.
        """
        max_degree = self.max_product_degree()
        sigs = self.arg_sigs.values()
        card = tuple(self.num_elem(sig) for sig in sigs)
        nelem = tf.reduce_sum(card)
        expon = 1.0 / max_degree
        objective = tf.math.pow(nelem, expon)
        min_objective = tf.math.pow(self.min_nelem, expon)
        max_objective = tf.math.pow(self.max_nelem, expon)

        ival_dist = tf.maximum(
                tf.nn.relu(min_objective - objective),
                tf.nn.relu(objective - max_objective)
                )
        mul = tf.math.pow(self.min_nelem, 1.0 / max_degree)
        mul *= 0.05
        return ival_dist * mul

    def __call__(self, ranks, arg_sigs, **kwargs):
        # kwargs should have:
        # all generated index dims declared with add_index_generator
        # all extra arguments declared in calls to computed_index
        self.prepare(ranks, arg_sigs, **kwargs)
        opt = tf.keras.optimizers.SGD(learning_rate=self.lr)
        index_lambda = lambda: self.index_loss()
        elem_lambda = lambda: self.elem_loss()

        losses = []
        if len(self.computed_indexes) > 0:
            losses.append(('index_loss', index_lambda))
        losses.append(('elem_loss', elem_lambda))

        # Calling apply_gradients on any variables with zero elements results
        # in: ./tensorflow/core/util/gpu_launch_config.h:129] Check failed:
        # work_element_count > 0 (0 vs. 0)
        # when calling apply_gradients
        free_vars = [ v for v in self.variables.values() if v.shape[0] > 0 ]

        # The first phase (index_loss) ensures that all index dims are positive
        # (free, generated, and computed). Under that condition, the next
        # surface (elem_loss) is monotonically increasing in all free
        # dimensions.  And, because of the finite interval solution, zero loss
        # is achievable
        with tf.device('/device:CPU:0'):
            for loss_name, loss_func in losses:
                for step in range(10000):
                    with tf.GradientTape() as tape:
                        objective = loss_func()
                    if step % 10 == 0:
                        max_degree = self.max_product_degree()
                        s = ', '.join(f'{n}: {f():5.3f}' for n, f in losses)
                        dims = { n: v.numpy().astype(np.int32).tolist() for n,
                                v in self.variables.items() }
                        print(f'{step}: minimizing {loss_name}, max-rank: '
                                f'{max_degree} dims: {dims}, {s}')
                    if objective == 0.0:
                        break
                    grads = tape.gradient(objective, free_vars)
                    for grad, var in zip(grads, free_vars):
                        if grad is None:
                            continue
                        # var.assign_sub(self.lr * grad)
                        var.assign(tf.maximum(var - self.lr * grad, 1.0))
                        assert all(c >= 1.0 for c in var.numpy()), 'after'
            
        # extract the dims map
        vars_map = self.all_dims()
        index_dims = {}
        for idx, var in vars_map.items():
            index_dims[idx] = var.numpy().astype(np.int32).tolist() 

        if any(d < 1 for dims in index_dims.values() for d in dims):
            raise SchemaError(
                f'Failed to find positive computed dims: {index_dims}')

        if self.elem_loss() > 0.0:
            raise SchemaError(
                f'Failed to achieve zero elem loss.  Final elem_loss: '
                f'{self.elem_loss():5.3f}')

        # create arg_shapes
        arg_shapes = {}
        for arg, sig in arg_sigs.items():
            arg_shapes[arg] = [ d for s in sig for d in index_dims[s] ]

        # create a map of free index usage for input args
        idx_usage = defaultdict(list)
        free_idxs = list(self.variables.keys())
        for arg, sig in arg_sigs.items():
            if arg not in self.op.params:
                continue
            for idx in sig:
                if idx not in free_idxs:
                    continue
                idx_usage[idx].append(arg) 

        tests = []
        tests.append((Success, arg_shapes))
        for idx, args in idx_usage.items():
            if ranks[idx] == 0:
                continue
            if len(args) == 1:
                continue
            for mut_arg in args:
                arg_shapes = defaultdict(list)
                for arg, sig in arg_sigs.items():
                    for usage in sig:
                        snip = list(index_dims[usage])
                        if arg == mut_arg and idx == usage:
                            # randomly mutate snip
                            c = randint(0, len(snip)-1)
                            s = randint(-snip[c]+1, 10)
                            # ensure we actually change it by a non-zero amount
                            snip[c] += s + int(s == 0)
                        arg_shapes[arg].extend(snip)
                tests.append((IndexUsageError, dict(arg_shapes)))
        return tests 

# Specifies a set of data tensor shapes, in which one tensor has an insertion
# or deletion relative to a valid set.
IndelMutation = namedtuple('IndelMutation', ['index_ranks', 'arg', 'delta'])

def make_indel_mutations(index_ranks_list, arg_sigs):
    """
    returns a list of IndelMutations 
    """
    arg_names = list(arg_sigs.keys())
    num_arg = len(arg_names)
    # arg rank values tuple => (index_ranks, changed_arg, delta)
    ranks_to_aug = {} 
    for index_ranks in index_ranks_list:
        arg_ranks = get_arg_ranks(index_ranks, arg_sigs)
        ranks_tup = tuple(arg_ranks.values())
        for t in range(num_arg):
            for delta in (-1, 1):
                mut_ranks = tuple(r + delta if i == t else r
                        for i, r in enumerate(ranks_tup))
                if mut_ranks in ranks_to_aug:
                    continue
                if any(r < 0 for r in mut_ranks):
                    continue
                mut = IndelMutation(index_ranks, arg_names[t], delta)
                ranks_to_aug[mut_ranks] = mut 
    return list(ranks_to_aug.values())

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
                for usage in sig:
                    snip = list(index_dims[usage])
                    if arg == mut_arg and idx == usage:
                        c = randint(0, len(snip)-1)
                        new_val, alt_val = sample(range(1, max_dimsize), 2)
                        snip[c] = new_val if new_val != snip[c] else alt_val
                    arg_shapes[arg].extend(snip)
            results.append(dict(arg_shapes))
    return results 

class SignatureShapes(object):
    """
    Generate shapes for all registered input signatures, plus point-mutated
    shapes, plus indel-mutated shapes.
    """
    def __init__(self, dims_graph, index_ranks_gen, index_dims_gen,
            target_nelem):
        self.dims_graph = dims_graph
        self.ranks_gen = index_ranks_gen
        self.dims_gen = index_dims_gen
        self.target_nelem = target_nelem

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
        # indexes appearing in at least one data tensor signature.  (some
        # indexes are merely used as intermediate quantities to simplify
        # computation)
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
                if idx in sig_indexes
                and dim < 0), None)
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

        # print(index_dims)
        arg_shapes = get_arg_shapes(index_dims, arg_sigs)
        for arg, shape in arg_shapes.items():
            nelem = np.prod(shape)
            if nelem < 1e8:
                continue
            raise SchemaError(
                f'Generated shape for argument \'{arg}\' was {shape} with '
                f'{nelem} elements, exceeding the maximum allowed 1e8')
        return index_dims

    def __call__(self, arg_sigs, **kwargs):

        # idx => [arg1, arg2, ...] (arguments that the index appears in)
        idx_usage = defaultdict(list)
        indexes = { idx for sig in arg_sigs.values() for idx in sig }
        for arg, sig in arg_sigs.items():
            for idx in sig:
                if idx not in indexes:
                    continue
                idx_usage[idx].append(arg) 

        index_ranks_list = list(self.ranks_gen.all_index_ranks())
        shapes_list = []
        # Success and IndexUsageError lists
        for index_ranks in index_ranks_list:
            max_dimsize = self.max_dimsize(index_ranks, arg_sigs, None)
            gen_dims_list = self.dims_gen(index_ranks)
            for gen_index_dims in gen_dims_list:
                index_dims = self.index_dims(index_ranks, arg_sigs,
                        gen_index_dims, max_dimsize, **kwargs)
                arg_shapes = get_arg_shapes(index_dims, arg_sigs)
                item = (index_ranks, (Success, arg_shapes))
                shapes_list.append(item)

                point_muts = shape_mutations(index_dims, arg_sigs, idx_usage,
                        max_dimsize)
                for arg_shapes in point_muts:
                    item = (index_ranks, (IndexUsageError, arg_shapes))
                    shapes_list.append(item)

        # NoMatchingRanks list
        indel_list = make_indel_mutations(index_ranks_list, arg_sigs)
        for indel in indel_list:
            max_dimsize = self.max_dimsize(indel.index_ranks, arg_sigs, indel)
            gen_dims_list = self.dims_gen(indel.index_ranks)
            for gen_index_dims in gen_dims_list:
                index_dims = self.index_dims(indel.index_ranks, arg_sigs,
                        gen_index_dims, max_dimsize, **kwargs)
                arg_shapes = shape_indels(index_dims, arg_sigs, indel,
                        max_dimsize)
                item = (indel.index_ranks, (NoMatchingRanks, arg_shapes))
                shapes_list.append(item)
        return shapes_list

class StatusAggregator(object):
    """
    Collects the first element of every pair tuple input.
    Expects each to be a SchemaStatus type, and returns the first non-Success
    one.
    """
    def __init__(self):
        pass

    def __call__(self, *args):
        statuses = tuple(a[0] for a in args)
        return [statuses]

class TupleElement(object):
    """
    Expect a tuple and return a particular element
    """
    def __init__(self, index):
        self.index = index

    def __call__(self, tup):
        return [tup[self.index]]

class SingleIndexDims(object):
    """
    A simple node which extracts a single index dimension
    """
    def __init__(self, index_name):
        self.name = index_name

    def __call__(self, index_dims):
        return [index_dims[self.name]]

class TensorStub(object):
    """
    Produce the (shape, dtype) combo needed to produce a tensor
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, arg_shapes, **kwargs):
        dtype_map = kwargs[Kind.DTYPES]
        dtype = dtype_map[self.arg_name]
        shape = arg_shapes[self.arg_name]
        return [(shape, dtype)]

def from_stub(stub):
    shape, dtype = stub
    nelem = np.prod(shape)
    if nelem > 1e7:
        raise SchemaError(f'Shape \'{shape}\' has {nelem} elements, '
        f'which exceeds 1e7 elements')
    if dtype.is_integer:
        lo = max(dtype.min, -1000)
        hi = min(dtype.max, 1000) 
        ten = tf.random.uniform(shape, lo, hi, dtype=tf.int64)
        ten = tf.cast(ten, dtype)
    elif dtype.is_floating:
        lo = max(dtype.min, -1.0)
        hi = min(dtype.max, 1.0)
        ten = tf.random.uniform(shape, lo, hi, dtype=tf.float64)
        ten = tf.cast(ten, dtype)
    elif dtype.is_bool:
        ten = tf.random.uniform(shape, 0, 2, dtype=tf.int32)
        ten = tf.cast(ten, dtype)
    elif dtype.is_quantized:
        lo, hi = -1000, 1000
        ten = tf.random.uniform(shape, lo, hi, dtype=tf.float32)
        quant = tf.quantization.quantize(ten, lo, hi, dtype)
        ten = quant.output
    elif dtype.is_complex:
        lo, hi = -1.0, 1.0
        real = tf.random.uniform(shape, lo, hi, dtype=tf.float64)
        imag = tf.random.uniform(shape, lo, hi, dtype=tf.float64)
        ten = tf.complex(real, imag, dtype)
    else:
        raise SchemaError(
            f'Unexpected dtype when generating tensor: dtype=\'{dtype.name}\'')
    return ten

class ShapeInt(object):
    """
    Expect the shape of index to be rank 1 and extract the first component as
    an integer.  
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, arg_shapes):
        shape = arg_shapes[self.arg_name]
        if len(shape) != 1:
            raise SchemaError(
                f'{type(self).__qualname__}: index \'{index}\' has a '
                f'non-rank-1 shape \'{shape}\'.  Cannot convert it to an '
                f'integer.')
        return [shape[0]]

class ShapeList(object):
    """
    Generate the current shape of the input signature
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, arg_shapes):
        shape = arg_shapes[self.arg_name]
        return [shape]

class ShapeTensor(object):
    """
    Generate the current shape of the input signature as a tensor
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, arg_shapes):
        shape = arg_shapes[self.arg_name]
        ten = tf.constant(shape, dtype=tf.int32)
        return [ten]

class ShapeTensor2D(object):
    """
    Generate a 2D tensor from dims and a list of signatures
    """
    def __init__(self, arg_name, num_rows):
        self.arg_name = arg_name
        self.num_rows = num_rows

    def __call__(self, arg_shapes):
        names = [ f'{self.arg_name}.{i}' for i in range(self.num_rows) ]
        rows = [ arg_shapes[n] for n in names ]
        ten = tf.constant(rows, dtype=tf.int32)
        ten = tf.transpose(ten, (1,0))
        return [ten]

class SigMap(object):
    """
    Aggregate all of the :sig nodes into a map of arg_name => sig
    """
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        sig_map = {}
        for kn, val in kwargs.items():
            sig_map[kpfx(kn)] = val
        return [sig_map]

class Closure(object):
    def __init__(self, obj):
        self.obj = obj

    def __call__(self):
        return self.obj

class Layout(object):
    def __init__(self, layouts):
        self.num_layouts = len(layouts)

    def __call__(self):
        return list(range(self.num_layouts))

class DataFormat(object):
    """
    Generate the special data_format argument, defined by the 'layout' API call
    """
    def __init__(self, layouts, rank_index):
        self.layouts = layouts
        self.rank_index = rank_index

    def __call__(self, rank_map, layout):
        rank = rank_map[self.rank_index]
        rmap = self.layouts[layout]
        # rank may be outside of the range of valid ranks, due to being a
        # mutation.  if so, choose a default
        default = next(iter(rmap.values()))
        data_format = rmap.get(rank, default)
        return [data_format]

class LayoutOption(object):
    def __init__(self, options):
        self.options = options

    def __call__(self, layout):
        return [self.options[layout]]

class Int(object):
    def __init__(self, lo, hi):
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

