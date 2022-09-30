import sys
import math
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import itertools
from random import randint
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

    def __call__(self, **kwargs):
        """
        Generate all valid dtype combinations.  Generates a list of maps.
        Each map has a full tensor_name => dtype for each input tensor
        """
        tests = self.dtype_cons.tests
        tensor_names = self.dtype_cons.tensors
        k = len(tensor_names)
        max_errors = 1

        # list of (type, dtype_tuple), where <type> is a SchemaStatus expected
        # to be produced from this tuple.
        combo_gen = util.dtype_combos(k, self.all_dtypes, tests, max_errors,
                kwargs)
        combos = list(combo_gen)
        combo_maps = [ (s, dict(zip(tensor_names, d))) for s, d in combos ]
        return combo_maps

class Ranks(object):
    def __init__(self, op, rank_candidates):
        self.op = op
        self.rcands = rank_candidates

    def __call__(self, arg_sigs):
        """
        Generate all allowed rank combinations.  Generates a list of maps.
        Each map has index => rank for each index in self.index
        """
        def sig_ranks(ranks, arg_sigs):
            tup = tuple(sum(ranks[s] for s in sig) for sig in arg_sigs.values())
            return tup

        idx_ranks_list = list(self.rcands.all_index_ranks())
        arg_ranks = set()
        for ranks in idx_ranks_list:
            tup = sig_ranks(ranks, arg_sigs)
            arg_ranks.add(tup)

        # Now, create mutations
        idx_tests_list = [] # items are (<expected_status_class>, idx_ranks)
        for ranks in idx_ranks_list:
            idx_tests_list.append((Success, ranks)) 
            mut = dict(ranks)
            for idx, rank in ranks.items():
                for adj in (-1, 1):
                    if rank + adj < 0:
                        continue
                    mut[idx] = rank + adj
                    tup = sig_ranks(ranks, arg_sigs)
                    if tup in arg_ranks:
                        continue
                    idx_tests_list.append((NoMatchingRanks, dict(mut)))
                mut[idx] = rank

        return idx_tests_list

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


class FreeIndex(object):
    # FuncNode object for free indices
    def __init__(self, gd, index_name):
        self.gd = gd
        self.idx = index_name

    def __call__(self):
        return self.gd.variables[self.idx]

class GenIndex(object):
    # FuncNode object for indices registered with add_index_generator
    def __init__(self, gd, index_name):
        self.gd = gd
        self.idx = index_name

    def __call__(self):
        val = self.gd.gen_index_dims[self.idx]
        return tf.constant(val, dtype=tf.float32)

class CompIndex(object):
    # FuncNode object for indices registered with computed_index
    def __init__(self, gd, comp_func, const_args):
        self.gd = gd
        self.args = const_args
        self.func = comp_func

    def __call__(self, *args):
        # args will be index shapes, which change during gradient descent
        # const_args are non-index aruments and remain constant during
        # gradient descent
        const_args = tuple(self.gd.extra_args[a] for a in self.args)
        comp_dims = self.func(*args, *const_args)
        if not (
                (isinstance(comp_dims, (tf.Tensor, tf.Variable)) and
                    comp_dims.shape.rank == 1) or
                (isinstance(comp_dims, np.ndarray) and
                    comp_dims.ndim == 1)
                ):
            raise SchemaError(
                f'{type(self).__qualname__}: function \'{self.func.__name__}\' '
                f'registered with computed_dims must return a 1D '
                f'tf.Tensor or np.ndarray.  Got \'{comp_dims}\'')
        return comp_dims

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
            var = tf.Variable(val, constraint = cons_fun, dtype=tf.float32)
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
                        var.assign_sub(self.lr * grad)
                    # opt.apply_gradients(zip(grads, free_vars))
                    # if loss_name == 'elem_loss' and index_lambda() > 0.0:
                        # raise SchemaError(f'Nonzero index_loss')
            
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

        # create arg_shapes
        tests = []
        arg_shapes = {}
        for arg, sig in arg_sigs.items():
            arg_shapes[arg] = [ d for s in sig for d in index_dims[s] ]
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

class SecondInPair(object):
    """
    Return the second member of a tuple pair 
    """
    def __init__(self):
        pass

    def __call__(self, pair):
        return [pair[1]]

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
        data_format = rmap[rank]
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

