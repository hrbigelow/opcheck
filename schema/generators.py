import sys
import math
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import itertools
from random import randint
from functools import partial
from .base import Kind, kind, kpfx
from .fgraph import FuncNode as F
from .error import SchemaError
from . import util

class DTypes(object):
    def __init__(self, dtype_cons):
        self.dtype_cons = dtype_cons

    def __call__(self):
        """
        Generate all allowed dtype combinations.  Generates a list of maps.
        Each map has a full tensor_name => dtype for each input tensor
        """
        # src_ten are tensor names which have independent dtypes
        src_ten, allowed_dtypes = zip(*self.dtype_cons.valid.items())
        # tensor_name => index 
        equiv_map = { trg: src_ten.index(src) for trg, src in
                self.dtype_cons.equiv.items() }
        equiv_map.update({v: i for i, v in enumerate(src_ten)})

        combos = []
        for combo in itertools.product(*allowed_dtypes):
            el = { name: combo[ind] for name,ind in equiv_map.items() }
            combos.append(el)
        return combos

class Ranks(object):
    def __init__(self, op, rank_candidates):
        self.op = op
        self.rcands = rank_candidates

    def __call__(self):
        """
        Generate all allowed rank combinations.  Generates a list of maps.
        Each map has index => rank for each index in self.index
        """
        return list(self.rcands.all_index_ranks())

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
        return self.gd.gen_index_dims[self.idx]

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
        return self.func(*args, *const_args)

class IndexDimsGD(object):
    """
    Perform gradient descent on two objectives to find suitable testing index
    dimensions.
    """
    def __init__(self, min_nelem, max_nelem):
        self.lr = 5.0
        self.log_min_nelem = math.log(min_nelem)
        self.log_max_nelem = math.log(max_nelem)
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

    def prepare(self, ranks, sigs, **kwargs):
        # split kwargs into index and non-index
        self.extra_args.clear()

        for k, v in kwargs.items():
            idx = kpfx(k)
            if kind(k) == Kind.GEN_DIMS:
                self.gen_index_dims[idx] = v
            else:
                self.extra_args[k] = v
        self.sigs = sigs
        for idx in self.variables.keys():
            rank = ranks[idx]
            # hard-coded reasonable choices for initialization
            val = tf.random.uniform([rank], 1.0, 4.0)
            cons_fun = lambda v: tf.clip_by_value(tf.round(v), 1.0, 1e10)
            var = tf.Variable(val, constraint = cons_fun, dtype=tf.float32)
            self.variables[idx] = var

    def all_dims(self):
        dims = { n.name: n.value() for n in self.nodes }
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
        dims = self.all_dims()
        comp_dims = [ dim for idx, dim in dims.items() if idx in
                self.computed_indexes ]
        if len(comp_dims) == 0:
            return 0
        st = tf.concat(comp_dims, axis=0)
        loss = tf.reduce_sum(tf.nn.relu(1.0 - st))
        return loss

    def elem_loss(self):
        dims = self.all_dims()
        nelem = tf.reduce_sum(tuple(self.num_elem(sig) for sig in self.sigs))
        log_nelem = tf.math.log(nelem)
        # the distance from interval [self.log_min_nelem, self.log_max_nelem]
        # done in log space since these are products of several variables
        # should it be k'th root instead, for k the maximum rank of a
        # signature?
        ival_dist = tf.maximum(
                tf.nn.relu(self.log_min_nelem - log_nelem),
                tf.nn.relu(log_nelem - self.log_max_nelem)
                )
        # should I use ival_dist**2 instead?
        return 2.0 * ival_dist

    def __call__(self, ranks, sigs, **kwargs):
        # kwargs should have:
        # all generated index dims declared with add_index_generator
        # all extra arguments declared in calls to computed_index
        self.prepare(ranks, sigs, **kwargs)
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        losses = [ lambda: self.index_loss(), lambda: self.elem_loss() ]

        # The first phase (index_loss) ensures that all index dims are positive
        # (free, generated, and computed). Under that condition, the next
        # surface (elem_loss) is monotonically increasing in all free
        # dimensions.  And, because of the finite interval solution, zero loss
        # is achievable
        with tf.device('/device:CPU:0'):
            for loss_func in losses:
                for step in range(100):
                    with tf.GradientTape() as tape:
                        objective = loss_func()
                    free_vars = self.variables.values()
                    grads = tape.gradient(objective, free_vars)
                    opt.apply_gradients(zip(grads, free_vars))
                    if step % 10 == 0:
                        print(f'loss: {step}: {objective:5.3f}')
                    if objective == 0.0:
                        break
            
        # extract the dims map
        dims = self.calc_dims()
        return [dims]

"""
class IndexDimsGD(object):
    def __init__(self, comp_dims, min_nelem, max_nelem):
        self.lr = 5.0
        self.comp_dims = comp_dims
        self.log_min_nelem = math.log(min_nelem)
        self.log_max_nelem = math.log(max_nelem)
        self.vars_map = {}
        self.const_map = {}
        self.calc_map = {}
        self.sigs = []

    def calc_dims(self, kwargs):
        idims_map = { **self.vars_map, **self.const_map }
        arg_names = self.comp_dims.get_args()
        call = {}
        for a in arg_names:
            if a == Kind.IDIMS:
                call[a] = idims_map
            else:
                call[a] = kwargs[a]
        self.calc_map = self.comp_dims(**call)

    def add_var(self, index, lo, hi, rank):
        with tf.device('/device:CPU:0'):
            val = tf.random.uniform([rank], lo, hi)
            cons_fun = lambda v: tf.clip_by_value(tf.round(v), 1.0, 1e10)
            var = tf.Variable(val, constraint = cons_fun, dtype=tf.float32)
            self.vars_map[index] = var

    def add_const(self, index, val):
        with tf.device('/device:CPU:0'):
            self.const_map[index] = tf.constant(val, dtype=tf.float32)

    def num_elem(self, sig):
        all_map = { **self.vars_map, **self.const_map, **self.calc_map }
        v = [all_map[s] for s in sig]
        c = tf.concat(v, axis=0) 
        nelem = tf.reduce_min(tf.sign(c)) * tf.abs(tf.reduce_prod(c))
        return nelem

    def elem_loss(self, kwargs):
        self.calc_dims(kwargs)
        nelem = tf.reduce_sum(tuple(self.num_elem(s) for s in self.sigs))
        log_nelem = tf.math.log(nelem)
        # the distance from interval [self.log_min_nelem, self.log_max_nelem]
        # done in log space since these are products of several variables
        ival_dist = tf.maximum(
                tf.nn.relu(self.log_min_nelem - log_nelem),
                tf.nn.relu(log_nelem - self.log_max_nelem)
                )

        # should I use ival_dist**2 instead?
        return 2.0 * ival_dist

    def index_loss(self, kwargs):
        # ensure all computed dimensions are >= 1
        self.calc_dims(kwargs)
        v = list(self.calc_map.values())
        if len(v) == 0:
            raise SchemaError(
                f'Cannot use index_loss if no computed dims exist')
        c = tf.concat(v, axis=0)
        loss = tf.reduce_sum(tf.nn.relu(1.0 - c))
        return loss

    def dims_map(self):
        m = {}
        all_map = { **self.vars_map, **self.const_map, **self.calc_map }
        for idx, var in all_map.items():
            m[idx] = var.numpy().astype(np.int32).tolist() 
        return m

    def __call__(self, **kwargs):
        # Expects args:
        # :ranks - ranks_map defining all indexes and their ranks
        # *:sig - set of all shape signatures to quantify 
        # *:dims - dims_maps that have been pre-generated

        # Perform gradient descent on all free indexes, such that:
        # 1.  all computed index dimensions are >= 1
        # 2.  the total number of elements over all signatures is within certain
            # bounds

        # free indexes are those indexes that are not pre-generated and not
        # computed.
        self.vars_map.clear()
        self.const_map.clear()
        self.sigs.clear()

        input_dims_map = {}
        for k, v in kwargs.items():
            if kind(k) == Kind.DIMS: 
                input_dims_map.update(v)
        for i, v in input_dims_map.items():
            self.add_const(i, v)

        ranks_map = kwargs[Kind.RANKS]
        all_inds = set(ranks_map.keys())
        computed_inds = self.comp_dims.indices()
        pregen_inds = set(input_dims_map.keys())
        free_inds = all_inds.difference(*computed_inds, *pregen_inds)
        for i in free_inds:
            rank = ranks_map[i]
            self.add_var(i, 1.0, 4.0, rank)

        for k, v in kwargs.items():
            if kind(k) == Kind.SIG:
                self.sigs.append(v)
        # TODO: exclude unnecessary sigs of single indices
        # TODO: include sigs for return tensors

        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        losses = []
        if len(computed_inds) != 0:
            # ensure computed inds are >= 1
            losses.append(lambda kw: self.index_loss(kw))

        # adjusts dims so total number of elements are within a range 
        losses.append(lambda kw: self.elem_loss(kw))
            

        # The first phase (index_loss) ensures that all index dims are positive
        # (free, generated, and computed). Under that condition, the next
        # surface (elem_loss) is monotonically increasing in all free
        # dimensions.  And, because of the finite interval solution, zero loss
        # is achievable
        with tf.device('/device:CPU:0'):
            for loss_func in losses:
                for step in range(100):
                    with tf.GradientTape() as tape:
                        objective = loss_func(kwargs)
                    free_vars = self.vars_map.values() 
                    grads = tape.gradient(objective, free_vars)
                    opt.apply_gradients(zip(grads, free_vars))
                    if step % 10 == 0:
                        print(f'loss: {step}: {objective:5.3f}')
                    if objective == 0.0:
                        break
            
        # extract the dims map
        dims_map = self.dims_map()
        return [dims_map]
"""

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

    def __call__(self, dims_map, sig, **kwargs):
        dtype_map = kwargs[Kind.DTYPES]
        dtype = dtype_map[self.arg_name]
        shape = [ d for s in sig for d in dims_map[s] ]
        return [(shape, dtype)]

def from_stub(stub):
    shape, dtype = stub
    if dtype.is_integer:
        ten = tf.random.uniform(shape, minval=-10, maxval=10,
                dtype=dtype)
    else:
        ten = tf.random.normal(shape, dtype=dtype)
    return ten

class ShapeInt(object):
    """
    Expect the shape of index to be rank 1 and extract the first component as
    an integer.  
    """
    def __init__(self):
        pass

    def __call__(self, dims_map, index):
        shape = dims_map[index]
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
    def __init__(self):
        pass

    def __call__(self, dims_map, sig):
        shape = [ d for s in sig for d in dims_map[s] ]
        return [shape]

class ShapeTensor(object):
    """
    Generate the current shape of the input signature as a tensor
    """
    def __init__(self):
        pass

    def __call__(self, dims_map, sig):
        shape = [ d for s in sig for d in dims_map[s] ]
        ten = tf.constant(shape, dtype=tf.int32)
        return [ten]

class ShapeTensor2D(object):
    """
    Generate a 2D tensor from dims and a list of signatures
    """
    def __init__(self):
        pass

    def __call__(self, dims_map, *sigs):
        rows = []
        for sig in sigs:
            shape = [ d for s in sig for d in dims_map[s] ]
            rows.append(shape)
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

