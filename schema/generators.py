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
    def __init__(self, op, rank_cons):
        self.op = op
        self.rcons = rank_cons

    def __call__(self):
        """
        Generate all allowed rank combinations.  Generates a list of maps.
        Each map has index => rank for each index in self.index
        """
        mins = self.rcons.mins_inds()
        maxs = self.rcons.maxs_inds()
        equiv = self.rcons.equiv_inds()

        k = len(self.op.index)
        index_order = list(self.op.index.keys())
        gen = util.feasible_region(k, mins, maxs, equiv, {})
        rank_list = list(gen)
        return [dict(zip(index_order, ranks)) for ranks in rank_list]

class Sig(object):
    """
    Generate a single signature using {sig_func} and any additional arguments.
    Argument names are ignored so that the schema-writer doesn't need to know
    the Kind.* extensions
    """
    def __init__(self, sig_func):
        self.sig_func = sig_func

    def __call__(self, *args):
        return [self.sig_func(*args)]

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
    Generate dimensions for {index_combo} using {gen_func}.  Used in Kind.DIMS
    nodes.  Has parent Kind.RANKS

    Calls gen_func(ranks_list, *gen_args).  ranks_list are the ranks of each
    index in index_combo in order.

    returns a list of shape tuples, one shape for each index in index_combo.  A
    shape is an integer list.  

    For example, if index_combo has two indices, a return value could be:
    [ 
      ([1,2,3], [4,5]),
      ([6,4,2], [5,4]) 
    ]
    """
    def __init__(self, gen_func, index_combo, gen_args):
        self.indices = index_combo
        self.func = gen_func
        self.gen_args = gen_args

    def __call__(self, ranks_map):
        ranks_list = [ ranks_map[i] for i in self.indices ]
        vals = self.func(ranks_list, *self.gen_args)
        return [ dict(zip(self.indices,v)) for v in vals ]

class IndexDimsGD(object):
    """
    Perform gradient descent on two objectives to find suitable testing index
    dimensions.
    """
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
        val = tf.random.uniform([rank], lo, hi)
        cons_fun = lambda v: tf.clip_by_value(tf.round(v), 1.0, 1e10)
        var = tf.Variable(val, constraint = cons_fun, dtype=tf.float32)
        self.vars_map[index] = var

    def add_const(self, index, val):
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
        c = tf.concat(list(self.calc_map.values()), axis=0)
        loss = tf.reduce_sum(tf.nn.relu(1.0 - c))
        return loss

    def dims_map(self):
        m = {}
        all_map = { **self.vars_map, **self.const_map, **self.calc_map }
        for idx, var in all_map.items():
            m[idx] = var.numpy().astype(np.int32).tolist() 
        return m

    def __call__(self, **kwargs):
        """
        Expects args:
        :ranks - ranks_map defining all indexes and their ranks
        *:sig - set of all shape signatures to quantify 
        *:dims - dims_maps that have been pre-generated

        Perform gradient descent on all free indexes, such that:
        1.  all computed index dimensions are >= 1
        2.  the total number of elements over all signatures is within 

        free indexes are those indexes that are not pre-generated and not
        computed.
        """
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
        losses = ( 
            # ensures positive index dimensions 
            lambda kw: self.index_loss(kw), 
            # adjusts dims so total number of elements are within a range 
            lambda kw: self.elem_loss(kw) 
        )

        # The first phase (index_loss) ensures that all index dims are positive
        # (free, generated, and computed). Under that condition, the next
        # surface (elem_loss) is monotonically increasing in all free
        # dimensions.  And, because of the finite interval solution, zero loss
        # is achievable
        with tf.device('/device:CPU:0'):
            for loss_func in losses:
                for step in range(1000):
                    with tf.GradientTape() as tape:
                        objective = loss_func(kwargs)
                    free_vars = self.vars_map.values() 
                    grads = tape.gradient(objective, free_vars)
                    opt.apply_gradients(zip(grads, free_vars))
                    # print(f'loss: {step}: {objective:5.3f}')
                    if objective == 0.0:
                        break
            
        # extract the dims map
        dims_map = self.dims_map()
        return [dims_map]


class IndexDimsBSearch(object):
    """
    Generate free (input and return) index dims.
    call inputs:
    RANKS, all SIG nodes (input and return), and individual DIMS nodes
    """
    def __init__(self, comp_dims):
        self.comp_dims = comp_dims

    def calc_dims(self, idims_map, kwargs):
        arg_names = self.comp_dims.get_args()
        call = {}
        for a in arg_names:
            if a == Kind.IDIMS:
                call[a] = idims_map
            else:
                call[a] = kwargs[a]
        calc_dims_map = self.comp_dims(**call)
        return calc_dims_map

    def calc_inds(self):
        return list(self.comp_dims.funcs.keys())

    def input_dims_map(self, kwargs):
        dims_map = {}
        for k, v in kwargs.items():
            if kind(k) == Kind.DIMS: 
                dims_map.update(v)
        return dims_map

    def get_free_inds(self, rank_map, kwargs):
        all_inds = list(rank_map.keys())
        comp_inds = list(self.comp_dims.funcs.keys())
        input_inds = list(self.input_dims_map(kwargs))
        return [ i for i in all_inds if i not in comp_inds and i not in
            input_inds ] 

    def dims_map(self, rank_map, flat_dims, kwargs):
        offset = 0
        free_inds = self.get_free_inds(rank_map, kwargs)
        dims_map = {}
        for i, idx in enumerate(free_inds):
            rank = rank_map[idx] 
            dims_map[idx] = flat_dims[offset:offset+rank]
            offset += rank
        input_dims_map = self.input_dims_map(kwargs)
        dims_map.update(input_dims_map)
        calc_dims_map = self.calc_dims(dims_map, kwargs) 
        return { **dims_map, **calc_dims_map } 

    def nelem(self, flat_dims, kwargs):
        rank_map = kwargs[Kind.RANKS]
        sig_keys = [ k for k in kwargs.keys() if kind(k) == Kind.SIG ]
        sigs_map = { kpfx(k): kwargs[k] for k in sig_keys }
        dims_map = self.dims_map(rank_map, flat_dims, kwargs)

        # Any tensor with negative dimensions is considered to have a negative
        # number of elements.  Return -1 so that the system searches for a
        # higher value.  This relies on the objective function being
        # monotonically increasing in all flat_dims
        if any(d < 0 for sh in dims_map.values() for d in sh):
            return -1
        sum_nelem = 0
        for sig in sigs_map.values():
            shape = [d for s in sig for d in dims_map[s]]
            sum_nelem += np.prod(shape)
        return sum_nelem

    def __call__(self, **kwargs):
        def nelem_wrap(flat_dims):
            return self.nelem(flat_dims, kwargs)
        rank_map = kwargs[Kind.RANKS]
        free_inds = self.get_free_inds(rank_map, kwargs)
        k = sum(rank for idx, rank in rank_map.items() if idx in free_inds)
        min_nelem = 100000
        max_nelem = 200000
        print(', '.join(f'{k}{i}' for k in free_inds for i in range(1, rank_map[k]+1)))
        dims, niter = util.bsearch_integers(k, min_nelem, max_nelem, nelem_wrap)
        print('num iterations: ', niter)
        dims_map = self.dims_map(rank_map, dims, kwargs)
        return [dims_map]

class Tensor(object):
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, sig, dims_map, dtype_map):
        dtype = dtype_map[self.arg_name]
        shape = [ d for s in sig for d in dims_map[s] ]
        if dtype.is_integer:
            ten = tf.random.uniform(shape, minval=-10, maxval=10,
                    dtype=dtype)
        else:
            ten = tf.random.normal(shape, dtype=dtype)
        return [ten] 

class ShapeInt(object):
    """
    Generate the current shape of the input signature as an integer
    """
    def __init__(self):
        pass

    def __call__(self, dims_map, sig):
        shape = [ d for s in sig for d in dims_map[s] ]
        assert len(shape) == 1
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

class ShapeIndex(object):
    """
    Generate the shape of the given index
    """
    def __init__(self, index):
        self.index = index

    def __call__(self, dims_map):
        shape = dims_map[self.index]
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
        return tf.constant(rows, dtype=tf.int32)

class SigRank(object):
    """
    Generate an integer list of length equal to the rank of {sig}, whose
    elements lie in [lo, hi]
    """
    def __init__(self, sig, lo, hi):
        self.sig = sig
        self.lo = lo
        self.hi = hi

    def __call__(self, rank_map):
        rank = sum(rank_map[s] for s in self.sig)
        val = [randint(self.lo, self.hi) for _ in range(rank)]
        return [val]

class Layout(object):
    def __init__(self):
        pass

    def __call__(self):
        return [0, 1]

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

