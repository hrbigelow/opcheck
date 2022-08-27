import tensorflow as tf
import numpy as np
import itertools
from random import randint
from functools import partial
from .base import Kind, kind, pfx
from . import util

class GenDTypes(object):
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

class GenRanks(object):
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

def pack_dims_map(rank_map, index_list, dims_list):
    """
    Computes a dims_map (idx => dims list) from the inputs.
    rank_map: idx => rank
    index_list: list of idx corresponding to the order of dims_list
    dims_list: flat dimensions.
    """
    dims_map = {}
    offset = 0
    for i, idx in enumerate(index_list):
        rank = rank_map[idx] 
        dims_map[idx] = dims_list[offset:offset+rank]
        offset += rank
    return dims_map 

class GenIndexDims(object):
    """
    Generate all (input and return) index dims
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

    def __call__(self, **kwargs):
        rank_map = kwargs[Kind.RANKS]
        sig_keys = [ k for k in kwargs.keys() if kind(k) == Kind.SIG ]
        sigs_map = { pfx(k): kwargs[k] for k in sig_keys }

        def nelem(rank_map, free_inds, calc_inds, flat_dims):
            free_dims_map = pack_dims_map(rank_map, free_inds, flat_dims)
            calc_dims_map = self.calc_dims(free_dims_map, kwargs) 
            dims_map = { **free_dims_map, **calc_dims_map } 
            sum_nelem = 0
            for sig in sigs_map.values():
                shape = [d for s in sig for d in dims_map[s]]
                sum_nelem += np.prod(shape)
            return sum_nelem

        calc_inds = list(self.comp_dims.funcs.keys())
        free_inds = [ k for k in rank_map.keys() if k not in calc_inds ]
        nelem_wrap = partial(nelem, rank_map, free_inds, calc_inds)

        k = sum(rank for idx, rank in rank_map.items() if idx in free_inds)
        min_nelem = 100000
        max_nelem = 200000
        dims = util.bsearch_integers(k, min_nelem, max_nelem, nelem_wrap)
        free_dims_map = pack_dims_map(rank_map, free_inds, dims)
        calc_dims_map = self.calc_dims(free_dims_map, kwargs) 
        dims_map = { **free_dims_map, **calc_dims_map } 
        return [dims_map]

class GenTensor(object):
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

class GenShape(object):
    """
    Generate the current shape of the input signature
    """
    def __init__(self):
        pass

    def __call__(self, dims_map, sig):
        shape = [ d for s in sig for d in dims_map[s] ]
        return [shape]

class GenSigRank(object):
    """
    """
    def __init__(self, sig, lo, hi):
        self.sig = sig
        self.lo = lo
        self.hi = hi

    def __call__(self, rank_map):
        rank = sum(rank_map[s] for s in self.sig)
        val = [randint(self.lo, self.hi+1) for _ in range(rank)]
        return [val]

