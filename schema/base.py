import tensorflow as tf
import numpy as np
from .error import SchemaError

def kpfx(kname):
    return kname.split(':')[0]

def kind(kname):
    idx = kname.index(':')
    return kname[idx:]

def kname(prefix, kind):
    return prefix+kind

def ksig(prefix):
    return prefix + Kind.SIG

def karg(prefix):
    return prefix + Kind.ARG

def kshp(prefix):
    return prefix + Kind.SHAPE

class Kind(object):
    # these cannot have prefixes
    SCHEMA = ':schema'
    DTYPES = ':dtypes'
    RANKS = ':ranks'
    IDIMS = ':input_dims'
    CDIMS = ':computed_dims'
    DIMS = ':dims'
    PSHAPE = ':predicated_shape'
    NONE = ':none'

    # these must have prefixes
    DTYPE = ':dtype'
    SIG = ':sig'
    # CONS = ':constraint'

    # these must have unique usages of prefixes
    ARG = ':arg'
    PSEUDO = ':pseudo'
    TENSOR = ':tensor'
    SHAPE = ':shape'
    RETURN_TENSOR = ':return_tensor'
    VALID_RETURN = ':valid_return'

class RankConstraints(object):
    def __init__(self, op):
        self.op = op

        # sig => max_rank
        self.maxs = {}

        # sig => min_rank
        self.mins = {}

        # index => index 
        self.equiv = {}

        # constraints applied during predicate
        # for generation, these are not applied, instead there exist
        # inverse functions that go in the opposite direction
        self.sig_funcs = {}
        self.sig_args = {}

        # a set of prefixes.  expect inputs of pfx:sig and pfx:shape 
        self.shape_sig = set()

    def free_inds(self):
        fi = [ k for k in self.op.index.keys() if k not in self.equiv ]
        return fi

    def inds(self, sig):
        fi = self.free_inds()
        map_sig = [ self.equiv.get(s, s) for s in sig ]
        return tuple(fi.index(m) for m in map_sig)

    def num_equated(self):
        return len(self.equiv)

    def index_limited(self, index):
        return index in self.mins or index in self.maxs

    def index_equated(self, index):
        return index in self.equiv
    
    def equate_ranks(self, target_index, source_index):
        self.equiv[target_index] = source_index

    def add_rank_limits(self, sig, min_val, max_val):
        if min_val is not None:
            prev_min_val = self.mins.get(sig, -1)
            self.mins[sig] = max(prev_min_val, min_val)
        if max_val is not None:
            prev_max_val = self.maxs.get(sig, 10000)
            self.maxs[sig] = min(prev_max_val, max_val)

    def add_shape_sig(self, prefix):
        self.shape_sig.add(prefix)

    def add_arg_rank(self, arg_name, sig):
        identity = lambda val: val
        node_kname = kname(arg_name, Kind.ARG)
        self.add_sig_func(sig, identity, (node_kname,))

    def add_sig_func(self, sig, func, arg_knames):
        self.sig_funcs[sig] = func
        self.sig_args[sig] = arg_knames 

    def mins_inds(self):
        d = self.mins.items()
        return { self.inds(sig): rank for sig,rank in d }

    def maxs_inds(self):
        d = self.maxs.items()
        return { self.inds(sig): rank for sig,rank in d }

    def const_inds(self, kwargs):
        # evaluate each sig_func, providing the 
        const_map = {}
        for sig, func in self.sig_funcs.items():
            arg_names = self.sig_args[sig]
            call_args = tuple(kwargs[a] for a in arg_names)
            rank = func(*call_args)
            inds = self.inds(sig) 
            const_map[inds] = rank

        # process the shape_sig entries.
        for prefix in self.shape_sig:
            shape = kwargs[kname(prefix, Kind.SHAPE)]
            sig = kwargs[kname(prefix, Kind.SIG)]
            inds = self.inds(sig)
            const_map[inds] = len(shape)

        return const_map

class DTypeConstraints(object):
    def __init__(self):
        self.valid = {}
        self.equiv = {}

    def add_valid(self, tensor_name, dtypes):
        self.valid[tensor_name] = tuple(dtypes)

    def add_equiv(self, target_tensor, source_tensor):
        self.equiv[target_tensor] = source_tensor

    def all(self):
        return (*self.valid, *self.equiv)

class CompDims(object):
    """
    Encapsulate the functions and arguments for computed index dimensions.
    The funcs are executed with tf.float32 tensor inputs and outputs, despite
    the fact that they are searching for integer dimensions.
    """
    def __init__(self):
        # idx => func
        self.funcs = {}

        # idx => arg_names
        self.args = {}

    def add(self, index, comp_func, arg_knames):
        """
        Register {index} to be computed by {comp_func}, taking {arg_names} as
        arguments
        """
        self.funcs[index] = comp_func
        self.args[index] = arg_knames

    def indices(self):
        return set(self.funcs.keys())

    def get_args(self):
        return { a for l in self.args.values() for a in l }

    def __call__(self, **kwargs):
        comp_dims_map = {}
        for index, func in self.funcs.items():
            arg_names = self.args[index]
            call_args = tuple(kwargs[a] for a in arg_names)
            comp_dims = func(*call_args)
            if not (
                    (isinstance(comp_dims, tf.Tensor) and
                        comp_dims.shape.rank == 1) or
                    (isinstance(comp_dims, np.ndarray) and
                        comp_dims.ndim == 1)
                    ):
                raise SchemaError(
                    f'{type(self).__qualname__}: function \'{func.__name__}\' '
                    f'registered with computed_dims must return a 1D '
                    f'tf.Tensor or np.ndarray.  Got \'{comp_dims}\'')
            comp_dims_map[index] = comp_dims
        return comp_dims_map


