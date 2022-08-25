from .error import *
from . import util

"""
Functions or function objects to be used in PredNodes
"""
def get_prefix(node_name):
    return split(':', node_name)[0]

def get_kind(node_name):
    return split(':', node_name)[1]

def name(prefix, kind):
    return f'{prefix}:{kind}'

class Kind(object):
    SCHEMA = 'schema'
    TENSOR = 'tensor'
    DTYPE = 'dtype'
    SHAPE = 'shape'
    SIG = 'sig'
    DTYPES = 'dtypes'
    IRANKS = 'index_ranks'
    IDIMS = 'input_index_dims'
    CDIMS = 'computed_dims'
    DIMS = 'all_dims'
    PSEUDO = 'pseudo'
    PSHAPE = 'predicated_shape'
    CONS = 'constraint'
    ARG = 'arg'

class GetTensor(object):
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, op):
        ten = op.get_arg(self.arg_name)
        if not isinstance(ten, tf.Tensor):
            return False, ArgTypeError(self.arg_name)
        else:
            return True, ten

class Sig(object):
    """
    Compute a signature using {sig_func} and optionally additional arguments.
    Always succeeds
    """
    def __init__(self, sig_func):
        self.sig_func = sig_func

    def __call__(self, **kwargs):
        return True, self.sig_func(**kwargs)

class ArgFunc(object):
    """
    Retrieve and validate the value of {arg_name} with {pred_func}.
    {pred_func} also accepts additional inputs
    """
    def __init__(self, arg_name, pred_func):
        self.arg_name = arg_name
        self.pred_func = pred_func

    def __call__(self, op, **kwargs):
        arg_val = op.get_arg(self.arg_name)
        return self.pred_func(arg_val, **kwargs)

def dtype(tensor):
    return True, tensor.dtype

def shape(tensor):
    return True, tensor.shape.as_list()

class ArgShape(ArgFunc):
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, op):
        shape = op.get_arg(self.arg_name)
        if not isinstance(shape, (tuple, list)):
            return False, ArgTypeError(self.arg_name, 'must be tuple or list')
        if not all(isinstance(v, int) for v in shape):
            return False, ArgValueError(self.arg_name, shape)
        if not all(v >= 0 for v in shape):
            return False, ArgValueError(self.arg_name, shape)
        return True, shape

class ValidDTypes(object):
    def __init__(self, dtype_cons):
        self.dtype_cons = dtype_cons

    def __call__(self, **kwargs):
        """Check that all tensor arguments have valid dtypes"""
        dtype_map = { get_prefix(k): v for k,v in kwargs.items() }
        assert (len(dtype_map) == len(kwargs)), 'ValidDTypes internal error'
        
        for ten_name, valid_dtypes in self.dtype_cons.valid.items():
            dtype = dtype_map[ten_name]
            if dtype not in valid_dtypes:
                return False, DTypeNotAllowed(ten_name, dtype)
        for trg_name, src_name in self.dtype_cons.equiv.items():
            src_dtype = dtype_map[src_name]
            trg_dtype = dtype_map[trg_name]
            if trg_dtype != src_dtype:
                return False, DTypeNotEqual(src_name, trg_name)
        return True, None

def get_sig_shape_map(kwargs):
    # convert a map of: pfx:sig => sig, pfx:shape => shape to
    # sig => shape
    pfxs = { get_prefix(k) for k in kwargs.keys() }
    ss_map = { d[f'{p}:{Kind.SIG}'] : d[f'{p}:{Kind.SHAPE}'] for p in pfxs }
    return ss_map

class IndexRanks(object):
    def __init__(self, op, rank_cons):
        self.op = op
        self.rcons = rank_cons

    def __call__(self, **kwargs):
        sig_shape_map = get_sig_shape_map(kwargs)
        const_map = {}
        for sig, shape in sig_shape_map.items():
            sig_inds = self.op.sig_indices(sig)
            const_map[sig_inds] = len(shape)

        k = len(self.op.index)
        rmins = self.rcons.rmins_inds()
        rmaxs = self.rcons.rmaxs_inds()
        req = self.rcons.req_inds()
        rank_list = list(util.feasible_region(k, rmins, rmaxs, req, const_map))

        if len(rank_list) == 0:
            return False, NoMatchingRanks()
        elif len(rank_list) > 1:
            return False, AmbiguousRanks()
        else:
            index_ranks = dict(zip(self.op.index.keys(), rank_list[0]))
            return True, index_ranks

def calc_sig_range(rank_map, idx, sig):
    ind = sig.index(idx)
    start = sum(rank_map[l] for l in sig[:ind])
    rank = rank_map[idx] 
    return [start, start + rank]

def input_index_dims(rank_map, **kwargs):
    sig_shape = get_sig_shape_map(kwargs)
    input_inds = { idx for sig in sig_shape.keys() for idx in sig }
    index_dims = {}
    for idx in input_inds:
        # find all usages of idx
        idx_shapes = set()
        for sig, shape in sig_shape.items():
            if idx not in sig:
                continue
            sub_range = calc_sig_range(rank_map, idx, sig)
            idx_shape = dims[slice(*sub_range)]
            idx_shapes.add(tuple(idx_shape))

        if len(idx_shapes) != 1:
            return False, IndexUsageError(idx)
        else:
            index_dims[idx] = idx_shapes.pop()
    return True, index_dims

class ComputedDims(object):
    def __init__(self, target_index, func):
        self.index = target_index
        self.func = func

    def __call__(self, dims_map, **kwargs):
        comp_dims = self.func(dims_map, **kwargs)
        if all(c >= 0 for c in comp_dims):
            return True, { self.index: comp_dims }
        else:
            return False, NegativeDimsError(self.index, comp_dims)

