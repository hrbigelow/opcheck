import tensorflow as tf
from .error import *
from . import util
from .base import Kind, pfx, kname

"""
Functions or function objects to be used in PredNodes.  Implement the __call__
method which is expected to return one of:
    True, <value>
    False, SchemaError

where <value> is used by the function object in each child node.
"""
class GetTensor(object):
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, op):
        ten = op._get_arg(self.arg_name)
        if not isinstance(ten, tf.Tensor):
            return False, ArgTypeError(self.arg_name)
        else:
            return True, ten

class GetReturnTensor(object):
    def __init__(self, index):
        self.index = index

    def __call__(self, op):
        ten = op._get_return(self.index)
        if not isinstance(ten, tf.Tensor):
            return False, ArgTypeError(f'Return Tensor {self.index}')
        else:
            return True, ten

class ValidReturnShape(object):
    def __init__(self, index):
        self.index = index

    def __call__(self, tensor, predicted_shape):
        actual_shape = tensor.shape.as_list()
        if actual_shape == predicted_shape:
            return True, None
        else:
            return False, ReturnShapeError(self.index) 

class Sig(object):
    """
    Compute a signature using {sig_func} and optionally additional arguments.
    Argument names of sig_func are ignored and it is called with positional
    arguments.  Always succeeds.
    """
    def __init__(self, sig_func):
        self.sig_func = sig_func

    def __call__(self, *args):
        return True, self.sig_func(*args)

class ArgFunc(object):
    """
    Retrieve and validate the value of {arg_name} with {pred_func}.
    {pred_func} also accepts additional inputs.
    Kind.SCHEMA must be included as the first parent.
    """
    def __init__(self, arg_name, pred_func):
        self.arg_name = arg_name
        self.pred_func = pred_func

    def __call__(self, op, *args):
        arg_val = op._get_arg(self.arg_name)
        return self.pred_func(arg_val, *args)

class ArgInt(object):
    """
    Retrieve the value for {arg_name} and validate that it is an integer
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, op):
        arg_val = op._get_arg(self.arg_name)
        if isinstance(arg_val, int):
            return True, arg_val
        else:
            return False, ArgValueError(self.arg_name, arg_val) 

class ArgString(object):
    """
    Retrieve the value for {arg_name} and validate that it is a string
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, op):
        arg_val = op._get_arg(self.arg_name)
        if isinstance(arg_val, str):
            return True, arg_val
        else:
            return False, ArgValueError(self.arg_name, arg_val)

def dtype(tensor):
    return True, tensor.dtype

def tensor_shape(tensor):
    return True, tensor.shape.as_list()

def predicted_shape(idims_map, cdims_map, sig):
    dims_map = { **idims_map, **cdims_map }
    shape = [ d for s in sig for d in dims_map[s] ]
    return True, shape

class ArgShape(ArgFunc):
    """
    Retrieve the value for {arg_name} and validate it as a shape
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, op):
        shape = op._get_arg(self.arg_name)
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
        dtype_map = { pfx(k): v for k,v in kwargs.items() }
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

class IndexRanks(object):
    def __init__(self, op, rank_cons):
        self.op = op
        self.rcons = rank_cons

    def __call__(self, **kwargs):
        k = len(self.op.index)
        mins = self.rcons.mins_inds()
        maxs = self.rcons.maxs_inds()
        equiv = self.rcons.equiv_inds()
        const = self.rcons.const_inds(kwargs)
        rank_list = list(util.feasible_region(k, mins, maxs, equiv, const))

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

def get_sig_shape_map(kwargs):
    # convert a map containing: pfx:sig => sig, pfx:shape => shape to
    # sig => shape
    d = kwargs
    pfxs = { pfx(k) for k in kwargs.keys() }
    ss_map = { d[kname(p,Kind.SIG)] : d[kname(p,Kind.SHAPE)] for p in pfxs }
    return ss_map

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
            idx_shape = shape[slice(*sub_range)]
            idx_shapes.add(tuple(idx_shape))

        if len(idx_shapes) != 1:
            return False, IndexUsageError(idx)
        else:
            index_dims[idx] = idx_shapes.pop()
    return True, index_dims

class ComputedDims(object):
    def __init__(self, comp_dims):
        self.comp_dims = comp_dims

    def __call__(self, **kwargs):
        comp_dims_map = self.comp_dims(**kwargs)
        for idx, dims in comp_dims_map.items():
            if any(c < 0 for c in dims):
                return False, NegativeDimsError(idx, dims)
        return True, comp_dims_map

class SigRank(object):
    """
    Expect the value of {arg_name} to be a list of length equal to the rank of
    {sig}
    """
    def __init__(self, arg_name, sig):
        self.arg_name = arg_name
        self.sig = sig

    def __call__(self, op, rank_map):
        arg_val = op._get_arg(self.arg_name)
        rank = sum(rank_map[s] for s in self.sig)
        if len(arg_val) != rank:
            return False, SigRankError(arg_name, rank, len(arg_val))
        else:
            return True, arg_val

        
