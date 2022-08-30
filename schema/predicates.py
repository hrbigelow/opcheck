import sys
import tensorflow as tf
from .error import *
from . import util
from .base import Kind, kpfx, kname

"""
Functions or function objects to be used in PredNodes.  Implement the __call__
method which is expected to return one of:
    True, <value>
    False, SchemaError

where <value> is used by the function object in each child node.
"""
class GetType(object):
    def __init__(self, arg_name, allowed_types):
        self.arg_name = arg_name
        self.allowed_types = allowed_types

    def __call__(self, op):
        arg_val = op._get_arg(self.arg_name)
        if not isinstance(arg_val, self.allowed_types):
            return False, ArgTypeError(self.arg_name)
        else:
            return True, arg_val
    
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
    Will be called as pred_func(arg_val, *args)
    Kind.SCHEMA must be included as the first parent to this node.
    """
    def __init__(self, arg_name, pred_func):
        self.arg_name = arg_name
        self.pred_func = pred_func

    def __call__(self, op, *args):
        arg_val = op._get_arg(self.arg_name)
        return self.pred_func(arg_val, *args)


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

class Shape(object):
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, shape):
        if not all(isinstance(v, int) for v in shape):
            return False, ArgValueError(self.arg_name, shape)
        if not all(v >= 0 for v in shape):
            return False, ArgValueError(self.arg_name, shape)
        return True, shape

class ShapeInt(object):
    """
    Interpret the integer as a shape
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, i):
        if i < 0:
            return False, ArgValueError(self.arg_name, i)
        else:
            return True, i

class ShapeTensor(object):
    """
    Interpret the tensor contents as a shape
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, ten):
        if ten.dtype != tf.int32:
            return False, ArgValueError(self.arg_name, ten)
        nums = ten.numpy().tolist()
        if any(n < 0 for n in nums):
            return False, ArgValueError(self.arg_name, ten)
        return True, nums

class ShapeTensorSlice(object):
    """
    Extract a specific slice from a tensor
    """
    def __init__(self, arg_name, slice_index):
        self.arg_name = arg_name
        self.slice_index = slice_index

    def __call__(self, ten):
        if ten.dtype not in (tf.int32, tf.int64):
            return False, ArgValueError(self.arg_name, ten)
        if self.slice_index >= ten.shape[1]:
            # Tensor has wrong shape
            return False, ArgValueError(self.arg_name, ten)
        nums = ten[:,self.slice_index].numpy().tolist()
        if any(n < 0 for n in nums):
            return False, ArgValueError(self.arg_name, ten)
        return True, nums

class ValidDTypes(object):
    def __init__(self, dtype_cons):
        self.dtype_cons = dtype_cons

    def __call__(self, **kwargs):
        """Check that all tensor arguments have valid dtypes"""
        dtype_map = { kpfx(k): v for k,v in kwargs.items() }
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
    pfxs = { kpfx(k) for k in kwargs.keys() }
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

class ArgLayout(object):
    """
    Represent a data format layout argument
    """
    def __init__(self, arg_name, layouts):
        # layouts is list of map elememts, each one is: rank => layout
        self.arg_name = arg_name
        self.format_to_layout = {}
        for l, rmap in enumerate(layouts):
            for fmt in rmap.values():
                self.format_to_layout[fmt] = l

    def __call__(self, data_format):
        if data_format not in self.format_to_layout:
            return False, ArgValueError(self.arg_name, data_format)
        else:
            return True, self.format_to_layout[data_format]

class ArgDataFormat(object):
    def __init__(self, arg_name, layouts, rank_index):
        self.arg_name = arg_name
        self.layouts = layouts
        self.rank_index = rank_index

    def __call__(self, arg_val, rank_map, layout):
        rmap = self.layouts[layout]
        rank = rank_map[self.rank_index]
        data_format = rmap[rank]
        if arg_val == data_format:
            return True, arg_val
        else:
            return False, ArgValueError(self.arg_name, arg_val) 

class ArgInt(object):
    def __init__(self, arg_name, lo, hi):
        self.arg_name = arg_name
        if lo is None:
            self.lo = -sys.maxsize - 1
        else:
            self.lo = lo
        if hi is None:
            self.hi = sys.maxsize
        else:
            self.hi = hi

    def __call__(self, op):
        arg_val = op._get_arg(self.arg_name) 
        if not isinstance(arg_val, int):
            return False, ArgTypeError(self.arg_name)
        elif arg_val not in range(self.lo, self.hi + 1):
            return False, ArgValueError(self.arg_name)
        else:
            return True, arg_val

class LayoutOption(object):
    """
    Return an option associated with the layout
    """
    def __init__(self, arg_name, sig_list):
        self.arg_name = arg_name
        self.sig_list = sig_list

    def __call__(self, layout):
        return True, self.sig_list[layout]

