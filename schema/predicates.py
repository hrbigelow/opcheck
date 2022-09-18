import sys
import numpy as np
import tensorflow as tf
from collections import defaultdict
from .error import *
from . import util
from .base import Kind, kind, kpfx, kname

"""
Functions or function objects to be used in PredNodes.  Implement the __call__
method which is expected to return one of:
    True, <value>
    False, SchemaError

where <value> is used by the function object in each child node.
"""
class ArgType(object):
    """
    Validate that the argument is of allowed type.  Used in Kind.ARG nodes.
    """
    def __init__(self, arg_name, allowed_types):
        self.arg_name = arg_name
        self.allowed_types = allowed_types

    def __call__(self, op):
        arg_val = op._get_arg(self.arg_name)
        if not isinstance(arg_val, self.allowed_types):
            return False, ArgTypeError(self.arg_name)
        else:
            return True, arg_val

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

def dtype(tensor):
    return True, tensor.dtype

def tensor_shape(tensor):
    return True, tensor.shape.as_list()

def predicted_shape(idims_map, cdims_map, sig):
    dims_map = { **idims_map, **cdims_map }
    shape = [ d for s in sig for d in dims_map[s] ]
    return True, shape

class ShapeList(object):
    """
    Interpret the contents as a shape.  Used in Kind.SHAPE nodes 
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, shape, *shapes):
        if not all(isinstance(v, int) for v in shape):
            return False, ArgValueError(self.arg_name, shape)
        if not all(v >= 0 for v in shape):
            return False, ArgValueError(self.arg_name, shape)
        else:
            return True, shape
        
class ShapeInt(object):
    """
    Interpret the integer as a shape.  Used in Kind.SHAPE nodes
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, i):
        if i < 0:
            return False, ArgValueError(self.arg_name, i)
        else:
            return True, [i]

class ShapeTensorFunc(object):
    """
    Interpret the tensor contents as a shape.
    Additionally, perform the checks defined by {pred_func}.
    {pred_func} accepts the integer list shape extracted from the tensor as
    well as any integer lists provided by *shapes.
    """
    def __init__(self, arg_name, pred_func):
        self.arg_name = arg_name
        self.func = pred_func 

    def __call__(self, ten, *shapes):
        if ten.dtype != tf.int32:
            return False, ArgValueError(self.arg_name, ten)
        nums = ten.numpy().tolist()
        if any(n < 0 for n in nums):
            return False, ArgValueError(self.arg_name, ten)
        else:
            return self.func(nums, *shapes)

class ShapeTensor(ShapeTensorFunc):
    """
    Specialization of ShapeTensorFunc that performs no additional checks
    """
    def __init__(self, arg_name):
        pred_func = lambda shape: (True, shape)
        super().__init__(arg_name, pred_func)

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
        elif ten.shape.rank != 2:
            return False, ArgValueError(self.arg_name, ten)  
        elif self.slice_index >= ten.shape[1]:
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
                return False, DTypeNotAllowed(ten_name, dtype, valid_dtypes)
        for trg_name, src_name in self.dtype_cons.equiv.items():
            src_dtype = dtype_map[src_name]
            trg_dtype = dtype_map[trg_name]
            if trg_dtype != src_dtype:
                stat = DTypeNotEqual(src_name, src_dtype, trg_name, trg_dtype)
                return False, stat
        return True, None

class ShapeMap(object):
    """
    Produce a map of arg_name => shape 
    """
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        # convert a map containing: pfx:sig => sig, pfx:shape => shape to
        # sig => shape
        shape_map = {}
        for kn, val in kwargs.items():
            prefix = kpfx(kn)
            suffix = kind(kn)
            if suffix == Kind.SHAPE:
                shape_map[prefix] = val 
        return True, shape_map

class SigMap(object):
    """
    Aggregate all of the :sig nodes into a map of arg_name => sig
    """
    def __init__(self):
        pass

    def __call__(self, **kwargs):
        sig_map = {}
        for kn, val in kwargs.items():
            prefix = kpfx(kn)
            suffix = kind(kn)
            if suffix == Kind.SIG:
                sig_map[prefix] = val 
        return True, sig_map

class IndexRanks(object):
    """
    Search the set of all valid index rank combinations, combined with all
    signature/layout combinations.

    For each combination, and for each constraint, compute rank_error,
    highlight_map, and suggestions, and accumulate them in an array.

    If exactly one index rank combination plus signature/layout is found to
    have zero rank errors over all the constraints, succeed, and return the
    index rank combination.

    If none are found, produce a NoMatchingRanks error with a selected set of
    'best' candidates (with minimal errors).

    """
    def __init__(self, op, rank_candidates, rank_cons):
        self.op = op
        self.cands = rank_candidates
        self.rcons = rank_cons

    def __call__(self, shape_map, **kwargs):
        fields = ['format', 'sigs', 'ranks', 'suggestions', 'highlight']
        Candidate = namedtuple('Candidate', fields)

        valid_map = None
        candidates = []

        pseudo_kname = kname('layout', Kind.PSEUDO)
        data_format, layout = kwargs.get(pseudo_kname, (None, None))
        # replace this with a single generative node which generates
        # Ranks, Sigs, Layout 
        for cand_ranks, sig_map, cand_format in self.op._ranks_sigs_format():
            if ((data_format is None and cand_format is None) or 
                    (data_format == cand_format)):
                layout_delta = 0
            else:
                layout_delta = 1

            deltas = []
            suggestions = []
            highlight_map = defaultdict(list)
            for c in self.rcons:
                delta = c.rank_error(sig_map, shape_map, cand_ranks,
                        **kwargs)
                deltas.append(delta)
                sug = c.suggestion(delta)
                if sug is not None:
                    suggestions.append(sug)
                if delta != 0:
                    hl = c.highlight_map(sig_map, shape_map, cand_ranks)
                    for arg, pos in hl.items():
                        highlight_map[arg].extend(pos)

            if layout_delta == 1:
                sug = f'Use layout {cand_format}'
                suggestions.append(sug)
                # highlight the argument itself
                highlight_map[Kind.DATA_FORMAT].append(0)

            if all(d == 0 for d in deltas) and layout_delta == 0:
                if valid_map is None:
                    valid_map = cand_ranks
                else:
                    raise SchemaError(
                        f'{type(self).__qualname__}: multiple valid rank '
                        f'combinations were found.  This means that schema'
                        f' \'{self.op.op_path}\' lacks proper rank '
                        f'constraints.  Current constraints are:\n'
                        f'{", ".join(c.name for c in self.rcons)}')

            cand = Candidate(cand_format, sig_map, cand_ranks, suggestions,
                    highlight_map)
            candidates.append((deltas, layout_delta, cand))

        def key(cand):
            delta, layout_delta, _ = cand
            num_errors = len(list(d for d in delta if d != 0)) + layout_delta
            tot_error = sum(abs(d) for d in delta) + layout_delta
            return num_errors, tot_error

        def filt(cand):
            num_errors, tot_error = key(cand)
            return num_errors < 3 and tot_error < 4

        if valid_map is None:
            filtered = filter(filt, candidates)
            top = sorted(filtered, key=key)
            report = [ t[2] for t in top ]
            return False, NoMatchingRanks(shape_map, data_format, report)
        else:
            return True, valid_map 

def calc_sig_range(rank_map, idx, sig):
    ind = sig.index(idx)
    start = sum(rank_map[l] for l in sig[:ind])
    rank = rank_map[idx] 
    return [start, start + rank]

class IndexDims(object):
    """
    Used by the IDIMS node
    """
    def __init__(self):
        pass

    def __call__(self, ranks, sigs, shapes):
        def nextn(it, n):
            return [ next(it) for _ in range(n) ]

        # for each index and each component, produce a usage map 
        idx_usage = {} 
                       # idx => [(dim => [arg1, ...]), 
                       #         (dim => [arg1, ...]), 
                       #         ...]
        for arg, sig in sigs.items():
            shape = shapes[arg]
            it = iter(shape)
            for idx in sig:
                r = ranks[idx]
                if idx not in idx_usage:
                    idx_usage[idx] = [defaultdict(list) for _ in range(r)]
                dims = nextn(it, r)
                for c, dim in enumerate(dims):
                    idx_usage[idx][c][dim].append(arg)
                
        idx_dims = {}
        for idx in list(idx_usage):
            comp = idx_usage[idx]
            if all(len(c) == 1 for c in comp):
                idx_usage.pop(idx)
                idx_dims[idx] = [i for c in comp for i in c] 
        
        if len(idx_usage) != 0:
            return False, IndexUsageError(idx_usage, ranks, sigs, shapes)
        else:
            return True, idx_dims

class Dims(object):
    """
    Validate the combination of index shapes using the enclosed function.
    status_func accepts the shapes of indices in index_combo in order.
    It returns an instance of SchemaStatus.
    """
    def __init__(self, name, status_func, index_combo):
        self.name = name
        self.func = status_func 
        self.indices = index_combo

    def __call__(self, idims_map, cdims_map):
        dims_map = { **idims_map, **cdims_map }
        shapes_list = [ np.array(dims_map[i]) for i in self.indices ]
        status = self.func(*shapes_list)
        valid = isinstance(status, Success)
        return valid, status

class ComputedDims(object):
    def __init__(self, comp_dims):
        self.comp_dims = comp_dims

    def __call__(self, dims_map, **kwargs):
        # convert dims tuples to np.arrays
        dims_map = { k: np.array(v) for k, v in dims_map.items() }
        kwargs[Kind.IDIMS] = dims_map

        comp_dims_map = self.comp_dims(**kwargs)
        # convert back to integer tuples
        comp_dims_map = { 
                k: tuple(v.astype(np.int32).tolist()) 
                for k, v in comp_dims_map.items() 
                }
        for idx, dims in comp_dims_map.items():
            if any(c < 0 for c in dims):
                return False, NegativeDimsError(idx, dims)
        return True, comp_dims_map

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
            return True, (data_format, self.format_to_layout[data_format])

class ArgDataFormat(object):
    def __init__(self, arg_name, layouts, rank_index):
        self.arg_name = arg_name
        self.layouts = layouts
        self.rank_index = rank_index

    def __call__(self, arg_val, rank_map, format_layout):
        layout = format_layout[1]
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

    def __call__(self, format_layout):
        _, layout = format_layout 
        return True, self.sig_list[layout]

class Closure(object):
    def __init__(self, obj):
        self.obj = obj

    def __call__(self):
        return self.obj

