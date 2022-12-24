import sys
import numpy as np
import tensorflow as tf
from collections import defaultdict
from .error import *
from . import base, fgraph
from .fgraph import NodeFunc, node_name

"""
A collection of fgraph.NodeFunc derived classes for use in fgraph.PredNodes.
Each class implements the __call__ which is expected to return a tuple of
either (True, <value>) or (False, SchemaError).  See fgraph.PredNode for
details on how the predicate graph works.

The constructed Predicate Graph is used to evaluate all arguments to the
framework op, and either return a Success state, or various kinds of
SchemaError classes expressing the nature of the error.

The failure value for each of these NodeFuncs should be a list of suggestions
in cost-increasing order.  An empty list signifies that the predicate could not
find any suggestions which would fix the framework op inputs.
"""

class ErrorReport(object):
    def __init__(self, func, *info):
        self.func = func
        self.info = info

    def report(self):
        return self.func.user_msg(*self.info)

class ReportNodeFunc(NodeFunc):
    """
    Same role as ge.ReportNodeFunc, this allows reporting errors
    """
    def __init__(self, name=None):
        super().__init__(name)

    def user_msg(self, *info):
        """
        A message describing the constraint(s) defined
        """
        raise NotImplementedError

class NoSuggestionsFound(ReportNodeFunc):
    def __init__(self):
        super().__init__()

    def user_msg(self):
        return 'OpSchema found no suggestions'

class DataTensor(ReportNodeFunc):
    """
    Represent a tensor with data (as opposed to shape-based meta-data)
    """
    def __init__(self, arg_name, gen_node):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.gen_node = gen_node

    def user_msg(self, received_val):
        msg =  f'Tensor argument \'{self.arg_name}\' must be a tensor. '
        msg += f'Received {received_val}'
        return msg

    def __call__(self, op):
        ten = op._get_arg(self.arg_name)
        if not isinstance(ten, tf.Tensor):
            return False, ErrorReport(self, ten)
        else:
            return True, ten

class GetReturnTensors(ReportNodeFunc):
    def __init__(self):
        super().__init__(None)

    def user_msg(self, ret_index, received_val):
        msg =  f'Return {ret_index} expected to be a tensor.  Received '
        msg += f'{received_val}'
        return msg

    def __call__(self, op):
        for ridx, ten in enumerate(op.returns):
            if not isinstance(ten, tf.Tensor):
                return False, ErrorReport(self, ridx, ten)
        return True, op.returns

class ValidReturnShapes(ReportNodeFunc):
    def __init__(self):
        super().__init__()

    def user_msg(self, ret_index, act_shape, pred_shape):
        msg =  f'Return tensor {ret_index} was expected to have shape '
        msg += f'{pred_shape} but was {act_shape}'
        return msg

    def __call__(self, op, tensors):
        for ridx, tensor in enumerate(tensors):
            actual_shape = tensor.shape.as_list()
            ret_name = f'return[{ridx}]'
            pred_shape = op.inf_result.get_arg_shape(ret_name)
            if actual_shape == pred_shape:
                return True, None
            else:
                return False, ErrorReport(self, ridx, actual_shape, pred_shape)

class TensorDType(NodeFunc):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, tensor):
        return True, tensor.dtype.name

class TensorShape(NodeFunc):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, tensor):
        return True, tensor.shape.as_list()

class ShapeList(ReportNodeFunc):
    """
    Interpret the contents as a shape.
    """
    def __init__(self, arg_name, gen_node, broadcast_mode):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.gen_node = gen_node
        self.broadcast_mode = broadcast_mode

    def user_msg(self, received_val):
        if self.broadcast_mode:
            msg =  f'Argument \'{self.arg_name}\' must be a list of '
            msg += f'non-negative integers or a single non-negative integer. '
            msg += f'Received \'{received_val}\'.'
        else:
            msg =  f'Argument \'{self.arg_name}\' must be a list of '
            msg += f'non-negative integers.  Received \'{received_val}\'.'
        return msg

    def __call__(self, op):
        shape = op._get_arg(self.arg_name)
        err = ErrorReport(self, shape)

        if isinstance(shape, int) and self.broadcast_mode:
            if shape >= 0:
                return True, shape
            else:
                return False, err

        if not isinstance(shape, list):
            return False, err
        if not all(isinstance(v, int) for v in shape):
            return False, err
        if not all(v >= 0 for v in shape):
            return False, err
        else:
            # In broadcast mode, return an integer rather than integer list.
            if self.broadcast_mode and len(shape) == 1:
                shape = shape[0]
            return True, shape

class RangeCheck(object):
    def __init__(self, valid_kinds, min_val=None, max_val=None):
        self.valid_kinds = valid_kinds
        self.min_val = min_val
        self.max_val = max_val

    def predicate_msg(self):
        if self.min_val is not None:
            if self.max_val is not None:
                pred = f' in [{self.min_val}, {self.max_val}]'
            else:
                pred = f' >= {self.min_val}'
        else:
            if self.max_val is not None:
                pred = f' <= {self.max_val}'

        if self.valid_kinds is None:
            return pred

        kind_names = tuple(k.__name__ for k in self.valid_kinds)
        kinds_msg = base.list_phrase_or(kind_names)
        return f'({kinds_msg}) {pred}'

    def valid(self, val):
        lo = -int(1e10) if self.min_val is None else self.min_val
        hi = int(1e10) if self.max_val is None else self.max_val 
        return lo <= val <= hi and (
                self.valid_kinds is None or
                isinstance(val, self.valid_kinds)
                )
        
class ShapeInt(ReportNodeFunc):
    """
    Interpret the integer as a shape.
    """
    def __init__(self, arg_name, lo=None, hi=None):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.ranged = RangeCheck((int,), lo, hi)

    def user_msg(self, received_val):
        msg =  f'Argument \'{self.arg_name}\' expected to be '
        msg += self.ranged.predicate_msg()
        return msg

    def __call__(self, op):
        i = op._get_arg(self.arg_name)
        if self.ranged.valid(i):
            return True, [i]
        else:
            return False, ErrorReport(self, i)

class ShapeTensorFunc(ReportNodeFunc):
    """
    Interpret the tensor contents as a shape.
    Additionally, perform the checks defined by {pred_func}.
    {pred_func} accepts the integer list shape extracted from the tensor as
    well as any integer lists provided by *shapes.
    """
    def __init__(self, arg_name, gen_node, pred_func, lo, hi):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.gen_node = gen_node
        self.func = pred_func 
        self.ranged = RangeCheck(None, lo, hi)

    def user_msg(self, received_val):
        msg =  f'Argument \'{self.arg_name}\' expected to be an int32 tensor '
        msg += f'with non-negative elements. '
        if not isinstance(received_val, tf.Tensor):
            msg += 'Received a {type(received_val)} instead.'
        elif received_val.dtype != tf.int32:
            msg += 'Received dtype = {received_val.dtype.name}.'
        else:
            nums = received_val.numpy().tolist()
            if not all(self.ranged.valid(n) for n in nums):
                msg += f'Elements must be {self.ranged.predicate_msg()}'
        return msg

    def __call__(self, op, *shapes):
        ten = op._get_arg(self.arg_name)
        err = ErrorReport(self, ten)
        if not isinstance(ten, tf.Tensor) or ten.dtype != tf.int32:
            return False, err 
        else:
            nums = ten.numpy().tolist()
            if not all(self.ranged.valid(n) for n in nums):
                return False, err 
            else:
                try:
                    return self.func(nums, *shapes)
                except BaseException as ex:
                    raise SchemaError(
                        f'Predicate function for {self.arg_name} argument called as: '
                        f'func({nums}, {shapes}) raised an exception: {ex}')

class ShapeTensor(ShapeTensorFunc):
    """
    Specialization of ShapeTensorFunc that performs no additional checks
    """
    def __init__(self, arg_name, gen_node, lo, hi):
        pred_func = lambda shape: (True, shape)
        super().__init__(arg_name, gen_node, pred_func, lo, hi)

class ShapeTensor2D(ReportNodeFunc):
    """
    Validate a 2D shape tensor, and return its contents as a tuple of integer
    lists.
    """
    def __init__(self, arg_name, gen_node, num_slices):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.gen_node = gen_node
        self.num_slices = num_slices

    def user_msg(self, ten):
        msg =  f'Argument \'{self.arg_name}\' must be a rank 2 integer tensor '
        msg += f'with shape[1] == {self.num_slices} and all non-negative '
        msg += 'elements. '
        if not isinstance(ten, tf.Tensor):
            msg += f'Got type \'{type(ten)}\'. '
        elif not ten.dtype.is_integer:
            msg += f'Tensor dtype was \'{ten.dtype.name}\'. '
        elif ten.shape.rank != 2:
            msg += f'Tensor rank was \'{ten.shape.rank}\'. '
        elif ten.shape[1] != self.num_slices:
            msg += f'Tensor shape[1] was \'{ten.shape[1]}\'. '
        else:
            rows = ten.numpy()
            for row in rows:
                if any(el < 0 for el in row):
                    msg += f'One or more elements were negative.'
        return msg

    def __call__(self, op):
        ten = op._get_arg(self.arg_name) 
        err = ErrorReport(self, ten)
        if not isinstance(ten, tf.Tensor):
            return False, err
        elif not ten.dtype.is_integer:
            return False, err
        elif ten.shape.rank != 2:
            return False, err
        elif ten.shape[1] != self.num_slices:
            return False, err
        else:
            vals = tf.transpose(ten).numpy()
            for row in vals:
                if any(el < 0 for el in row):
                    return False, err
            tup = tuple(vals.tolist())
            return True, tup

class SliceShape(ReportNodeFunc):
    """
    Get a slice from a tuple of shapes.
    """
    def __init__(self, name, tup_index):
        super().__init__(f'{name}.{tup_index}')
        self.index = tup_index

    def __call__(self, shape_tup):
        vals = shape_tup[self.index]
        return True, vals 

class DTypes(NodeFunc):
    """
    Aggregate the outputs of TensorDType nodes.  Always succeeds
    """
    def __init__(self):
        super().__init__()

    def __call__(self, **dtypes_map):
        return True, dtypes_map

class ShapeMap(NodeFunc):
    """
    Produce a map of arg_name => shape 
    """
    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        shape_map = kwargs
        return True, shape_map

class SigMap(NodeFunc):
    """
    Aggregate all of the :sig nodes into a map of arg_name => sig
    """
    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        sig_map = kwargs
        return True, sig_map

class TupleElement(NodeFunc):
    """
    Extract one element from a tuple, always succeeds
    """
    def __init__(self, index):
        super().__init__()
        self.index = index

    def __call__(self, tup):
        return True, tup[self.index]

class GetShapes(TupleElement):
    """
    Get the (possibly broadcast-realized) Shapes from Inventory
    """
    def __init__(self):
        super().__init__(3)

class Inventory(NodeFunc):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def __call__(self, dtypes, obs_shapes, args):
        """
        Operates in two modes.
        In the first mode, avail_edits is zero:
            If successful, returns [Fix], which is a zero-cost "fix".
            If failed, returns the empty list
        Assume that self.op.avail_edits is set appropriately.
        """
        self.op._prep_inference(dtypes, obs_shapes, args)
        fixes = []

        all_nodes = set(self.op.inf_graph.values())
        exc_nodes = (self.op.obs_shapes, self.op.obs_dtypes, self.op.obs_args)
        live_nodes = all_nodes.difference(exc_nodes)
        out_nodes = (self.op.report_inode, )
        for fix in fgraph.gen_graph_values(live_nodes, out_nodes, self.op):
            fixes.append(fix[0]) 

        if self.op.avail_edits == 0:
            # If zero edits are possible, the single fix should be the
            # unique, zero-edit fix
            if len(fixes) == 1:
                return True, fixes
            elif len(fixes) > 1:
                fix_str = '\n\n'.join(repr(f) for f in fixes)
                raise SchemaError(
                    f'{type(self).__qualname__}: Got multiple matches with '
                    f'zero edits for framework op \'{self.op.op_path}\'\n'
                    f'Fixes:\n{fix_str}\n'
                    f'Observed shapes:\n{obs_shapes}\n'
                    )
            else:
                # no fixes found 
                return False, []
        else:
            return False, fixes

class DataFormat(ReportNodeFunc):
    def __init__(self, formats, gen_node, arg_name):
        super().__init__(arg_name)
        self.formats = formats
        self.arg_name = arg_name
        self.gen_node = gen_node

    def user_msg(self, received_val):
        msg =  f'Argument \'{self.arg_name}\' was expected to be one of '
        msg += f'{", ".join(self.formats.all_formats())}. '
        msg += f'Received \'{received_val}\''
        return msg

    def __call__(self, op):
        if self.arg_name is None:
            return True, self.formats.single()

        data_format = op._get_arg(self.arg_name)
        valid = (data_format in self.formats.all_formats())
        if valid:
            return True, data_format
        else:
            return False, ErrorReport(self, data_format)

class ArgInt(ReportNodeFunc):
    def __init__(self, arg_name, lo, hi):
        super().__init__(arg_name)
        self.arg_name = arg_name
        if lo is None:
            self.lo = -sys.maxsize - 1
        else:
            self.lo = lo
        if hi is None:
            self.hi = sys.maxsize
        else:
            self.hi = hi

    def user_msg(self, received_val):
        msg =  f'Argument \'{self.arg_name}\' was expected to be an integer '
        msg += f'in the range [{self.lo}, {self.hi}]. '
        msg += f'Received \'{received_val}\''
        return msg

    def __call__(self, op):
        arg_val = op._get_arg(self.arg_name) 
        err = ErrorReport(self, arg_val)
        if not isinstance(arg_val, int):
            return False, err
        elif arg_val not in range(self.lo, self.hi + 1):
            return False, err
        else:
            return True, arg_val

class Sig(NodeFunc):
    """
    Return an option associated with the layout
    """
    def __init__(self, name, sig_list):
        super().__init__(name)
        self.sig_list = sig_list

    def __call__(self, layout):
        return True, self.sig_list[layout]

class Options(ReportNodeFunc):
    def __init__(self, arg_name, gen_node, options):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.gen_node = gen_node
        try:
            iter(options)
        except TypeError:
            raise SchemaError(
                f'{type(self).__qualname__}: \'options\' argument must be '
                f'iterable.  Got {type(options)}')
        self.options = options

    def user_msg(self, received_val):
        msg =  f'Argument \'{self.arg_name}\' must be one of '
        msg += f'{", ".join(self.options)}. '
        msg += f'Received \'{received_val}\'.'
        return msg

    def __call__(self, op):
        arg_val = op._get_arg(self.arg_name)
        if arg_val in self.options:
            return True, arg_val
        else:
            return False, ErrorReport(self, arg_val)

class ArgMap(NodeFunc):
    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        return True, kwargs

class Schema(NodeFunc):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def __call__(self):
        return True, self.op

