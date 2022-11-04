import sys
import numpy as np
import tensorflow as tf
from collections import defaultdict
from .error import *
from . import base, fgraph
from .fgraph import NodeFunc, node_name, gen_graph_values

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
    def user_msg(self, *info):
        """
        A message describing the constraint(s) defined
        """
        raise NotImplementedError

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

class GetReturnTensor(ReportNodeFunc):
    def __init__(self, ret_name):
        super().__init__(ret_name)
        self.ret_name = ret_name

    def user_msg(self, received_val):
        msg =  f'{self.ret_name} expected to be a tensor.  Received '
        msg += f'{received_val}'
        return msg

    def __call__(self, op):
        ten = op._get_return(self.ret_name)
        if not isinstance(ten, tf.Tensor):
            return False, ErrorReport(self, ten)
        else:
            return True, ten

class ValidReturnShape(ReportNodeFunc):
    def __init__(self, ret_name):
        super().__init__(ret_name)
        self.ret_name = ret_name

    def user_msg(self, act_shape, pred_shape):
        msg =  f'Return tensor was expected to have shape {pred_shape} but '
        msg += f'was {act_shape}'
        return msg

    def __call__(self, tensor, shapes):
        actual_shape = tensor.shape.as_list()
        predicted_shape = shapes[self.ret_name]
        if actual_shape == predicted_shape:
            return True, None
        else:
            return False, ErrorReport(self, actual_shape, predicted_shape)

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
        
class ShapeInt(ReportNodeFunc):
    """
    Interpret the integer as a shape.
    """
    def __init__(self, arg_name, gen_node):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.gen_node = gen_node

    def user_msg(self, received_val):
        msg =  f'Argument \'{self.arg_name}\' expected to be a non-negative '
        msg += f'integer.  Received \'{received_val}\' instead.'
        return msg

    def __call__(self, op):
        i = op._get_arg(self.arg_name)
        if not isinstance(i, int) or i < 0:
            return False, ErrorReport(self, i)
        else:
            return True, [i]

class ShapeTensorFunc(ReportNodeFunc):
    """
    Interpret the tensor contents as a shape.
    Additionally, perform the checks defined by {pred_func}.
    {pred_func} accepts the integer list shape extracted from the tensor as
    well as any integer lists provided by *shapes.
    """
    def __init__(self, arg_name, gen_node, pred_func):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.gen_node = gen_node
        self.func = pred_func 

    def user_msg(self, received_val):
        msg =  f'Argument \'{self.arg_name}\' expected to be an int32 tensor '
        msg += f'with non-negative elements. '
        if not isinstance(received_val, tf.Tensor):
            msg += 'Received a {type(received_val)} instead.'
        elif received_val.dtype != tf.int32:
            msg += 'Received dtype = {received_val.dtype.name}.'
        else:
            nums = ten.numpy().tolist()
            if any(n < 0 for n in nums):
                msg += 'One or more elements were less than zero.'
        return msg

    def __call__(self, op, *shapes):
        ten = op._get_arg(self.arg_name)
        err = ErrorReport(self, ten)
        if not isinstance(ten, tf.Tensor) or ten.dtype != tf.int32:
            return False, err 
        else:
            nums = ten.numpy().tolist()
            if any(n < 0 for n in nums):
                return False, err 
            else:
                return self.func(nums, *shapes)

class ShapeTensor(ShapeTensorFunc):
    """
    Specialization of ShapeTensorFunc that performs no additional checks
    """
    def __init__(self, arg_name, gen_node):
        pred_func = lambda shape: (True, shape)
        super().__init__(arg_name, gen_node, pred_func)

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
        if not instance(ten, tf.Tensor):
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
        self.gen_node = gen_node
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
        for fix in gen_graph_values(live_nodes, (self.op.report_inode,)):
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

class IndexDimsConstraint(NodeFunc):
    """
    Constrains the dimensions of {indices} by applying the predicate function
    {status_func}, called as status_func(Index1, Index2, ...) where each Index
    object is an flib.Index object derived from one of the indices.
    """
    def __init__(self, name, status_func):
        super().__init__(name)
        self.func = status_func 

    def __call__(self, ranks, arg_sigs, shapes, op, **kwargs):
        shapes_list = []
        indices = []
        for idx, dims in kwargs.items():
            ind = Index(idx, op.index[idx], np.array(dims))
            shapes_list.append(ind)
            indices.append(idx)
        status = self.func(*shapes_list)
        if isinstance(status, Success):
            return True, status
        else:
            # The constraint was violated.  The explanatory message must now
            # include descriptions of any computed dims, either directly or
            # indirectly used by the constraint
            formula_texts = []
            dimension_texts = []
            ancestor_inds = list(indices)
            for idx, ind in zip(indices, shapes_list):
                tem = op.comp_dims_templates.get(idx, None)
                if tem is None:
                    continue
                _, (ftxts, dtxts, ainds) = tem.value()
                desc = op.index[idx].replace(' ', '_')
                formula_texts.extend(ftxts)
                dimension_texts.extend(dtxts)
                ancestor_inds.extend(ainds)

            # apply broadcasting rules to each index
            index_highlight = {}
            for idx in ancestor_inds:
                r = ranks[idx]
                if r == 1:
                    index_highlight[idx] = [True]
                elif r == len(status.mask):
                    index_highlight[idx] = status.mask
                else:
                    raise SchemaError(
                        f'All source indices of computed indices must be '
                        f'rank 1 or the same rank as the computed index. '
                        )

            # compile all texts into one message
            pairs = zip(formula_texts, dimension_texts)
            comp_dims_text = '\n\n'.join(f'{f}\n{d}\n' for f, d in pairs)
            if comp_dims_text != '':
                main_text = comp_dims_text + '\n' + status.text
            else:
                main_text = status.text
            err = IndexConstraintError(index_highlight, main_text, ranks,
                    arg_sigs, shapes)
            return False, err
        return valid, status


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
            return True, self.formats.default()

        data_format = op._get_arg(self.arg_name)
        valid = (data_format in self.formats.all_formats())
        if valid:
            return True, data_format
        else:
            return False, ErrorReport(self, data_format)

class ArgInt(ReportNodeFunc):
    def __init__(self, arg_name, lo, hi):
        super().__init__(f'{arg_name}[{lo}-{hi}]')
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

