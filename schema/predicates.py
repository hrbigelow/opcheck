import sys
import numpy as np
import tensorflow as tf
from collections import defaultdict
from .error import *
from . import util, base, fgraph
from .fgraph import NodeFunc, node_name
from .flib import Index
from .base import GenMode

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
class DataTensor(NodeFunc):
    """
    Represent a tensor with data (as opposed to shape-based meta-data)
    """
    def __init__(self, arg_name, gen_node):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.gen_node = gen_node

    def __call__(self, op):
        ten = op._get_arg(self.arg_name)
        if not isinstance(ten, tf.Tensor):
            return False, [[self.gen_node]]
        else:
            return True, ten

class GetReturnTensor(NodeFunc):
    def __init__(self, ret_name):
        super().__init__(ret_name)
        self.ret_name = ret_name

    def __call__(self, op):
        ten = op._get_return(self.ret_name)
        if not isinstance(ten, tf.Tensor):
            return False, [[self]] 
        else:
            return True, ten

class ValidReturnShape(NodeFunc):
    def __init__(self, ret_name):
        super().__init__(ret_name)
        self.ret_name = ret_name

    def __call__(self, tensor, shapes):
        actual_shape = tensor.shape.as_list()
        predicted_shape = shapes[self.ret_name]
        if actual_shape == predicted_shape:
            return True, None
        else:
            return False, [[self]] 

class TensorDType(NodeFunc):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, tensor):
        return True, tensor.dtype

class TensorShape(NodeFunc):
    def __init__(self, name):
        super().__init__(name)

    def __call__(self, tensor):
        return True, tensor.shape.as_list()

class ShapeList(NodeFunc):
    """
    Interpret the contents as a shape.
    """
    def __init__(self, arg_name, gen_node, broadcast_mode):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.gen_node = gen_node
        self.broadcast_mode = broadcast_mode

    def __call__(self, op):
        shape = op._get_arg(self.arg_name)
        if not isinstance(shape, list):
            return False, [[self.gen_node]] 
        if not all(isinstance(v, int) for v in shape):
            return False, [[self.gen_node]]
        if not all(v >= 0 for v in shape):
            return False, [[self.gen_node]]
        else:
            # In broadcast mode, return an integer rather than integer list.
            # see base.shape_rank
            if self.broadcast_mode and len(shape) == 1:
                shape = shape[0]
            return True, shape
        
class ShapeInt(NodeFunc):
    """
    Interpret the integer as a shape.
    """
    def __init__(self, arg_name, gen_node):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.gen_node = gen_node

    def __call__(self, op):
        i = op._get_arg(self.arg_name)
        if not isinstance(i, int):
            return False, [[self.gen_node]]
        if i < 0:
            return False, [[self.gen_node]]
        else:
            return True, [i]

class ShapeTensorFunc(NodeFunc):
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

    def __call__(self, op, *shapes):
        ten = op._get_arg(self.arg_name)
        if not isinstance(ten, tf.Tensor):
            return False, [[self.gen_node]]
        if ten.dtype != tf.int32:
            return False, [[self.gen_node]]
        nums = ten.numpy().tolist()
        if any(n < 0 for n in nums):
            return False, [[self.gen_node]]
        else:
            return self.func(nums, *shapes)

class ShapeTensor(ShapeTensorFunc):
    """
    Specialization of ShapeTensorFunc that performs no additional checks
    """
    def __init__(self, arg_name, gen_node):
        pred_func = lambda shape: (True, shape)
        super().__init__(arg_name, gen_node, pred_func)

class ShapeTensor2D(NodeFunc):
    """
    Validate a 2D shape tensor, and return its contents as a tuple of integer
    lists.
    """
    def __init__(self, arg_name, gen_node, num_slices):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.gen_node = gen_node
        self.num_slices = num_slices

    def __call__(self, op):
        ten = op._get_arg(self.arg_name) 
        if not ten.dtype.is_integer:
            return False, [[self.gen_node]]
        elif ten.shape.rank != 2:
            return False, [[self.gen_node]]
        elif ten.shape[1] != self.num_slices:
            return False, [[self.gen_node]]
        vals = tf.transpose(ten).numpy()
        tup = tuple(vals.tolist())
        return True, tup

# TODO: add gen_node
class SliceShape(NodeFunc):
    """
    Get a slice from a tuple of shapes.
    """
    def __init__(self, name, tup_index):
        super().__init__(f'{name}.{tup_index}')
        self.gen_node = gen_node
        self.index = tup_index

    def __call__(self, shape_tup):
        vals = shape_tup[self.index]
        if any(v < 0 for v in vals):
            return False, [[self]]
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

    def get_hits(self, max_edits, dtypes, shapes, args):
        self.op._prep_gen_inference(max_edits, dtypes, shapes, args)

        # ranks, dtypes, sigs, arg_shapes, error
        report_nodes = (self.op.error_node,)
        live_nodes = self.op.inference_nodes
        hits = []
        for hit in fgraph.gen_graph_values(live_nodes, report_nodes):
            hits.append(hit[0]) # extract from tuple
        return hits

    def __call__(self, dtypes, shapes, args):
        # prepare the inventory graph
        hits = self.get_hits(0, dtypes, shapes, args)
        if len(hits) > 1:
            raise SchemaError(
                f'{type(self).__qualname__}: Got multiple matches with '
                f'zero edits for framework op \'{self.op.op_path}\'')

        elif len(hits) == 1:
            return True, hits

        else:
            # produce up to some number of suggestions 
            min_hits = 1
            all_hits = []
            for dist in range(1, 3):
                hits = self.get_hits(dist, dtypes, shapes, args)
                all_hits.extend(hits)
                if len(all_hits) >= min_hits:
                    break
            return False, all_hits

def calc_sig_range(rank_map, idx, sig):
    ind = sig.index(idx)
    start = sum(rank_map[l] for l in sig[:ind])
    rank = rank_map[idx] 
    return [start, start + rank]

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

class TemplateFunc(NodeFunc):
    """
    Calls template_func(*inds, *extra)
    where inds are a set of OpGrind indices, and extra are any non-index arguments.

    See API computed_index for more details.

    Recursively calls any TemplateFunc nodes registered on parent indices, and
    accumulates the resulting texts and indices
    """
    def __init__(self, index_name, func, num_index_args, op):
        super().__init__(index_name)
        self.index = index_name
        self.func = func
        self.nidx = num_index_args
        self.op = op

    def __call__(self, comp_dims, **kwargs):
        keys = list(kwargs.keys()) # order preserved as of Python 3.6: (pep-468)
        idx_args = keys[:self.nidx]
        extra_args = keys[self.nidx:]

        formula = [] 
        indices = []

        # just takes the values as-is.  For indices, these are the dimensions
        computation = list(kwargs.values())
        ftexts = []
        ctexts = []

        calc_desc = self.op.index[self.index].replace(' ', '_')
        calc_comp = str(comp_dims)

        for idx in idx_args:
            v = kwargs[idx]
            desc = self.op.index[idx].replace(' ', '_')
            formula.append(desc)
            indices.append(idx)
            parent = self.op.comp_dims_templates.get(idx, None)
            if parent is not None:
                _, (ftext, ctext, inds) = parent.value()
                ftexts.extend(ftext)
                ctexts.extend(ctext)
                indices.extend(inds)

        formula.extend(kwargs[e] for e in extra_args)
        
        formula_text = calc_desc + ' = ' + self.func(*formula)
        computation_text = calc_comp + ' = ' + self.func(*computation)
        ftexts.append(formula_text)
        ctexts.append(computation_text)
        return True, (ftexts, ctexts, indices)

class DataFormat(NodeFunc):
    def __init__(self, formats, gen_node, arg_name):
        super().__init__(arg_name)
        self.formats = formats
        self.arg_name = arg_name
        self.gen_node = gen_node

    def __call__(self, op):
        if self.arg_name is None:
            return True, self.formats.default()

        data_format = op._get_arg(self.arg_name)
        valid = (data_format in self.formats.all_formats())
        if valid:
            return True, data_format
        else:
            return False, [[self.gen_node]]

class ArgInt(NodeFunc):
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

    def __call__(self, op):
        arg_val = op._get_arg(self.arg_name) 
        if not isinstance(arg_val, int):
            return False, [[self]]
        elif arg_val not in range(self.lo, self.hi + 1):
            return False, [[self]] 
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

class Options(NodeFunc):
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

    def __call__(self, op):
        arg_val = op._get_arg(self.arg_name)
        if arg_val in self.options:
            return True, arg_val
        else:
            return False, [[self.gen_node]]

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

