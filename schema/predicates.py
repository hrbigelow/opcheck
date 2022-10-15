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
"""
# TODO: Specialize ArgType for DataTensor, ShapeTensor etc
class ArgType(NodeFunc):
    """
    Validate that the argument is of allowed type.
    """
    def __init__(self, arg_name, allowed_types):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.allowed_types = allowed_types

    def __call__(self, op):
        arg_val = op._get_arg(self.arg_name)
        if not isinstance(arg_val, self.allowed_types):
            return False, ArgTypeError(self.arg_name)
        else:
            return True, arg_val

class DataTensor(ArgType):
    """
    Represent a tensor with data (as opposed to shape-based meta-data)
    """
    def __init__(self, arg_name):
        super().__init__(arg_name, tf.Tensor) 

    def __call__(self, op):
        return super().__call__(op)

class GetReturnTensor(NodeFunc):
    def __init__(self, ret_name):
        super().__init__(ret_name)
        self.ret_name = ret_name

    def __call__(self, op):
        ten = op._get_return(self.ret_name)
        if not isinstance(ten, tf.Tensor):
            return False, ArgTypeError(f'Return Tensor {self.ret_name}')
        else:
            return True, ten

class ValidReturnShape(NodeFunc):
    def __init__(self, ret_name):
        super().__init__(ret_name)
        self.ret_name = ret_name

    def __call__(self, tensor, predicted_shape):
        actual_shape = tensor.shape.as_list()
        if actual_shape == predicted_shape:
            return True, None
        else:
            return False, ReturnShapeError(self.ret_name) 

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

class PredictedShape(NodeFunc):
    def __init__(self, name):
        super().__init__(name)
        self.ret_name = name

    def __call__(self, ret_sigs, **kwargs):
        sig = ret_sigs[self.ret_name]
        shape = [ kwargs[s] for s in sig ]
        return True, shape

class ShapeList(NodeFunc):
    """
    Interpret the contents as a shape.
    """
    def __init__(self, arg_name, broadcast_mode):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.broadcast_mode = broadcast_mode

    def __call__(self, shape, *shapes):
        if not all(isinstance(v, int) for v in shape):
            return False, ArgValueError(self.arg_name, shape)
        if not all(v >= 0 for v in shape):
            return False, ArgValueError(self.arg_name, shape)
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
    def __init__(self, arg_name):
        super().__init__(arg_name)
        self.arg_name = arg_name

    def __call__(self, i):
        if i < 0:
            return False, ArgValueError(self.arg_name, i)
        else:
            return True, [i]

class ShapeTensorFunc(NodeFunc):
    """
    Interpret the tensor contents as a shape.
    Additionally, perform the checks defined by {pred_func}.
    {pred_func} accepts the integer list shape extracted from the tensor as
    well as any integer lists provided by *shapes.
    """
    def __init__(self, arg_name, pred_func):
        super().__init__(arg_name)
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


# TODO: differentiate each SchemaStatus result
class ShapeTensor2D(NodeFunc):
    """
    Validate a 2D shape tensor, and return its contents as a tuple of integer
    lists.
    """
    def __init__(self, arg_name, num_slices):
        super().__init__(arg_name)
        self.arg_name = arg_name
        self.num_slices = num_slices

    def __call__(self, op):
        ten = op._get_arg(self.arg_name) 
        if not ten.dtype.is_integer:
            return False, ArgValueError(self.arg_name, ten)
        elif ten.shape.rank != 2:
            return False, ArgValueError(self.arg_name, ten)  
        elif ten.shape[1] != self.num_slices:
            return False, ArgValueError(self.arg_name, ten)
        vals = tf.transpose(ten).numpy()
        tup = tuple(vals.tolist())
        return True, tup

class SliceShape(NodeFunc):
    """
    Get a slice from a tuple of shapes.
    """
    def __init__(self, name, tup_index):
        super().__init__(f'{name}.{tup_index}')
        self.index = tup_index

    def __call__(self, shape_tup):
        vals = shape_tup[self.index]
        if any(v < 0 for v in vals):
            return False, ArgValueError(self.arg_name, vals)
        return True, vals 

class DTypes(NodeFunc):
    """
    Aggregate the outputs of TensorDType nodes.  Always succeeds
    """
    def __init__(self):
        super().__init__()

    def __call__(self, **dtypes_map):
        return True, dtypes_map

"""
class DTypes(NodeFunc):
    def __init__(self, dtype_cons):
        super().__init__()
        self.dtype_cons = dtype_cons

    def __call__(self, index_ranks, layout, **kwargs):
        # Check that all tensor arguments have valid dtypes
        arg_dtypes = kwargs
        stat = self.dtype_cons.status(arg_dtypes, index_ranks, layout)
        valid = isinstance(stat, Success)
        return valid, stat
"""

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

class GetRanks(TupleElement):
    """
    Get the Ranks from RanksSigsShapes
    """
    def __init__(self):
        super().__init__(0)

class GetArgSigs(TupleElement):
    """
    Get the Sigs from RanksSigsShapes
    """
    def __init__(self):
        super().__init__(2)

class GetReturnSigs(TupleElement):
    """
    Get the Sigs from RanksSigsShapes
    """
    def __init__(self):
        super().__init__(2)

class GetShapes(TupleElement):
    """
    Get the (possibly broadcast-realized) Shapes from RanksSigsShapes
    """
    def __init__(self):
        super().__init__(3)

class Inventory(NodeFunc):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def get_hits(self, max_edit_dist):
        self.op.max_edit_dist = max_edit_dist
        hits = list(fgraph.gen_graph_values(
                        self.op.inv_live_nodes,
                        self.op.inv_output_nodes))
        return hits

    def __call__(self, dtypes, shapes, args):
        # prepare the inventory graph
        self.op.obs_dtypes.set_cached(dtypes)
        self.op.obs_shapes.set_cached(shapes)
        self.op.obs_args.set_cached(args)
        self.op.generation_mode = GenMode.Inference
        
        hits = self.get_hits(0)
        if len(hits) > 0:
            return True, hits

        # produce up to some number of suggestions 
        max_suggs = 5
        all_hits = []
        for dist in range(1, 3):
            hits = self.get_hits(dist)
            all_hits.extend(hits)
            if len(all_hits) >= max_suggs:
                break
        return False, all_hits


class RanksSigsShapes(NodeFunc):
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

    If the predicate succeeds, the value returned is:
    (index_ranks, arg_sigs, arg_shapes)
    where arg_shapes are broadcast-resolved shapes.  That is, if an original
    arg => shape mapping had an integer shape, the resolved shape will be an
    integer list of appropriate rank.

    If fails, value is NoMatchingRanks
    """
    def __init__(self, op, rank_candidates, rank_cons):
        super().__init__()
        self.op = op
        self.cands = rank_candidates
        self.rcons = rank_cons

    @staticmethod
    def broadcast_shapes(ranks, shapes, sigs):
        bcast_shapes = {}
        for arg, shape in shapes.items():
            if isinstance(shape, list):
                bcast_shapes[arg] = shape
            elif isinstance(shape, int):
                sig = sigs[arg]
                rank = sum(ranks[idx] for idx in sig)
                bcast_shapes[arg] = [shape] * rank
        return bcast_shapes

    @classmethod
    def error_key(cls, deltas):
        num_errors = len(list(d for d in deltas if d != 0))
        tot_error = sum(abs(d) for d in deltas)
        return num_errors, tot_error

    @classmethod
    def qualified(cls, deltas):
        num_errors, tot_error = cls.error_key(deltas)
        return num_errors < 3 and tot_error < 3

    def __call__(self, shapes, data_format, **kwargs):
        """
        shapes: arg => shape.  shape may be an integer list or an integer.
        If an integer, it represents any shape which is a broadcasted version
        of that integer.
        """
        fields = ['format', 'sigs', 'ranks', 'suggestions', 'highlight']
        Candidate = namedtuple('Candidate', fields[:3])
        Report = namedtuple('Report', fields)

        result = None
        error_tuples = []

        it = 0
        for ranks, arg_sigs, ret_sigs, cand_format in self.op.ranks_sigs_format_list:
            layout_delta = (data_format == cand_format)
            deltas = []
            for c in self.rcons:
                delta = c.rank_error(arg_sigs, shapes, ranks, **kwargs)
                deltas.append(delta)

            if all(d == 0 for d in deltas) and layout_delta == 0:
                if result is None:
                    result = (ranks, arg_sigs, ret_sigs)
                else:
                    raise SchemaError(
                        f'{type(self).__qualname__}: multiple valid rank '
                        f'combinations were found.  This means that schema'
                        f' \'{self.op.op_path}\' lacks proper rank '
                        f'constraints.  Current constraints are:\n'
                        f'{", ".join(c.name for c in self.rcons)}')

            cand = Candidate(cand_format, arg_sigs, ranks)
            if self.qualified(deltas + [layout_delta]):
                error_tuples.append((deltas, layout_delta, cand))

        if result is None:
            ordered = sorted(error_tuples, key=self.error_key)
            bestk = ordered[:self.op.max_report_candidates]
            reports = []
            for deltas, layout_delta, cand in bestk:
                report = Report(cand.format, cand.sigs, cand.ranks, [], {})
                for c, delta in zip(self.rcons, deltas):
                    sug = c.suggestion(delta)
                    if sug is not None:
                        report.suggestions.append(sug)
                    if delta != 0:
                        hl = c.highlight_map(arg_sigs, shapes, ranks)
                        for arg, pos in hl.items():
                            report.highlight_map[arg].extend(pos)

                if layout_delta == 1:
                    sug = f'Use layout {cand_format}'
                    report.suggestions.append(sug)
                    data_format_arg = self.op.data_formats.arg_name
                    # highlight the argument itself
                    report.highlight_map[data_format_arg].append(0)
                reports.append(report)

            # restore the shape map to its original form
            orig_shapes = {}
            for arg, shape in shapes.items():
                if isinstance(shape, int):
                    shape = [shape]
                orig_shapes[arg] = shape
            return False, NoMatchingRanks(orig_shapes, data_format, report)
        else:
            # resolve integer shapes
            ranks, cand_arg_sigs, cand_ret_sigs = result
            bcast_shapes = self.broadcast_shapes(ranks, shapes, cand_arg_sigs)
            return True, (ranks, cand_arg_sigs, cand_ret_sigs,
                    bcast_shapes)

def calc_sig_range(rank_map, idx, sig):
    ind = sig.index(idx)
    start = sum(rank_map[l] for l in sig[:ind])
    rank = rank_map[idx] 
    return [start, start + rank]

class IndexDimsUsage(NodeFunc):
    """
    Validates index usage.  Expands each signature using ranks, then applies
    the expanded signatures to the shapes map.  For any indices used in more
    than one argument, detects whether the used dimensions are the same in each
    instance.

    {shapes}: arg => shape
    {ranks}: idx => rank
    {sigs}: arg => sig

    The values of {shapes} can be either:
    1. integer list - specifies the shape directly
    2. integer - true shape is a broadcasting of the integer to any rank

    Returns index_dims: idx => dims

    """
    def __init__(self):
        super().__init__()

    def __call__(self, ranks, arg_sigs, shapes):
        # expand signatures

        # for each index and each component, produce a usage map 
        idx_usage = {} 
        for arg, sig in arg_sigs.items():
            shape = shapes[arg]
            it = base.shape_iter(shape)
            for idx in sig:
                r = ranks[idx]
                if idx not in idx_usage:
                    idx_usage[idx] = [defaultdict(list) for _ in range(r)]
                dims = base.shape_nextn(it, r)
                for c, dim in enumerate(dims):
                    idx_usage[idx][c][dim].append(arg)
                
        index_dims = {}
        for idx in list(idx_usage):
            comp = idx_usage[idx]
            if all(len(c) == 1 for c in comp):
                idx_usage.pop(idx)
                index_dims[idx] = [i for c in comp for i in c] 
        
        if len(idx_usage) != 0:
            for k, v in idx_usage.items():
                idx_usage[k] = [dict(dd) for dd in v]
            return False, IndexUsageError(idx_usage, ranks, arg_sigs, shapes)
        else:
            return True, index_dims

class SingleIndexDims(NodeFunc):
    """
    A simple node which extracts a single index dimension.  Should always
    return true, even if the index isn't provided.  Presence or absence of
    indices here may be a function of the layout.
    """
    def __init__(self, index_name):
        super().__init__(index_name)
        self.index_name = index_name

    def __call__(self, index_dims):
        return True, index_dims.get(self.index_name, None)

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

class ComputedDims(NodeFunc):
    """
    Apply a broadcasting function to compute dimensions
    """
    def __init__(self, index_name, func, num_index_args):
        super().__init__(index_name)
        self.index = index_name
        self.func = func
        self.nidx = num_index_args

    def __call__(self, *args):
        idx_args, extra_args = args[:self.nidx], args[self.nidx:]
        idx_args = [ np.array(a) for a in idx_args ]
        dims_ary = self.func(*idx_args, *extra_args)
        dims = dims_ary.astype(np.int32).tolist()
        return True, dims

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

"""
class Layout(NodeFunc):
    def __init__(self, formats, name):
        super().__init__(name)
        self.formats = formats

    def __call__(self, data_format):
        layout = self.formats.layout(data_format)
        if layout is None:
            return False, ArgValueError(self.formats.arg_name, data_format)
        else:
            return True, layout 
"""


class DataFormat(NodeFunc):
    def __init__(self, formats, arg_name):
        super().__init__(arg_name)
        self.formats = formats
        self.arg_name = arg_name

    def __call__(self, op):
        if not self.formats.configured:
            return True, self.formats.default()

        arg_val = op._get_arg(self.arg_name)
        valid = self.formats.valid_format(arg_val)
        if valid:
            return True, arg_val
        else:
            return False, ArgValueError(self.arg_name, arg_val) 

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
            return False, ArgTypeError(self.arg_name)
        elif arg_val not in range(self.lo, self.hi + 1):
            return False, ArgValueError(self.arg_name)
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

class Closure(NodeFunc):
    def __init__(self, name, obj):
        super().__init__(name)
        self.obj = obj

    def __call__(self):
        return self.obj

class Options(NodeFunc):
    def __init__(self, arg_name, options):
        super().__init__(arg_name)
        self.arg_name = arg_name
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
            return False, NonOptionError(arg_name, arg_val) 

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
        return (True, self.op)

