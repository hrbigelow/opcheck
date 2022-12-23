import tensorflow as tf
import numpy as np
import enum
import re
from collections import namedtuple, OrderedDict
from .error import SchemaError
from . import fgraph
from . import oparg

LAYOUT = ':layout'
SINGLE_FORMAT = ':single_format'
INDEX_RANKS = ':index_ranks'

ALL_DTYPES = (
        'int8', 'int16', 'int32', 'int64',
        'uint8', 'uint16', 'uint32', 'uint64',
        'float16', 'float32', 'float64',
        'qint8', 'qint16', 'qint32',
        'bfloat16',
        'bool',
        'complex64', 'complex128',
        )

class CompDimsMode(enum.Enum):
    Dims = 0
    OneLetterCode = 1
    SnakeCaseDesc = 2
    StringDims = 3

def snake_case(phrase):
    return phrase.replace(' ', '_')

def dims_string(dims):
    if isinstance(dims, tuple):
        return repr(list(dims))
    else:
        return repr(dims)

def list_phrase_or(items):
    # generate 'A, B, or C'
    if not items:
        return None
    elif len(items) < 3:
        return ' or '.join(str(i) for i in items)
    else:
        return ', '.join(str(i) for i in items[:-1]) + ' or ' + str(items[-1])

def non_negative(shape):
    if isinstance(shape, int):
        return shape >= 0
    else:
        return all(s >= 0 for s in shape)


def ungroup_dims(gr_dims_map):
    # gr_dims_map is e.g. { 'bc': ([1,2], [2,3]), 'e': [5,6] }
    dims_map = {}
    for sig, dims in gr_dims_map.items():
        if len(sig) == 1:
            dims_map[sig] = dims
        else:
            dims_map.update(dict(zip(sig, dims)))
    return dims_map

def broadcastable_to(dims_list, rank):
    # check that each dims in dims_list is the same rank or rank agnostic
    kinds = (tuple, list)
    f = filter(lambda d: isinstance(d, kinds) and len(d) != 1, dims_list)
    return all(len(d) == rank for d in f)

def bcast_dim(dims, comp):
    # get the component comp of dims
    if isinstance(dims, int):
        return dims
    else:
        if len(dims) == 1:
            return dims[0]
        else:
            return dims[comp]

def range_under_size(idx_ranges, max_prod, rng):
    """
    generate a tuple of dims between idx_ranges, with prod() <= total
    each element of idx_ranges represents a component of an index.
    """
    if len(idx_ranges) == 0:
        return []

    if any(lo > hi for lo, hi in idx_ranges):
        raise SchemaError(f'range_under_size: invalid ranges: {idx_ranges}')

    lows = [lo for lo, _ in idx_ranges]
    ncomp = len(idx_ranges)
    runs = [np.prod(lows[i+1:], dtype=int) for i in range(ncomp)]
    if lows[0] * runs[0] > max_prod:
        raise SchemaError(
            f'range_under_size: idx_ranges {idx_ranges} '
            f'cannot fit within maximum product {max_prod}')

    cumul = 1
    dims = []
    for c, (lo, hi) in enumerate(idx_ranges):
        used = cumul * runs[c]
        if used == 0:
            ub = 1e10
        else:
            ub = max_prod // used
        d = rng.randint(lo, min(hi, ub))
        cumul *= d
        dims.append(d)
    return dims

class ShapeKind(enum.Enum):
    """
    For describing the kind of input that defines a shape
    """
    DataTensor = 0
    List = 1
    Int = 2
    Tensor = 3
    Tensor2D = 4

class ShapeEdit(object):
    def __init__(self, op, index_ranks, arg_sigs, layout):
        self.op = op
        self.arg_sigs = arg_sigs  # arg => sig (the index signature for arg
        self.index_ranks = index_ranks  # idx => rank
        self.arg_delta = {}
        self.usage_map = {}       # idx => (dims => [arg1, arg2, ...]) 
        self.layout = layout
        self.index_pred_error = None
        self.formulas = None
        self.comp_dims = None

    def __repr__(self):
        r  = f'layout: {self.layout}\n'
        r += f'arg_sigs: {self.arg_sigs}\n'
        r += f'arg_delta: {self.arg_delta}\n'
        r += f'usage_map: {self.usage_map}\n'
        r += f'pred: {self.index_pred_error}\n'
        return r

    def add_indels(self, arg_delta):
        self.arg_delta = arg_delta

    def add_idx_usage(self, usage_map):
        self.usage_map = usage_map

    def add_comp_dims(self, comp_dims):
        self.comp_dims = comp_dims

    def add_constraint_error(self, pred, formulas):
        # predicate {pred}, which accepts {formulas} as arguments has been
        # violated with the imputed {index_dims}.  This function is only called
        # if there are no index usage errors
        self.index_pred_error = pred
        self.formulas = formulas

    def indel_cost(self):
        # this assumes that each indel incurs an additional downstream cost of
        # 1.  this is a simple heuristic to make up for the fact that there is
        # no check for index usage error or index constraint error 
        return sum(abs(d) + 1 for d in self.arg_delta.values())

    def idx_usage_cost(self):
        # cost of 1 for every additional usage of an index (regardless of 
        # multiplicity of that usage)
        ucost = sum(len(u) - 1 for u in self.usage_map.values())
        return ucost

    def pred_cost(self):
        return 0 if self.index_pred_error is None else 1
    
    def cost(self):
        return self.indel_cost() + self.idx_usage_cost() + self.pred_cost()

    def code(self):
        code = 0
        if self.indel_cost() != 0:
            for arg, delta in self.arg_delta.items():
                if delta > 0:
                    code |= FixKind.InsertDim.value
                else:
                    code |= FixKind.DeleteDim.value
        if self.idx_usage_cost() != 0:
            code |= FixKind.IndexUsage.value
        if self.pred_cost() != 0:
            code |= FixKind.IndexPred.value
        return code

    def arg_templates(self):
        # create arg => template map
        arg_templ = {}
        for arg, sig in self.arg_sigs.items():
            templ = tuple(idx for idx in sig for _ in
                    range(self.index_ranks[idx]))
            arg_templ[arg] = templ
        return arg_templ

    def arg_index_slice(self, arg, idx):
        # return the slice that idx takes up in the arg shape 
        sig = self.arg_sigs[arg]
        off = 0
        for tmp_idx in sig:
            rank = self.index_ranks[tmp_idx]
            if tmp_idx == idx:
                return off, off + rank 
            off += rank

    def get_input_dims(self, use_scalars=False):
        # return the imputed index dims, or raise an error if ambiguous
        # if `use_scalars`, convert any length 1 dims to scalar if the
        # index has a constant rank of 1
        if any(len(usage) > 1 for usage in self.usage_map.values()):
            raise RuntimeError(
                f'{type(self).__qualname__}: ambiguous index usage')
        index_dims = {}
        for idx, usage in self.usage_map.items():
            assert len(usage) == 1, 'Index Usage should be length 1'
            dims = next(iter(usage))
            if isinstance(dims, int):
                index_dims[idx] = dims
                continue

            if use_scalars and self.op.index[idx].fixed_rank():
                if len(dims) != 1:
                    raise RuntimeError(
                        f'Index {idx} is fixed rank = 1 but found usage of '
                        f'non-rank-1')
                index_dims[idx] = dims[0]
            else:
                index_dims[idx] = dims
        return index_dims

    def maybe_get_index_dim(self, idx):
        if idx in self.usage_map:
            usage = self.usage_map[idx]
            if len(usage) == 1:
                return next(iter(usage))
            else:
                return [None] * self.index_ranks[idx]
        elif self.comp_dims is not None and idx in self.comp_dims:
            return self.comp_dims[idx]
        else:
            return [None] * self.index_ranks[idx]

    def get_arg_shape(self, arg):
        """
        Get the imputed shape of arg
        """
        if self.cost() != 0:
            raise RuntimeError(
                f'Cannot call {type(self).__qualname__} with non-zero cost')
        sig = self.arg_sigs[arg]
        index_dims = self.get_input_dims()
        if self.comp_dims is not None:
            index_dims.update(self.comp_dims)
        shape = [ dim for idx in sig for dim in index_dims[idx] ]
        return shape

    def highlighted(self, arg, idx):
        # return whether the usage of idx in arg should be highlighted
        if idx in self.usage_map and len(self.usage_map[idx]) > 1:
            return True
        elif (self.index_pred_error is not None and idx in
                self.index_pred_error.indices):
            return True
        else:
            return False

class ValueEdit(object):
    def __init__(self, name, obs_val, imputed_val):
        self.name = name
        self.obs_val = obs_val
        self.imp_val = imputed_val

    def __repr__(self):
        f = f'{type(self).__qualname__}[{self.name}]'
        f += f'({self.obs_val}, {self.imp_val})'
        return f

    def cost(self):
        return 0 if self.obs_val == self.imp_val else 1

class DataFormatEdit(object):
    def __init__(self, df_name, observed_fmt, used_fmt, imputed_fmt):
        self.arg_name = df_name
        self.observed = observed_fmt
        self.used = used_fmt
        self.imputed = imputed_fmt

    def __repr__(self):
        s = f'{type(self).__qualname__}'
        s += f'(obs: {self.observed}, used: {self.used}, imp: {self.imputed})'
        return s

    def cost(self):
        return 0 if self.used == self.imputed else 2

    def code(self):
        if self.cost() == 0:
            return 0
        else:
            return FixKind.Layout.value
        
class DTypesEdit(object):
    def __init__(self, rule_kind, info):
        # one of 'indiv', 'equate', 'combo', or None
        self.kind = rule_kind
        self.info = info

    def __hash__(self):
        return hash((self.kind, self.info))

    def __repr__(self):
        return f'{type(self).__qualname__}[{self.kind},{self.info}]'

    def code(self):
        if self.kind == 'indiv':
            return FixKind.DTypeIndiv.value
        elif self.kind == 'equate':
            return FixKind.DTypeEquate.value
        elif self.kind == 'combo':
            return FixKind.ComboExcluded.value
        else:
            return 0

    def cost(self):
        return 0 if self.kind is None else 1

def parse_dtype_expr(type_expr):
    # return the matching dtypes 
    exprs = {
            'int': [8, 16, 32, 64],
            'uint': [8, 16, 32, 64],
            'float': [16, 32, 64],
            'qint': [8, 16, 32],
            'bfloat': [16],
            'bool': [''],
            'complex': [64, 128]
            }

    types = [ ', '.join(f'{k}{v}' for v in exprs[k]) for k in exprs ]
    type_str = '\n'.join(t for t in types)
    err_msg = SchemaError(
        f'Received invalid dtype expression \'{type_expr}\'.\n'
        f'dtype expression must match the pattern:\n'
        f'([a-z]+)(8|16|32|64|128)?([\+\-])?\n'
        f'The first capture is the data type and must be one of: '
        f'int, uint, float, qint, bfloat, bool, complex\n'
        f'The second capture is the size.  It is optional. '
        f'The third is an optional \'+\' or \'-\''
        f'The list of valid constructed types are:\n'
        f'{type_str}\n'
        )

    # expect format to be {pfx}{q}[+-]*
    ma = re.match('([a-z]+)(8|16|32|64|128)?([\+\-])?', type_expr)
    if ma is None:
        raise err_msg
    pfx, q, rng = ma.groups()
    if q is None:
        ids = [ f'{pfx}{sz}' for sz in exprs[pfx] ]
    else:
        if rng is None:
            ids = [ type_expr ]
        elif rng == '+':
            ids = [ f'{pfx}{sz}' for sz in exprs[pfx] if sz >= int(q) ]
        else:
            ids = [ f'{pfx}{sz}' for sz in exprs[pfx] if sz <= int(q) ]
    try:
        dtypes = [ tf.dtypes.as_dtype(i) for i in ids ]
    except TypeError:
        raise err_msg
    return ids

class ComboRule(object):
    def __init__(self):
        # tensor => [excluded_dtype, ...], None means all are excluded
        self.dtypes = None 

        # idx => [excluded_rank, ...], None means all are excluded 
        self.ranks = None 

        # set of excluded layouts, None means all are excluded
        self.layouts = None

    def __repr__(self):
        f = f'{type(self).__qualname__}({self.dtypes}, {self.ranks}, '
        f += f'{self.layouts}'
        return f

    def exclude_dtypes(self, arg, *dtype_exprs):
        self.dtypes = {}
        for dtype_expr in dtype_exprs:
            dtypes = parse_dtype_expr(dtype_expr)
            excluded = self.dtypes.setdefault(arg, [])
            excluded.extend(dtypes)

    def exclude_rank(self, idx, *ranks):
        if self.ranks is None:
            self.ranks = {}
        exc_ranks = self.ranks.setdefault(idx, [])
        exc_ranks.extend(ranks)

    def exclude_layout(self, *layouts):
        self.layouts = set()
        for layout in layouts:
            self.layouts.add(layout)

    def match(self, obs_dtypes, index_ranks, layout):
        # return True if this ComboRule matches the observations
        if self.dtypes is not None:
            for arg, dtypes in self.dtypes.items():
                obs_dtype = obs_dtypes[arg]
                if obs_dtype not in dtypes:
                    return False
        if self.ranks is not None:
            for idx, ranks in self.ranks.items():
                obs_rank = index_ranks[idx]
                if obs_rank not in ranks:
                    return False
        if self.layouts is not None:
            if layout not in self.layouts:
                return False 
        return True

class DTypeRules(object):
    """
    Represents combinations of dtypes, ranks and layouts not implemented by the
    framework.
    """
    def __init__(self):
        self.initialized = False
        self.combos = []

        # arg => [valid_dtype, ...].  initialized from API call valid_dtypes 
        self.indiv_rules = {}

        # target_tensor => source_tensor
        self.equate_rules = {}

    def init_fields(self, data_tensors, indices):
        self.data_tensors = data_tensors
        self.indices = indices
        self.initialized = True

    def add_indiv_rule(self, tensor, valid_types):
        self.indiv_rules[tensor] = valid_types

    def add_equate_rule(self, target_tensor, source_tensor):
        self.equate_rules[target_tensor] = source_tensor

    def add_combo(self, *field_val_pairs):
        """
        {field_val_pairs} is an even-length list of field, val, field, val, ...
        field is one of: 
        - data tensors registered in init_fields
        - one-letter index names registered in init_fields
        - the constant LAYOUT, if has_layout

        val may be a scalar or a tuple of:
        - dtype string, such as 'int32' for data tensor fields
        - integer specifying a rank of an index field
        - the LAYOUT field,  an integer in [0, num_layouts), as defined
          by the call to arg_layout.
        """
        nitem = len(field_val_pairs)
        if nitem % 2 != 0:
            raise RuntimeError(
                f'{type(self).__qualname__}: field_val_pairs must be '
                f'even-length list.  Got length {len(field_val_pairs)} items')
        combo_rule = ComboRule()
        for i in range(0, nitem, 2):
            field = field_val_pairs[i]
            values = field_val_pairs[i+1]
            if not isinstance(values, (tuple, list)):
                values = (values,)
            if field in self.data_tensors:
                combo_rule.exclude_dtypes(field, *values)
            elif field in self.indices:
                combo_rule.exclude_rank(field, *values)
            elif field == LAYOUT:
                combo_rule.exclude_layout(*values)
            else:
                raise RuntimeError(
                    f'{type(self).__qualname__}: got field \'{field}\' which '
                    f'is not a known data tensor, index or the constant '
                    f'\'{LAYOUT}\'\n'
                    f'Known data tensors are: {self.data_tensors}'
                    f'Known indices are: {self.indices}')
        self.combos.append(combo_rule)

    def edit(self, obs_dtypes, index_ranks, layout):
        # check each indiv rule
        for arg, valid_dtypes in self.indiv_rules.items():
            obs_dtype = obs_dtypes[arg]
            if obs_dtype not in valid_dtypes:
                return DTypesEdit('indiv', arg)

        # check each equate rule
        for arg, source_arg in self.equate_rules.items():
            obs_dtype = obs_dtypes[arg]
            obs_src_dtype = obs_dtypes[source_arg]
            if obs_dtype != obs_src_dtype:
                return DTypesEdit('equate', arg)

        matched = self.matched_rule(obs_dtypes, index_ranks, layout)
        if matched is None:
            return DTypesEdit(None, None)
        else:
            return DTypesEdit('combo', matched)

    def matched_rule(self, obs_dtypes, index_ranks, layout):
        """
        Returns a matching exclusion rule for the set of observed dtypes,
        index_ranks and layout.  If no rule matches, return None
        """
        for combo in self.combos:
            if combo.match(obs_dtypes, index_ranks, layout):
                return combo
        return None

class DataFormats(object):
    """
    A 'data_format' is a string like 'NCW', 'NHWC', etc.  A 'layout' is a
    notion of rank-agnostic data_format.  For 1D, 2D, 3D convolutions, the
    data_formats 'NCW', 'NCHW', 'NCDHW' all correspond to a notion of 'channel
    first' (layout 0), while the data_formats 'NWC', 'NHWC', and 'NDHWC' are
    'channel last', and given layout 1.

    {formats} is a map of data_format => (layout, rank).  The layout is an
    integer index specifying the layout.  The rank specifies RANK(rank_index)
    or None if it is agnostic to the index.
    """
    def __init__(self, arg_name, formats, rank_index):
        self.arg_name = arg_name
        if formats is None:
            self.formats = { SINGLE_FORMAT: (0, None) }
            self.rank_index = None
        else:
            key_func = lambda t: '___' if t is None else t[0]
            sorted_keys = sorted(formats.keys(), key=key_func)
            self.formats = OrderedDict({k: formats[k] for k in sorted_keys})
            self.rank_index = rank_index

    def single(self):
        # return a pseudo-format for ops that have no switch for data_format
        return SINGLE_FORMAT

    def default_format(self, ranks):
        """
        Return the format consistent with the ranks in the case when the user
        does not submit an argument value for data_format
        """
        if None not in self.formats:
            raise RuntimeError(
                f'Cannot call default_format if None is not permitted for '
                f'{self.arg_name}')
        def_layout = self.formats[None][0]
        return self.data_format(def_layout, ranks)

    def observed_format(self, obs_args):
        if self.arg_name is None:
            return SINGLE_FORMAT
        else:
            return obs_args.get(self.arg_name, None)

    def num_layouts(self):
        return len({ lr[0] for lr in self.formats.values() })

    def all_formats(self):
        return list(self.formats.keys())

    def data_format(self, layout, ranks):
        """
        Return the data_format corresponding to the layout and rank
        combination.
        """
        items = self.formats.items()
        rank = None if self.rank_index is None else ranks[self.rank_index]
        if rank is None:
            return next((df for df, (l, _) in items if l == layout), None)
        else:
            return next((df for df, (l, r) in items if l == layout and (r is None
                or r == rank)), None)

    def layout_formats(self, layout):
        """
        Return all formats for a particular layout, in rank order
        """
        items = self.formats.items()
        fmts = [ (df, r) for df, (l, r) in items if l == layout and df is not None]
        fmts = sorted(fmts, key=lambda tup: tup[1])
        return [ df for df, _ in fmts ]

    def layout(self, data_format):
        """
        Return the layout corresponding with this data format
        """
        if data_format not in self.formats:
            raise RuntimeError(
                f'{type(self).__qualname__}: received unknown data_format '
                f'\'{data_format}\'')
        return self.formats[data_format][0]

    def rank(self, data_format):
        """
        Return the rank corresponding with this data format
        """
        if data_format not in self.formats:
            raise RuntimeError(
                f'{type(self).__qualname__}: received unknown data_format '
                f'\'{data_format}\'')
        return self.formats[data_format][1]

class IndexPredicate(object):
    def __init__(self, name, cwise, pfunc, pfunc_t, indices):
        self.name = name
        self.cwise = cwise
        self.pfunc = pfunc
        self.pfunc_t = pfunc_t
        self.indices = indices

    def get_formula(self, op, snake_case):
        inputs = tuple(op.index[idx].display_name(snake_case) for idx in
                self.indices)
        return self.pfunc_t(*inputs)

    def __call__(self, *dims):
        """
        Evaluate the predicate either component-wise or all at once
        """
        if self.cwise:
            ranks = { len(d) for d in dims if isinstance(d, (list, tuple)) }
            if len(ranks) > 1:
                raise SchemaError(
                    f'IndexPredicate is component-wise but input indices '
                    f'{self.indices} have different ranks')
            elif len(ranks) == 0:
                try:
                    return self.pfunc(*dims)
                except BaseException as ex:
                    raise SchemaError(
                        f'IndexPredicate {self.name} while running predicate '
                        f'function func({dims}) got exception: {ex}')
            else:
                rank = ranks.pop()
                for c in range(rank):
                    ins = tuple(d if isinstance(d, int) else d[c] for d in dims)
                    if not self.pfunc(*ins):
                        return False
                return True
        else:
            try:
                return self.pfunc(*dims)
            except BaseException as ex:
                raise SchemaError(f'IndexPredicate error in non-cw mode: {ex}')
    
class SumRangeConstraint(object):
    """
    Expresses the constraint RANK(sig) in [lo, hi].  When called, it provides a
    residual range based on values provided for some subset of indexes in the
    signature.
    """
    def __init__(self, sig, lo, hi):
        self.sig = sig
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return f'{type(self).__name__}: RANK({self.sig}) in [{self.lo}, {self.hi}]'

    def __call__(self, **index_ranks):
        prev = [ r for idx, r in index_ranks.items() if idx in self.sig ]
        part = sum(prev)
        final = (len(prev) + 1 == len(self.sig))
        lo = max(0, self.lo - part) if final else 0
        hi = max(0, self.hi - part)
        return lo, hi

class ShapeFuncConstraint(object): 
    """
    Expresses the constraint RANK(sig) = func(obs_shapes[shape_arg])
    """
    def __init__(self, sig, func, shape_arg):
        self.sig = sig
        self.func = func
        self.shape_arg = shape_arg

    def __repr__(self):
        r =  f'{type(self).__qualname__}: RANK({self.sig}) = '
        r += f'{self.func.__name__}(...)'
        return r

    def __call__(self, obs_shapes, **index_ranks):
        shape = obs_shapes[self.shape_arg]
        rank = self.func(shape)
        if rank is None:
            return 0, -1

        residual = sum(index_ranks.get(idx, 0) for idx in self.sig)
        target = rank - residual
        return target, target

class SigRankValueConstraint(object):
    """
    Defines the constraint RANK(sig) = arg
    """
    def __init__(self, arg, sig):
        self.arg = arg
        self.sig = sig

    def __repr__(self):
        r =  f'{type(self).__qualname__}: RANK({self.sig}) = {self.arg}'
        return r

    def __call__(self, obs_args, **index_ranks):
        rank = obs_args[self.arg]
        residual = sum(index_ranks.get(idx, 0) for idx in self.sig)
        target = rank - residual
        return target, target

class ReportKind(enum.Enum):
    CaratTable = 0 
    ArrowTable = 1
    DType = 2

class FixKind(enum.Enum):
    DTypeEquate = 1    # input tensors have differing dtypes but should match
    DTypeIndiv = 2     # one input tensor has a disallowed dtype
    ComboExcluded = 4  # this combo of dtypes, index ranks and/or layout is excluded 
    InsertDim = 8      # fix by inserting dimension(s) to a particular tensor 
    DeleteDim = 16     # fix by deleting dimension(s) from a particular tensor
    IndexUsage = 32    # an index appearing multiple places has differing dimension
    IndexPred = 64     # an index predicate is violated
    Layout = 128       # fix by trying an alternate layout

    @classmethod
    def codestring(cls, code):
        codes = []
        for opt in cls:
            if opt.value & code:
                codes.append(opt.name)
        return '|'.join(codes)

class Fix(object):
    def __init__(self, df_edit, dtype_edit, shape_edit, **kwargs):
        self.df = df_edit
        self.dtype = dtype_edit
        self.shape = shape_edit
        self.kwargs = kwargs

    def __repr__(self):
        f = f'df: {self.df}\ndtype: {self.dtype}\nshape: {self.shape}'
        f += f'kwargs: {self.kwargs}\n'
        f += f'cost(df,dt,indel,usg,prd): '
        f += f'{self.df.cost()}, '
        f += f'{self.dtype.cost()}, '
        f += f'{self.shape.indel_cost()}, '
        f += f'{self.shape.idx_usage_cost()}, '
        f += f'{self.shape.pred_cost()}'
        return f

    def code(self):
        code = self.dtype.code() + self.shape.code() + self.df.code()
        return code

    def summary(self):
        """
        Display a logical, one-line summary of this fix, for indexing purposes
        """
        shape = self.shape
        ranks = [shape.index_ranks[idx] for idx in shape.op.index.keys()]
        rank_str = ','.join(str(r) for r in ranks)
        r =  f'L:{self.shape.layout}, R:{rank_str}'
        if self.df.cost() != 0:
            r += f' DF:{self.df.observed}=>{self.df.imputed}'
        if self.dtype.cost() != 0:
            r += f' DT:{self.dtype.kind}'
        if self.shape.cost() != 0:
            if self.shape.indel_cost() != 0:
                items = []
                for arg, delta in self.shape.arg_delta.items():
                    if delta > 0:
                        item = f'{arg}(ins {delta})'
                    else:
                        item = f'{arg}(del {abs(delta)})'
                    items.append(item)
                indel_msg = ', '.join(items)
                r += f' Indels({indel_msg})'
            if self.shape.idx_usage_cost() != 0:
                items = []
                for idx, usage in self.shape.usage_map.items():
                    if len(usage) > 1:
                        all_args = [a for u in usage.values() for a in u]
                        arg_msg = ','.join(all_args)
                        item = f'{idx}:{arg_msg}'
                        items.append(item)
                usage_msg = ', '.join(items)
                r += f' Usage({usage_msg})'
            if self.shape.pred_cost() != 0:
                pred_name = self.shape.index_pred_error.name
                r += f' PredError({pred_name})'
        return r

    def kind(self):
        dtype_cost = self.dtype.cost()
        rank_cost = self.shape.indel_cost()

        if dtype_cost != 0:
            return ReportKind.DType
        elif rank_cost != 0:
            return ReportKind.ArrowTable
        else:
            return ReportKind.CaratTable

    def cost(self):
        return self.df.cost() + self.dtype.cost() + self.shape.cost()

Formula = namedtuple('Formula', ['dims', 'code_path', 'desc_path', 'dims_path'])

class RenderCompDims(object):
    def __init__(self, op):
        self.op = op

    def set_inputs(self, dims_inputs):
        for name, val in dims_inputs.items():
            if isinstance(val, oparg.OpArg):
                val = val.value()
            self.op.dims_graph_input[name] = val

    def init_dims_graph(self, idx_info):
        # initialize input nodes
        for node in self.op._gen_dims_nodes():
            sig = node.sub_name
            if all(idx in idx_info for idx in sig):
                if len(sig) == 1:
                    val = idx_info[sig]
                else:
                    val = tuple(idx_info[idx] for idx in sig)
                node.set_cached(val)
            else:
                pass

    def _run_comp_graph(self, comp_mode, inputs):
        """
        Produce a map of idx => (info, info_formula) for each computed index
        `idx`.  info and info_formula are the lhs and rhs of an equation in
        terms of of the formats specified by comp_mode, CompDimsMode enum.
        """
        self.op.comp_dims_mode = comp_mode 
        self.init_dims_graph(inputs)
        comp_nodes = self.op._comp_dims_nodes()
        input_nodes = self.op._dims_input_nodes()
        live_nodes = comp_nodes + input_nodes
        gen = fgraph.gen_graph_map(live_nodes, comp_nodes, full_name=False)
        result = list(gen)
        assert len(result) == 1, 'Internal Error with comp graph'
        return result[0]

    def get_olc(self):
        """
        Get a map of idx => (olc, olc_formula)
        olc is the one-letter-code of the computed index
        olc_formula is a formula in terms of one-letter-codes
        """
        return self._run_comp_graph(CompDimsMode.OneLetterCode, {})

    def get_snake(self):
        """
        Get a map of computed indexes of idx => (snake_index, snake_formula)
        snake_index is the snake-cased description of an index
        snake_formula is a formula for the computed index in terms of
        snake_indexes
        """
        return self._run_comp_graph(CompDimsMode.SnakeCaseDesc, {})

    def get_sdims(self, index_dims):
        """
        Get a map of computed indexes of idx => (dims, sdims_formula)
        dims is the integer or integer list computed dimensions of idx
        sdims_formula is a formula in terms of string representations of 
        dimensions
        """
        left_dims = { idx: (dims, None) for idx, dims in index_dims.items() }
        return self._run_comp_graph(CompDimsMode.StringDims, left_dims)

    def get_dims(self, index_dims):
        """
        Get a map of computed indexes of idx => dims
        """
        return self._run_comp_graph(CompDimsMode.Dims, index_dims)

    def formula_map(self, input_dims):
        codes = self.get_olc()
        descs = self.get_snake()
        sdims = self.get_sdims(input_dims)
        comp_nodes = self.op._comp_dims_nodes()
        comp_names = [ n.sub_name for n in comp_nodes ]
        formulas = {}
        for idx in comp_names:
            code_path = '{} = {}'.format(*codes[idx])
            desc_path = '{} = {}'.format(*descs[idx])
            dims_path = '{} = {}'.format(*sdims[idx])
            dims = sdims[idx][0] 
            formulas[idx] = Formula(dims, code_path, desc_path, dims_path)
        return formulas

# convert rows of arbitrary objects to tabular row strings
def tabulate(rows, sep, left_align=True):
    """
    {rows} is a list of rows, where each row is a list of arbitrary items

    Produces a tuple.  The first item is a string-representation of {rows},
    such that each item is column-aligned, using {sep} as a field separator.
    
    rows may have different numbers of items.  the longest row defines the
    number of columns, and any shorter rows are augmented with empty-string
    items.

    The second item is a list of (beg, end) column position tuples
    corresponding to each column.
    """
    def get(items, i):
        try:
            return items[i]
        except IndexError:
            return ''
    
    ncols = max(len(row) for row in rows)
    if isinstance(left_align, bool):
        left_align = [left_align] * ncols

    w = [max(len(str(get(row, c))) for row in rows) for c in range(ncols)]
    t = []
    for row in rows:
        fields = []
        for c in range(len(row)):
            align = '<' if left_align[c] else '>'
            field = f'{str(row[c]):{align}{w[c]}s}'
            fields.append(field)
        t.append(sep.join(fields))

    begs = [sum(w[:s]) + len(sep) * s for s in range(ncols)]
    ends = [sum(w[:s+1]) + len(sep) * s for s in range(ncols)]
    return t, list(zip(begs, ends))

