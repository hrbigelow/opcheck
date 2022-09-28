from collections import namedtuple

class OpCheckInternalError(BaseException):
    """A bug in OpCheck"""
    def __init__(self, ex):
        self.ex = ex

class SchemaError(BaseException):
    """Represents an error in the Schema definition"""
    def __init__(self, msg):
        self.msg = msg

class SchemaStatus(BaseException):
    """Represent violations of schema constraints"""
    def __init__(self):
        pass

    def message(self, op):
        raise NotImplementedError

class Success(SchemaStatus):
    def __init__(self):
        pass

    def message(self, op):
        return 'Success'

class NotApplicable(SchemaStatus):
    def __init__(self):
        pass

    def message(self, op):
        return f'Not Applicable'

class FrameworkError(SchemaStatus):
    def __init__(self, framework_error):
        self.ex = framework_error

    def message(self, op):
        return f'{self.ex}'

class NoMatchingRanks(SchemaStatus):
    """
    No matching index rank combinations could be found which explain the
    shape-related input arguments.
    """
    def __init__(self, shape_map, data_format, report):
        self.shape_map = shape_map
        self.data_format = data_format
        self.report = report

    def message(self, op):
        msg = op._rank_error_report(self.shape_map, self.data_format,
                self.report)
        return msg

class ArgTypeError(SchemaStatus):
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def message(self, op):
        msg = f'Argument \'{self.arg_name}\' received invalid type'
        return msg

class ArgValueError(SchemaStatus):
    def __init__(self, arg_name, arg_val):
        self.arg_name = arg_name
        self.arg_val = arg_val

    def message(self, op):
        msg = (f'Argument \'{self.arg_name}\' received invalid value '
                f'{self.arg_val}')
        return msg

class DTypeNotEqual(SchemaStatus):
    def __init__(self, src_name, src_dtype, trg_name, trg_dtype):
        self.src_name = src_name
        self.src_dtype = src_dtype
        self.trg_name = trg_name
        self.trg_dtype = trg_dtype

    def message(self, op):
        msg = (f'Tensors \'{self.trg_name}\' and \'{self.src_name}\' must have '
                f'equal dtypes.\n'
                f'Got {self.trg_name}.dtype = {self.trg_dtype.name} and '
                f'{self.src_name}.dtype = {self.src_dtype.name}')
        return msg

class DTypeNotValid(SchemaStatus):
    def __init__(self, ten_name, ten_dtype, valid_dtypes):
        self.ten_name = ten_name
        self.ten_dtype = ten_dtype
        self.valid_dtypes = valid_dtypes

    def message(self, op):
        msg = (f'Tensor \'{self.ten_name}\' had dtype={self.ten_dtype.name}. '
                f'The allowed dtypes are '
                f'{", ".join(d.name for d in self.valid_dtypes)}')
        return msg

class IndexUsageError(SchemaStatus):
    def __init__(self, idx_usage, ranks, sigs, shapes):
        self.idx_usage = idx_usage
        self.ranks = ranks
        self.sigs = sigs
        self.shapes = shapes

    def message(self, op):
        return op._index_usage_error(self.idx_usage, self.ranks,
                self.sigs, self.shapes)

class ComponentConstraintError(SchemaStatus):
    """
    Returned by an index dims constraint function.  The contents are then
    incorporated into an IndexConstraintError
    """
    def __init__(self, text, error_mask):
        self.text = text
        self.mask = error_mask

class IndexConstraintError(SchemaStatus):
    def __init__(self, index_highlight, text, ranks, sigs, shapes):
        self.index_highlight = index_highlight
        self.text = text
        self.ranks = ranks
        self.sigs = sigs
        self.shapes = shapes

    def message(self, op):
        return op._index_constraint_error(self.text, self.index_highlight,
                self.ranks, self.sigs, self.shapes)

class NonOptionError(SchemaStatus):
    def __init__(self, arg_name, arg_val):
        self.arg_name = arg_name
        self.arg_val = arg_val

    def message(self, op):
        msg = (f'Argument \'{self.arg_name}\' had a non-option value of '
                f'{self.arg_val}')
        return msg

class CustomError(SchemaStatus):
    def __init__(self, message):
        self.msg = message

    def message(self, op):
        return self.msg

class ReturnShapeError(SchemaStatus):
    """The output at {out_idx} does not match the shape implied by its
    signature"""
    def __init__(self, out_idx):
        self.idx = out_idx

    def message(self, op):
        msg = f'The output at {self.idx} does not match the shape predicted '
        msg += f'by its signature'
        return msg

class OutputNumberMismatch(SchemaStatus):
    """The number of outputs returned differed from expected"""
    def __init__(self, exp_num_outputs, act_num_outputs):
        self.expected_num_outputs = exp_num_outputs
        self.actual_num_outputs = act_num_outputs

    def message(self, op):
        msg = f'Expected {self.expected_num_outputs} but got '
        msg += f'{self.actual_num_outputs}'
        return msg

class ReturnTypeError(SchemaStatus):
    """The type of the return object was not as expected"""
    def __init__(self, index, expected_type, actual_type):
        self.index = index
        self.expected_type = expected_type
        self.actual_type = actual_type

    def message(self, op):
        msg = f'Return at position {self.index} was expected to be of type '
        msg += f'{self.expected_type}, but was a {self.actual_type}'
        return msg

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

