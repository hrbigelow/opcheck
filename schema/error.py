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

    {shape_args} are the list of shape-related argument names
    {ranks_deltas} is a list of tuples with two members.  The first is the
 

    A rank-delta is a list of integers of the same length as {shape_args}
    """
    def __init__(self, sig_shape):
        self.sigs = { k: v[0] for k, v in sig_shape.items() }
        self.shapes = { k: v[1] for k, v in sig_shape.items() }

    @staticmethod
    def error_key(tup):
        _, _, tot, sum_abs = tup
        return tot, sum_abs

    def message(self, op):
        sig_inst = op._signature_instantiations()
        arg_list = self.shapes.keys()
        diffs = []
        for si in sig_inst:
            diff = [ len(si[k]) - len(v) for k, v in self.shapes.items() ]
            tot = len([e for e in diff if e != 0])
            sum_abs = sum(abs(e) for e in diff)
            diffs.append((si, diff, tot, sum_abs))
        diffs = sorted(diffs, key=self.error_key)
        return str(diffs[:3])

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

class TensorDTypeError(SchemaStatus):
    def __init__(self, ten_name):
        self.ten_name = ten_name

    def message(self, op):
        msg = f'Tensor \'{self.ten_nane}\' has invalid dtype'
        return msg

class IndexUsageError(SchemaStatus):
    def __init__(self, idx):
        self.idx = idx

    def message(self, op):
        return f'Index {self.idx} inconsistent'

class NegativeDimsError(SchemaStatus):
    def __init__(self, idx, dims):
        self.idx = idx
        self.dims = list(dims)

    def message(self, op):
        msg = f'Dimensions of index \'{self.idx}\' contain negative values: '
        msg += f'{self.dims}'
        return msg

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
    w = [max(len(str(get(row, c))) for row in rows) for c in range(ncols)]
    if left_align:
        t = [sep.join(f'{str(row[c]):<{w[c]}s}' for c in range(len(row)))
                for row in rows]
    else:
        t = [sep.join(f'{str(row[c]):>{w[c]}s}' for c in range(len(row)))
                for row in rows]

    begs = [sum(w[:s]) + len(sep) * s for s in range(ncols)]
    ends = [sum(w[:s+1]) + len(sep) * s for s in range(ncols)]
    return t, list(zip(begs, ends))

"""
def print_indices(self):
    rows = [['index group', 'description']]
    for letter, tup in self.index.items():
        ilist = ', '.join(self.index_list(letter))
        if ilist == '':
            ilist = '<empty>'
        rows.append([ilist, tup.name])
    tab, _ = tabulate(rows, '   ', True)
    return '\n'.join(tab)

def print_shapes(self, shapes, highlight):
    msg = ''
    for shape in shapes:
        msg += '\n'.join(shape.index_usage(highlight))
        msg += '\n\n'
    return msg

def print_inputs(self, highlight=None):
    return self.print_shapes(self.input_shapes, highlight)

def print_outputs(self, highlight=None):
    return self.print_shapes(self.return_shapes, highlight)
"""

