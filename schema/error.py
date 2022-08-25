class SchemaError(BaseException):
    """Represents an error in the Schema definition"""
    def __init__(self, msg):
        self.msg = msg

class SchemaStatus(BaseException):
    """Represent violations of schema constraints"""
    def __init__(self):
        pass

    def message(self, op):
        """
        """
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
        return f'{repr(self.ex.message)}'

class NoMatchingRanks(SchemaStatus):
    def __init__(self):
        pass

    def message(self, op):
        msg = 'No matching ranks found'
        return msg

class AmbiguousRanks(SchemaStatus):
    def __init__(self):
        pass

    def message(self, op):
        msg = 'Ambiguous ranks'
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

class SigRankError(SchemaStatus):
    def __init__(self, arg_name, expected_length, actual_length):
        self.arg_name = arg_name
        self.expected_length = expected_length
        self.actual_length = actual_length

    def message(self, op):
        msg = (f'Argument \'{self.arg_name}\' was expected to be of length '
                f'{self.expected_length} but had actual length '
                f'{self.actual_length}')
        return msg


class RankDependentArgError(SchemaStatus):
    def __init__(self, arg_name):
        self.arg_name = arg_name


class ShapeError(SchemaStatus):
    """
    tensor {ten_name} at {idx} has sub-dimensions {ten_sub_dims}, which are not
    equal to the inferred dimensions of {idx}
    """
    def __init__(self, ten_name, idx, ten_sub_dims):
        self.ten_name = ten_name
        self.idx = idx
        self.ten_sub_dims = ten_sub_dims

    def message(self, op):
        dims_map = op.get_index_dims()
        expect_dims = dims_map[self.idx] 
        msg = f'Tensor input {self.ten_name} had sub-dimensions '
        msg += f'{self.ten_sub_dims} but expected {expect_dims}'
        return msg

class OutputShapeError(SchemaStatus):
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
def tabulate(rows, sep, left_justify=True):
    n = len(rows[0])
    w = [max(len(str(row[c])) for row in rows) for c in range(n)]
    if left_justify:
        t = [sep.join(f'{str(row[c]):<{w[c]}s}' for c in range(n))
                for row in rows]
    else:
        t = [sep.join(f'{str(row[c]):>{w[c]}s}' for c in range(n))
                for row in rows]

    begs = [sum(w[:s]) + len(sep) * s for s in range(n)]
    ends = [sum(w[:s+1]) + len(sep) * s for s in range(n)]
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

