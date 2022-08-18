class SchemaError(object):
    """Represent violations of schema constraints"""
    def __init__(self):
        pass

    def message(self, op):
        """

        """
        raise NotImplementedError

class NoMatchingRanks(SchemaError):
    def __init__(self):
        pass

    def message(self, op):
        msg = 'No matching ranks found'
        return msg

class AmbiguousRanks(SchemaError):
    def __init__(self):
        pass

    def message(self, op):
        return ''

class ArgTypeError(SchemaError):
    def __init__(self, arg_name):
        self.arg_name = arg_name

class TensorDTypeError(SchemaError):
    def __init__(self, ten_name):
        self.ten_name = ten_name

class IndexUsageError(SchemaError):
    def __init__(self, idx):
        self.idx = idx

class ShapeError(SchemaError):
    def __init__(self, ten_name, index_letter, ten_sub_dims):
        self.ten_name = ten_name
        self.index_letter = index_letter
        self.ten_sub_dims = ten_sub_dims

    def message(self, op):
        expect_dims = op.index[self.index_letter].dims()
        msg = (f'Tensor input {self.ten_name} had sub-dimensions {self.ten_sub_dims} '
                f'but expected {expect_dims}')
        return msg

class OutputShapeError(SchemaError):
    """The output at {out_idx} does not match the shape implied by its
    signature"""
    def __init__(self, out_idx):
        self.idx = out_idx

class OutputNumberMismatch(SchemaError):
    """The number of outputs returned differed from expected"""
    def __init__(self, num_actual_outputs):
        self.num_actual_outputs = num_actual_outputs


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

