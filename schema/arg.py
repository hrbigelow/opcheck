from tensorflow import Tensor
import random

class RankInput(object):
    # An input which defines the rank of some signature
    def __init__(self, schema, name, sig):
        self.p = schema.p
        self.name = name
        self.sig = sig

    def __repr__(self):
        return f'RankInput({self.name}({self.sig})'

    def rank(self):
        rank = self.p.get_arg(self.name)
        if not isinstance(rank, int):
            raise RuntimeError(
                f'Expected argument \'{self.name}\' to be an integer '
                f'but it is a {type(rank)}')
        return rank

    def valid_rank(self):
        return self.p.sig_rank(self.sig) == self.rank()

class InputArg(object):
    def __init__(self, schema, name):
        self.p = schema.p
        self.name = name

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    def get(self):
        return self.p.get_arg(self.name)

class Output(object):
    def __init__(self, schema, out_idx):
        self.p = schema.p
        self.idx = out_idx

    def __repr__(self):
        return f'{self.__class__.__name__}[{self.idx}]'

    def get(self):
        return self.p.get_output(self.idx)
    

class Shape(object):
    """Interprets an input or output {arg} to have signature {sig}"""
    def __init__(self, arg, sig):
        self.arg = arg
        self.sig = sig

    def __repr__(self):
        return f'{self.__class__.__name__}({self.arg})[{self.sig}]'

    # return whether the actual rank and rank predicted by the indices
    # are the same
    def valid_rank(self):
        return self.arg.p.sig_rank(self.sig) == self.rank()

    def dims(self):
        raise NotImplementedError

    def rank(self):
        return len(self.dims())

    def sub_dims(self, letter_idx):
        b, e = self.arg.p.sig_range(letter_idx, self.sig)
        return self.dims()[b:e]

    # return a 3-member array, for example:
    # 'input[ b,  i1,  i2, k]',
    # '     [10, 100, 100, 3]',
    # '                    ^ '
    # shows the signature interpretation (b, i1, i2, k), the actual shape
    # (10, 100, 100, 3), and the highlighted usage of the sig_letter (k)
    # This is useful for highlighting shape constraint violations to the user
    def index_usage(self, highlight_letter=None):
        rows = [ self.p.sig_list(self.sig), self.dims() ]
        table, coords = tabulate(rows, ', ', left_justify=False) 
        out1 = f'{self.name}[{table[0]}]'
        out2 = f'[{table[1]}]'
        out = [ [out1], [out2] ]
        width = len(table[0])

        if highlight_letter is not None:
            b, e = self.p.sig_range(highlight_letter, self.sig)
            rng = range(coords[b][0], coords[e-1][1])
            highlight = ''.join('^' if i in rng else ' ' for i in range(width))
            out3 = f'{highlight} ' # trailing space aligns with closing bracket
            out.append([out3])

        justify, _ = tabulate( out, '', left_justify=False)
        return justify

class ShapeInput(Shape):
    """Input argument {name} interpreted with signature {sig}"""
    def __init__(self, schema, name, sig):
        arg = InputArg(schema, name)
        super().__init__(arg, sig)

class ShapeOutput(Shape):
    """Output result {idx} interpreted with signature {sig}"""
    def __init__(self, schema, idx, sig):
        arg = Output(schema, idx)
        super().__init__(arg, sig)

class TensorShapeInput(ShapeInput):
    def __init__(self, schema, name, sig):
        super().__init__(schema, name, sig)

    def dims(self):
        ten = self.arg.get()
        if not isinstance(ten, Tensor):
            raise RuntimeError(
                f'TensorShapeInput: expected argument \'{self.name}\' to be '
                f'a Tensor but it is a {type(ten)}')
        return ten.shape.as_list()

class TensorShapeOutput(ShapeOutput):
    def __init__(self, schema, out_idx, sig):
        super().__init__(schema, out_idx, sig)

    def dims(self):
        ten = self.arg.get()
        if not isinstance(ten, Tensor):
            raise RuntimeError(
                f'{self.__class__.__name__} expected output \'{self.idx}\' to be '
                f'a Tensor but it is a {type(ten)}')
        return ten.shape.as_list()

class ListShapeInput(ShapeInput):
    """Represents an integer list argument which defines a signature shape.
    For example:

    # For tf.scatter_nd (shape parameter defines w plus e (write address plus
    # slice element) index shape)
    ListShapeInput(schema, 'shape', 'we')
    """
    def __init__(self, schema, name, sig):
        super().__init__(schema, name, sig)

    def dims(self):
        shape = self.arg.get()
        if not isinstance(shape, list):
            raise RuntimeError(
                f'{self.__class__.__name__}: expected argument \'{self.name}\' '
                f'to be a list but it is a {type(shape)}')
        return shape


class ArgCheck(object):
    """Base class for representing checked arguments.
    """
    def __init__(self, schema, arg_name):
        self.schema = schema
        self.name = arg_name

    def call_value(self):
        """Return the value supplied by the call"""
        return self.schema.get_arg(self.name)

    def valid_call(self):
        """Judge whether the call value is valid within the constraints
        defined in this ArgCheck object"""
        raise NotImplementedError

    def test_values(self):
        """Generate a list of test values to run the test harness.  May or may
        not be an exhaustive list"""
        pass

    def error_message(self):
        """A message to return to the user if valid_call() returns False"""
        raise NotImplementedError

class StaticArg(ArgCheck):
    """ArgCheck that represents an argument that can take on a fixed set of
    values.  For example:

    StaticArg(schema, 'padding', ['VALID', 'SAME'])
    StaticArg(schema, 'indexing', ['ij', 'xy'])
    """
    def __init__(self, schema, arg_name, options_list):
        super().__init__(schema, arg_name)
        self.options = options_list

    def valid_call(self):
        return self.call_value() in self.options

    def test_values(self):
        return self.options

    def error_message(self):
        return (
                f'Argument \'{self.name}\' received invalid value '
                f'\{self.call_value()}\'.  Allowed values are '
                f'{self.options}')

class SigRankRange(ArgCheck):
    """Define a list of integers in a range, whose length matches the rank of
    the signature

    For example, these will generate two separate dilations or strides arrays
    between 1 and 9, having the same rank as the 'i' (input spatial)

    SigRankRange(schema, 'dilations', 'i', 1, 10, 2) 
    SigRankRange(schema, 'strides', 'i', 1, 10, 2)
    """
    def __init__(self, schema, arg_name, signature, beg, end, num_test):
        super().__init__(schema, arg_name)
        self.schema.p.check_sig(signature)
        self.sig = signature
        self.beg = beg
        self.end = end
        self.num_test = num_test

    def valid_call(self):
        vals = self.call_value()
        rank = self.schema.p.sig_rank(self.sig)
        rng = range(self.beg, self.end)
        return len(vals) == rank and all(v in rng for v in vals)

    def test_values(self):
        rank = self.schema.p.sig_rank(self.sig)
        a, b = self.rng.start, self.rng.stop
        return [ 
                [random(self.beg, self.end-1) for _ in range(rank) ]
                for _ in range(self.num_test) 
                ]

