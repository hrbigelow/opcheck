from tensorflow import Tensor
from ast_nodes import EinTup, SchemaFunctionExpr
from schema_internal import SchemaInternal, ShapeInput, RankInput

class Schema(object):
    # Interface for authors to define schemas for ops in op_schema directory
    def __init__(self, op_path):
        self.p = SchemaInternal(op_path)

    def set_init(self, init_func):
        """Define the schema init function"""
        self.p.init_schema = init_func

    def add_index(self, letter_name, full_name):
        self.p.index[letter_name] = EinTup(full_name)

    def get_arg(self, arg_name):
        if arg_name not in self.p.arguments:
            raise RuntimeError(
                f'\'{arg_name}\' not found in call arguments. '
                f'Arguments are: {self.arguments.keys()}')
        return self.p.arguments[arg_name]

    def add_input_tensor(self, name, signature):
        self.p.check_sig(name, signature)
        ten = self.get_arg(name)
        if not isinstance(ten, Tensor):
            raise RuntimeError(
                f'Expected argument \'{name}\' to be a Tensor but it is a '
                f'{type(ten)}')
        shape = ten.shape.as_list()
        self.p.input_shapes.append(ShapeInput(self, name, 'tensor', shape,
            signature))

    # Add a list parameter called 'name' to the input shapes to check
    def add_input_shape(self, name, signature):
        self.p.check_sig(name, signature)
        shape = self.get_arg(name)
        if not isinstance(shape, list):
            raise RuntimeError(
                f'Expected argument \'{name}\' to be a list but it is a '
                f'{type(shape)}')
        self.p.input_shapes.append(ShapeInput(self, name, 'param', shape,
            signature))

    def add_input_rank(self, name, signature):
        self.p.check_sig(name, signature)
        rank = self.get_arg(name)
        if not isinstance(rank, int):
            raise RuntimeError(
                f'Expected argument \'{name}\' to be an integer but it is a '
                f'{type(rank)}')
        self.p.input_ranks.append(RankInput(self, name, rank, signature))

    def append_output_tensor(self, name, signature):
        self.p.check_sig(name, signature)
        # the shape gets populated during 'validate' call
        self.p.output_shapes.append(ShapeInput(self, name, 'output tensor', 
                None, signature))

    def equate_rank(self, letter1, letter2):
        tup1 = self.p.get_index(letter1)
        tup2 = self.p.get_index(letter2)
        tup1.equate_rank(tup2)

    def set_rank_range(self, letter_name, rng):
        tup = self.p.get_index(letter_name)
        tup.set_rank_range(rng)

    # constraint is a function accepting self 
    def add_dims_constraint(self, letter_name, constraint):
        tup = self.p.get_index(letter_name)
        static_expr = SchemaFunctionExpr(constraint, self)
        tup.add_gen_expr(static_expr)

