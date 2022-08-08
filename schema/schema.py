from ast_nodes import EinTup, SchemaFunctionExpr
from .arg import *
from .schema_internal import SchemaInternal

class Schema(object):
    # Interface for authors to define schemas for ops in op_schema directory
    def __init__(self, op_path):
        self.p = SchemaInternal(op_path)

    def set_init(self, init_func):
        """Define the schema init function"""
        self.p.init_schema = init_func

    def set_calltime_config(self, config_func):
        self.p.calltime_config = config_func

    def add_index(self, letter_name, full_name):
        self.p.index[letter_name] = EinTup(full_name)

    def get_index(self, letter_name):
        if letter_name not in self.p.index:
            raise RuntimeError(
                f'Schema does not contain index \'{letter_name}\'. '
                f'Available indices are {self.p.index.keys()}')
        return self.p.index[letter_name]

    def get_arg(self, arg_name, default=None):
        """Get the call-time argument provided to the framework function, or
        default if None"""
        return self.p.get_arg(arg_name, default)

    def add_input_tensor(self, arg_name, signature):
        """Tell the schema to expect {arg_name} to be a Tensor and interpret its
        shape as {signature}"""
        self.p.check_sig(signature, arg_name)
        self.p.input_shapes.append(TensorShapeInput(self, arg_name, signature))

    def add_input_shape(self, arg_name, signature):
        """Tell the schema to expect {arg_name} to be a list and interpret its
        value to be the shape of {signature}."""
        self.p.check_sig(signature, arg_name)
        self.p.input_shapes.append(ListShapeInput(self, arg_name, signature))

    def add_input_sigrank(self, arg_name, signature, beg, end, num_test):
        """Expect {arg_name} to be a list of length rank({signature}), with
        elements in [{beg}, {end}).  For testing, produce {num_test} values"""
        pass

    def add_input_static(self, arg_name, value_list):
        """Expect {arg_name} to take on one of the values in {value_list}"""
        pass

    def set_shape_signature(self, arg_name, signature):
        """Hook to set the {signature} associated with {arg_name} at runtime """
        self.p.check_sig(signature, arg_name)
        shape = next((sh for sh in self.p.input_shapes if sh.name == arg_name), 
                None)
        if shape is None:
            raise RuntimeError(
                f'set_shape_signature: could not find any shape '
                f'called \'{arg_name}\'.  Shapes are registered with '
                f'add_input_tensor() or add_input_shape()')
        shape.sig = signature

    def add_input_rank(self, arg_name, signature):
        """Expect {arg_name} to be an integer which defines the rank of
        {signature}"""
        self.p.check_sig(signature, arg_name)
        self.p.input_ranks.append(RankInput(self, arg_name, signature))

    def append_output_tensor(self, signature):
        idx = len(self.p.output_shapes) 
        self.p.check_sig(signature, f'output {idx}')
        # the shape gets populated during 'validate' call
        self.p.output_shapes.append(TensorShapeOutput(self, idx, signature))

    def equate_index_ranks(self, idx1, idx2):
        """Constrain ranks of indices {idx1} and {idx2} to be equal"""
        tup1 = self.p.get_index(idx1)
        tup2 = self.p.get_index(idx2)
        tup1.equate_rank(tup2)

    def set_index_rank_range(self, idx, rng):
        """Constrain index {idx} to have a rank in {rng}"""
        tup = self.p.get_index(idx)
        tup.set_rank_range(rng)

    def set_index_rank(self, idx, val):
        """Constrain index {idx} to have rank equal to {val}"""
        tup = self.p.get_index(idx)
        tup.set_rank_range(range(val, val+1))

    def set_index_rank_constraint(self, idx, rank_func):
        """Constrain the rank of index {idx} to the function value
        {rank_func}(op).  The function is evaluated at op runtime """
        tup = self.p.get_index(idx)
        rank_func = SchemaFunctionExpr(rank_func, self)
        tup.add_rank_expr(rank_func)
        
    # constraint is a function accepting self 
    def set_index_dims_constraint(self, idx, dims_func):
        """Constrain the dims of {idx} to the function value {dims_func}(op).
        The function is evaluated at op runtime.
        {dims_func}(op) should return an integer list of length rank(idx) 
        """
        tup = self.p.get_index(idx)
        dims_func = SchemaFunctionExpr(dims_func, self)
        tup.add_gen_expr(dims_func)

    def equate_element_type(self, tensor_name1, tensor_name2):
        """Declare that two tensor inputs must have the same element type"""
        pass

    def allowed_element_types(self, tensor_name, type_list):
        """Declare that tensor can only have certain element types"""
        pass

