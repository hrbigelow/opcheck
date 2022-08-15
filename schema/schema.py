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

    def index(self, idx, description):
        """Add index {idx} with {description} to the schema.  {idx} must be a
        single letter and can be referred to in later signatures"""
        self.p.index[idx] = description

    def get_index(self, idx):
        if idx not in self.p.index:
            raise RuntimeError(
                f'Schema does not contain index \'{idx}\'. '
                f'Available indices are {self.p.index.keys()}')
        return self.p.index[idx]

    def get_index_dims(self, idx):
        """
        Get the current dims inferred for {idx}.  Must be called after the Dims
        Resolution Phase
        """
        if idx not in self.p.index_dims:
            raise RuntimeError(
                f'Index dims for \'{idx}\' not available. '
                f'Available dims are {self.p.index_dims.keys()}')
        return self.p.index_dims[idx]

    def get_index_rank(self, idx):
        """Get the current rank inferred for {idx}.  Cannot be called until 
        after the Dims Resolution Phase"""
        if idx not in self.p.index_ranks:
            raise RuntimeError(
                f'Index rank for \'{idx}\' not available. '
                f'Available dims are {self.p.index_ranks.keys()}')
        return self.p.index_ranks[idx]

    def get_arg(self, arg_name, default=None):
        """Get the call-time argument provided to the framework function, or
        default if None"""
        return self.p.get_arg(arg_name, default)

    def arg_tensor(self, arg_name, signature):
        """Expect {arg_name} to be a Tensor with {signature}"""
        self.p.check_sig(signature, arg_name)
        self.p.check_arg_added(arg_name, self.arg_shape)
        typed_arg = self.p.maybe_add_arg(arg_name, Tensor)
        self.p.arg_shape[arg_name] = TensorShapeArg(self, arg_name, signature) 

    def arg_shape(self, arg_name, signature):
        """Expect {arg_name} to be a list which defines the shape of {signature}
        """ 
        self.p.check_sig(signature, arg_name)
        self.p.check_arg_added(arg_name, self.arg_shape)
        self.p.arg_shape[arg_name] = ListShapeArg(self, arg_name, signature)

    def arg_rank(self, arg_name, signature):
        """Expect {arg_name} to be an integer that defines the rank of
        {signature}"""
        self.p.check_sig(signature, arg_name)
        self.p.check_arg_added(arg_name, self.arg_rank)
        self.p.arg_rank[arg_name] = RankArg(self, arg_name, signature) 

    def arg_rank_func(self, arg_name, signature, func):
        """Call {func} on the value of {arg_name}, and set the rank of
        {signature} to the return value."""
        self.p.check_sig(signature, arg_name)
        self.p.check_arg_added(arg_name, self.arg_rank)
        self.p.arg_rank[arg_name] = RankFuncArg(self, arg_name, signature,
                func)

    def arg_option(self, arg_name, options):
        """Expect {arg_name} to take on one of the values in {options}"""
        pass

    def add_input_sigrank(self, arg_name, signature, beg, end, num_test):
        """Expect {arg_name} to be a list of length rank({signature}), with
        elements in [{beg}, {end}).  For testing, produce {num_test} values"""
        pass

    def append_output_tensor(self, signature):
        idx = len(self.p.output_shapes) 
        self.p.check_sig(signature, f'output {idx}')
        # the shape gets populated during 'validate' call
        self.p.output_shapes.append(TensorShapeOutput(self, idx, signature))

    def limit_ranks(self, sig, min_val, max_val):
        """Declare that the valid ranks of {sig} lie in the interval
        [{min_val}, {max_val}]"""
        self.p.check_sig(sig, 'rank limits')
        self.p.add_rank_limits(sig, min_val, max_val)

    def equate_ranks(self, sig1, sig2):
        """Declare that the rank of {sig1} and {sig2} must be equal"""
        self.p.check_sig(sig1, 'equate ranks')
        self.p.check_sig(sig2, 'equate ranks')
        self.p.rank_equiv.append((sig1, sig2))

    def set_rank(self, idx, rank_func):
        """Set the rank of {idx} to the output of {rank_func}(op).  Note that
        {rank_func} cannot access other ranks or dimensions, since it executes
        during the rank inference phase"""
        pass
        
    # constraint is a function accepting self 
    def set_index_dims_constraint(self, idx, dims_func):
        """Constrain the dims of {idx} to the function value {dims_func}(op).
        {dims_func} is evaluated at the end of the Dims Resolution Phase. 
        {dims_func}(op) should return an integer list of length rank(idx) 
        """
        self.index_dims_funcs[idx] = dims_func

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


    def equate_element_type(self, tensor_name1, tensor_name2):
        """Declare that two tensor inputs must have the same element type"""
        pass

    def allowed_element_types(self, tensor_name, type_list):
        """Declare that tensor can only have certain element types"""
        pass

