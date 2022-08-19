import tensorflow as tf
from numpy.random import randint
from .arg import *
from .fgraph import GenNode, PredNode
from .schema_internal import SchemaInternal

class Schema(object):
    # Interface for authors to define schemas for ops in op_schema directory
    def __init__(self, op_path):
        self.p = SchemaInternal(op_path)

    def index(self, idx, description):
        """Add index {idx} with {description} to the schema.  {idx} must be a
        single letter and can be referred to in later signatures"""
        self.p.index[idx] = description

    def get_arg(self, arg_name, default=None):
        """Get the call-time argument provided to the framework function, or
        default if None"""
        return self.p.get_arg(arg_name, default)

    def arg_tensor(self, arg_name, signature):
        """Expect {arg_name} to be a Tensor with {signature}"""
        self.p.check_sig(signature, arg_name)
        self.p.set_arg_type(arg_name, Tensor)
        # self.p.arg_shape[arg_name] = TensorShapeArg(self, arg_name, signature) 
        def gen(dims, dtypes):
            return schema_internal.generate_tensor(arg_name, signature, dims,
                    dtypes)
        GenNode.add_node(arg_name, gen, DIMS, DTYPES)
        # Info for validating signature shape
        self.p.check_arg_added(arg_name, self.p.sig_dims)
        def dims_func(op):
            ten = op.get_arg(arg_name)
            return ten.shape.as_list()
        self.p.sig_dims[arg_name] = dims_func

    def arg_shape(self, arg_name, signature):
        """
        Expect {arg_name} to be a list which defines the shape of {signature}
        """ 
        self.p.check_sig(signature, arg_name)
        self.p.set_arg_type(arg_name, list)
        # self.p.arg_shape[arg_name] = ListShapeArg(self, arg_name, signature)
        def gen(dims):
            shape = schema_internal.sig_dims(dims, signature)
            return [shape]
        GenNode.add_node(arg_name, gen, DIMS)
        # Info for validating signature shape
        self.p.check_arg_added(arg_name, self.p.sig_dims)
        def dims_func(op):
            return op.get_arg(arg_name)
        self.p.sig_dims[arg_name] = dims_func

    def arg_rank(self, arg_name, signature):
        """
        Expect {arg_name} to be an integer that defines the rank of {signature}
        """
        self.p.check_sig(signature, arg_name)
        self.p.set_arg_type(arg_name, int)
        # self.p.check_arg_added(arg_name, self.p.arg_rank)
        def gen(ranks):
            rank = sum(r for idx,r in ranks if idx in signature)
            return [rank]
        GenNode.add_node(arg_name, gen, RANK)
        # Info for validating signature rank
        self.p.check_arg_added(arg_name, self.p.sig_ranks)
        def rank_func(op):
            return op.get_arg(arg_name)
        self.p.sig_ranks[arg_name] = rank_func
        # self.p.arg_rank[arg_name] = RankArg(self, arg_name, signature) 

    def arg_rank_func(self, signature, func, *arg_names):
        """
        Register {func} to define the rank of {signature}.  Called as
        func(*arg_vals), where arg_vals are derived from arg_names.  arg_names
        must appear as parameters in the framework op.
        """
        self.p.check_sig(signature, 'arg_rank_func')
        for name in arg_names:
            self.p.check_arg_name(name)
        self.p.check_arg_added(arg_name, self.p.sig_ranks)

        def rank_func(op):
            args = tuple(op.get_arg(n) for n in arg_names)
            return func(*args)
        self.p.sig_ranks[arg_name] = rank_func

    def arg_option(self, arg_name, options):
        """Expect {arg_name} to take on one of the values in {options}"""
        try:
            iter(options)
        except TypeError:
            raise RuntimeError(
                f'{type(self).__qualname__}: \'options\' argument must be '
                f'iterable.  Got {type(options)}')
        def gen():
            return options
        GenNode.add_node(arg_name, gen)
        def pred(op):
            arg_val = op.get_arg(arg_name)
            if arg_val in options:
                return True, arg_val
            else:
                return False, NonOptionError(arg_name, arg_val) 
        PredNode.add_node(arg_name, pred, OP)

    def arg_unchecked(self, arg_name):
        """
        Declare {arg_name} to be an argument unchecked by OpCheck 
        """
        self.p.set_arg_type(arg_name, None)

    def tensor_valid_dtypes(self, tensor_name, type_list):
        """
        Declare {tensor_name} can have any of the dtype strings in {type_list}.
        Names in {type_list} are converted via tf.dtypes.as_dtype(name).
        e.g. names like 'int32', 'int64', 'float32'
        """
        arg_type = self.p.get_arg_type(tensor_name)
        if arg_type != tf.Tensor:
            raise RuntimeError(
                f'{type(self).__name__}: Parameter \'{tensor_name}\' is '
                f'registered as type {arg_type}.  Can only call on tf.Tensor')
        if tensor_name in self.p.dtype_valid:
            raise RuntimeError(
                f'{self.__qualname__}: Tensor \'{tensor_name}\' is already '
                f'registered with valid dtypes')

        dtypes = []
        for t in type_list:
            try:
                dt = tf.dtypes.as_dtype(t)
                dtypes.append(dt)
            except TypeError:
                raise RuntimeError(
                    f'{self.__qualname__}: Type string \'{t}\' is not a valid '
                    f'tf.dtype representation')
        self.p.dtype_valid[tensor_name] = tuple(dtypes)

    def tensor_equate_dtypes(self, src_tensor, trg_tensor):
        """
        Declare that {trg_tensor} have the same dtype as {src_tensor}.
        Must first call arg_valid_dtypes(src_tensor, ...).
        trg_tensor must not be called in arg_valid_dtypes if it is called
        here.
        """
        src_type = self.get_arg_type(src_tensor)
        trg_type = self.get_arg_type(trg_tensor)
        if src_type != tf.Tensor or trg_type != tf.Tensor:
            raise RuntimeError(
                f'{type(self).__name__}: Can only be called on two tensors. '
                f'Parameters \'{src_tensor}\' and \'{trg_tensor}\' are types '
                f'{src_type} and {trg_type}')
        if src_tensor not in self.p.dtype_valid:
            raise RuntimeError(
                f'{self.__qualname__}: Must first register valid types for '
                f'src_tensor (\'{src_tensor}\'')
        if trg_tensor in self.p.dtype_valid:
            raise RuntimeError(
                f'{self.__qualname__}: trg_tensor (\'{trg_tensor}\') '
                f'was already called in arg_valid_dtypes so cannot be called '
                f'here')
        self.p.dtype_equiv.append((src_tensor, trg_tensor))

    def add_input_sigrank(self, arg_name, signature, beg, end):
        """
        Expect {arg_name} to be a list of length rank({signature}), with
        elements in [{beg}, {end})
        """
        def gen(ranks):
            rank = sum(ranks[s] for s in signature)
            val = [randint(beg, end) for _ in range(rank)]
            return [val]
        GenNode.add_node(arg_name, gen, RANK)
        def pred(op, ranks):
            arg_val = op.get_arg(arg_name)
            rank = sum(ranks[s] for s in signature)
            if len(arg_val) != rank:
                return False, SigRankError(arg_name, rank, len(arg_val))
            else:
                return True, arg_val
        PredNode.add_node(arg_name, pred, OP, RANK)

    def append_return_tensor(self, signature):
        """
        Append a return tensor to the list of expected return tensors and
        expect it to have {signature}.
        """
        idx = len(self.p.return_shapes) 
        self.p.check_sig(signature, f'return {idx}')
        # the shape gets populated during 'validate' call
        self.p.return_shapes.append(TensorShapeReturn(self, idx, signature))

    def limit_ranks(self, sig, min_val, max_val):
        """
        Declare that the valid ranks of {sig} lie in the interval [{min_val},
        {max_val}]
        """
        self.p.check_sig(sig, 'rank limits')
        self.p.add_rank_limits(sig, min_val, max_val)

    def equate_ranks(self, sig1, sig2):
        """Declare that the rank of {sig1} and {sig2} must be equal"""
        self.p.check_sig(sig1, 'equate ranks')
        self.p.check_sig(sig2, 'equate ranks')
        self.p.rank_equiv.append((sig1, sig2))

    def index_dims_func(self, idx, dims_func, *arg_names):
        """
        Register a custom function {dims_func} to define the dims of {idx}.
        dims_func will be called as dims_func(dims_map, *arg_vals) where
        arg_vals are produced from {arg_names} at runtime.  {arg_names} may be
        any names of the framework op arguments.  dims_map is a map of idx =>
        dims for the currently resolved index dimensions.
        """
        for name in arg_names:
            self.p.check_arg_name(name)
        self.p.check_arg_added(idx, self.p.dims_from_dims)
        self.p.dims_from_dims[idx] = (dims_func, arg_names)

    def index_rank_func(self, idx, rank_func, *arg_names):
        """
        Register a custom function {rank_func} to define the dims of {idx}.
        Will be called as rank_func(rank_map, *arg_vals), where arg_vals are
        produced from {arg_names} at runtime.  {arg_names} must be argument
        names appearing in the framework op signature.  rank_map is a map of
        idx => rank for the currently resolved index dimensions.
        """
        for name in arg_names:
            self.p.check_arg_name(name)
        self.p.check_arg_added(idx, self.p.dims_from_rank)
        self.p.dims_from_rank[idx] = (rank_func, arg_names)

