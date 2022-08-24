import tensorflow as tf
import functools
from numpy.random import randint
from .fgraph import GenNode, PredNode
from . import backend
from .error import *

IName = backend.IName 

class SchemaApi(object):

    # Interface for authors to define schemas for ops in op_schema directory
    def __init__(self, op_path):
        self.p = backend.Backend(op_path)

    def index(self, idx, description):
        """
        Add index {idx} with {description} to the backend.  {idx} must be a
        single letter and can be referred to in later signatures
        """
        self.p.index[idx] = description

    def get_arg(self, arg_name, default=None):
        """
        Get the call-time argument provided to the framework function, or
        default if None
        """
        return self.p.get_arg(arg_name, default)

    def arg_tensor_func(self, arg_name, sig_func, *sig_func_args):
        """
        Register {arg_name} as a tensor.  Its signature is computed at
        call-time as {sig_func}(*sig_func_vals). 

        sig_func_vals are the resolved values of {sig_func_args}.

        sig_func_args must be names of operation argments or pseudo-arguments
        declared with arg_pseudo.
        """
        def gen_tensor(sigs_map, dims_map, dtypes):
            sig = sigs_map[arg_name]
            return backend.generate_tensor(arg_name, sig, dims_map, dtypes)

        # Only generate tensor inputs, not tensor return values
        if not arg_name.startswith(IName.RETURN):
            self.p.set_arg_type(arg_name, tf.Tensor)
            GenNode.add_node(arg_name, gen_tensor, IName.SIGS, IName.DIMS,
                    IName.DTYPES)

        # Used by the predicate function valid_dims.
        # Only one PredNode for all shape-related inputs is created
        self.p.check_arg_added(arg_name, self.p.arg_sigs)
        self.p.arg_sigs[arg_name] = (sig_func, sig_func_args)

        def dims_func():
            if arg_name.startswith(IName.RETURN):
                idx = self.p.get_return_index(arg_name)
                ten = self.p.get_return(idx)
            else:
                ten = self.get_arg(arg_name)
            dims = ten.shape.as_list()
            return dims

        self.p.check_arg_added(arg_name, self.p.arg_dims)
        self.p.arg_dims[arg_name] = dims_func, tuple() 

    def arg_tensor(self, arg_name, signature):
        """
        Expect {arg_name} to be a Tensor with {signature}
        """
        return self.arg_tensor_func(arg_name, lambda: signature)

    def append_return_tensor(self, signature):
        """
        Append a return tensor to the list of expected return tensors and
        expect it to have {signature}.
        """
        return self.append_return_tensor_func(lambda: signature)

    def append_return_tensor_func(self, sig_func, *arg_names):
        """
        Append a tensor to the list of expected returns.  {sig_func}(*arg_vals)
        should return the expected signature associated with the tensor.

        arg_vals are the values of arg_names resolved at call-time. 

        arg_names may be any names declared using this API, as well as those in
        schema.api.IName
        """
        idx = self.p.num_returns()
        arg_name = self.p.get_return_name(idx)
        return self.arg_tensor_func(arg_name, sig_func, *arg_names)

    def arg_shape(self, arg_name, signature):
        """
        Expect {arg_name} to be a list which defines the shape of {signature}
        """ 
        self.p.check_sig(signature, arg_name)
        self.p.set_arg_type(arg_name, list)
        def gen(dims_map):
            shape = backend.calc_sig_dims(dims_map, signature)
            return [shape]
        GenNode.add_node(arg_name, gen, IName.DIMS)
        # Info for validating signature shape
        self.p.check_arg_added(arg_name, self.p.arg_sigs)
        def sig_func():
            return signature
        self.p.arg_sigs[arg_name] = sig_func, tuple()

        def dims_func():
            arg_val = self.get_arg(arg_name)
            return arg_val
        self.p.arg_dims[arg_name] = dims_func, tuple() 

    def arg_rank(self, arg_name, signature):
        """
        Expect {arg_name} to be an integer that defines the rank of {signature}
        """
        self.p.check_sig(signature, arg_name)
        self.p.set_arg_type(arg_name, int)
        def gen(ranks):
            rank = sum(r for idx,r in ranks.items() if idx in signature)
            return [rank]
        GenNode.add_node(arg_name, gen, IName.RANKS)
        # Info for validating signature rank
        self.p.check_arg_added(signature, self.p.sig_ranks)
        def rank_func(op):
            return op.get_arg(arg_name)
        self.p.sig_ranks[signature] = rank_func

    def arg_rank_func(self, signature, func, *arg_names):
        """
        Register {func} to define the rank of {signature}.  Called as
        func(*arg_vals), where arg_vals are derived from arg_names.  arg_names
        must appear as parameters in the framework op.
        """
        self.p.check_sig(signature, 'arg_rank_func')
        for name in arg_names:
            self.p.check_arg_name(name)
        self.p.check_arg_added(signature, self.p.sig_ranks)

        def rank_func(op):
            args = tuple(op.get_arg(n) for n in arg_names)
            return func(*args)
        self.p.sig_ranks[signature] = rank_func

    def arg_rank_dependent(self, arg_name, gen_func, pred_func):
        """
        Declare {arg_name} to be validated by {pred_func}(rank_map, arg_val)
        and generated with {gen_func}(rank_map).  arg_val is the call time
        value of {arg_name}, and rank_map contains the inferred ranks of each
        index at call time.
        """
        self.p.check_arg_name(arg_name)
        GenNode.add_node(arg_name, gen_func, IName.RANKS)

        def pred_wrap(ranks):
            arg_val = self.p.get_arg(arg_name)
            valid = pred_func(ranks, arg_val)
            if valid:
                return True, None
            else:
                return False, RankDependentArgError(arg_name) 
        PredNode.add_node(arg_name, pred_wrap, IName.RANKS)

    def arg_pseudo(self, pseudo_name, gen_func, val_func, arg_name):
        """
        Creates a pseudo-input argument called {pseudo_name}, which is used to
        break a dependency cycle in nodes of the Generation Graph or Predicate
        graph.

        {gen_func}() generates all legal values for the pseudo argument during
        the schema validation phase.

        {val_func}(arg_val) returns a derived value which represents the
        pseudo-input's value.  It is as if that value were provided directly to
        the framework operation.

        val_func must always succeed, even if arg_val is not a valid input for
        the real argument {arg_name}.  val_func may assume that the type of
        arg_val is already validated.
        """
        GenNode.add_node(pseudo_name, gen_func)

        def wrap_pred():
            arg_val = self.p.get_arg(arg_name)
            pseudo_val = val_func(arg_val)
            return True, pseudo_val
        PredNode.add_node(pseudo_name, wrap_pred)

    def arg_func(self, arg_name, gen_func, pred_func, *arg_names):
        """
        While running the op, validates {arg_name} with a call to the
        predicate as: {pred_func}(arg_val, *arg_vals)

        Generates testing values for {arg_name} with a call to the generator
        function as: {gen_func}(*arg_vals)

        arg_vals are the values resolved at call-time from {arg_names}.
        arg_names may contain any names defined by this Schema API.
        Additionally, arg_names may contain IName enum values.
        """
        GenNode.add_node(arg_name, gen_func, *arg_names)
        def wrap_pred(*arg_vals):
            arg_val = self.p.get_arg(arg_name)
            valid = pred_func(arg_val, *arg_vals)
            if valid:
                return True, arg_val
            else:
                return False, ArgValueError(arg_name, arg_val) 
        PredNode.add_node(arg_name, wrap_pred, *arg_names)

    def arg_option(self, arg_name, options):
        """Expect {arg_name} to take on one of the values in {options}"""
        try:
            iter(options)
        except TypeError:
            raise SchemaError(
                f'{type(self).__qualname__}: \'options\' argument must be '
                f'iterable.  Got {type(options)}')
        def gen():
            return options
        GenNode.add_node(arg_name, gen)
        def pred():
            arg_val = self.get_arg(arg_name)
            if arg_val in options:
                return True, arg_val
            else:
                return False, NonOptionError(arg_name, arg_val) 
        PredNode.add_node(arg_name, pred)

    def arg_unchecked(self, arg_name):
        """
        Declare {arg_name} to be an argument unchecked by OpCheck 
        """
        self.p.set_arg_type(arg_name, None)

    def tensor_valid_dtypes(self, tensor_name, type_list):
        """
        Declare that {tensor_name} can have any of the dtype strings in
        {type_list}.  Names in {type_list} are converted via
        tf.dtypes.as_dtype(name).  e.g. names like 'int32', 'int64', 'float32'
        """
        arg_type = self.p.get_arg_type(tensor_name)
        if arg_type != tf.Tensor:
            raise SchemaError(
                f'{type(self).__name__}: Parameter \'{tensor_name}\' is '
                f'registered as type {arg_type}.  Can only call on tf.Tensor')
        if tensor_name in self.p.dtype_valid:
            raise SchemaError(
                f'{self.__qualname__}: Tensor \'{tensor_name}\' is already '
                f'registered with valid dtypes')

        dtypes = []
        for t in type_list:
            try:
                dt = tf.dtypes.as_dtype(t)
                dtypes.append(dt)
            except TypeError:
                raise SchemaError(
                    f'{type(self).__qualname__}: Type string \'{t}\' is not '
                    f'a valid tf.dtype representation')
        self.p.dtype_valid[tensor_name] = tuple(dtypes)

    def tensor_equate_dtypes(self, trg_tensor, src_tensor):
        """
        Declare that {trg_tensor} have the same dtype as {src_tensor}.
        Must first call arg_valid_dtypes(src_tensor, ...).
        trg_tensor must not be called in arg_valid_dtypes if it is called
        here.
        """
        src_type = self.p.get_arg_type(src_tensor)
        trg_type = self.p.get_arg_type(trg_tensor)
        if src_type != tf.Tensor or trg_type != tf.Tensor:
            raise SchemaError(
                f'{type(self).__name__}: Can only be called on two tensors. '
                f'Parameters \'{src_tensor}\' and \'{trg_tensor}\' are types '
                f'{src_type} and {trg_type}')
        if src_tensor not in self.p.dtype_valid:
            raise SchemaError(
                f'{self.__qualname__}: Must first register valid types for '
                f'src_tensor (\'{src_tensor}\'')
        if trg_tensor in (*self.p.dtype_valid, *self.p.dtype_equiv):
            raise SchemaError(
                f'{self.__qualname__}: trg_tensor (\'{trg_tensor}\') '
                f'already has a dtype constraint')
        self.p.dtype_equiv[trg_tensor] = src_tensor

    def add_input_sigrank(self, arg_name, signature, lo, hi):
        """
        Expect {arg_name} to be a list of length rank({signature}), with
        elements in [{lo}, {hi}]
        """
        def gen(ranks):
            rank = sum(ranks[s] for s in signature)
            val = [randint(lo, hi+1) for _ in range(rank)]
            return [val]
        GenNode.add_node(arg_name, gen, IName.RANKS)
        def pred(ranks):
            arg_val = self.get_arg(arg_name)
            rank = sum(ranks[s] for s in signature)
            if len(arg_val) != rank:
                return False, SigRankError(arg_name, rank, len(arg_val))
            else:
                return True, arg_val
        PredNode.add_node(arg_name, pred, IName.RANKS)

    def limit_ranks(self, sig, min_val, max_val):
        """
        Declare that the valid ranks of {sig} lie in the interval [{min_val},
        {max_val}]
        """
        self.p.check_sig(sig, 'rank limits')
        self.p.add_rank_limits(sig, min_val, max_val)

    def equate_ranks(self, target_sig, source_sig):
        """
        Declare that the rank of {target_sig} be equal to {source_sig}.
        It is required that all indices in {source_sig} appear in some
        signature in a limit_ranks call.
        """
        self.p.check_sig(target_sig, 'equate ranks')
        self.p.check_sig(source_sig, 'equate ranks')
        self.p.equate_ranks(target_sig, source_sig)

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

