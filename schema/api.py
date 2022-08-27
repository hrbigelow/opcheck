import tensorflow as tf
from . import predicates as pr
from . import generators as ge
from . import base
from .base import Kind
from . import fgraph
from .error import *
from .fgraph import PredNode as P, GenNode as G

"""
Every API call will mutate the Generative Graph and the Predicate Graph
logically in tandem.  It should maintain the invariants:

1. Every value set produced by the Generative Graph should be valid as judged
   by the Predicate Graph

2. The set of values produced by the Generative Graph is "complete" in the
sense that it it explores every combination of dtypes, ranks, and option
arguments.  It need not explore every possible setting of dimensions or tensor
contents.

Both finished graphs must have exactly one node corresponding to each input
argument, either the arg_name:tensor node (for tensor arguments) or the
arg_name:arg node for non-tensors.
"""

class SchemaApi(object):
    def __init__(self, op_path):
        self.op_path = op_path

        # idx => description
        self.index = {}
        # param_name => kname (the name of the GenNode producing the param value)
        self.params = {}
        self.gen_graph = None
        self.input_pred_graph = None
        self.return_pred_graph = None
        self.rank_cons = base.RankConstraints(self)
        self.dtype_cons = base.DTypeConstraints()
        self.comp_dims = base.CompDims()
        self.num_returns = 0

        # error status
        self.input_status = None
        self.framework_status = None
        self.return_status = None

        # call time values
        self.arguments = {}
        self.returns = []

    def _init_schema(self, func_sig, init_schema_func):
        self.params = { k: None for k in func_sig.parameters.keys() }
        self._init_pred_graph()
        self._init_gen_graph()
        init_schema_func(self)
        self._add_pred_graph()
        self._add_gen_graph()
        self._validate_constraints()

    def _prepare_call(self, bound_args):
        """Call during the framework call phase"""
        self.arguments = bound_args
        self.returns.clear()
        self.input_status = None
        self.framework_status = None
        self.return_status = None

    def _check_args(self):
        """
        The main function to check all input arguments for all constraints
        registered on the schema
        """
        error = fgraph.pred_graph_evaluate(self.input_pred_graph)
        self.input_status = Success() if error is None else error

    def _check_return(self, op_return):
        """
        Check the return tensors' shapes and types against those predicted by
        the framework
        """
        if not isinstance(self.input_status, Success):
            self.return_status = NotApplicable()
            return

        if not isinstance(op_return, (list, tuple)):
            op_return = (op_return,)
        self.returns = list(op_return)
        error = fgraph.pred_graph_evaluate(self.return_pred_graph)
        self.return_status = Success() if error is None else error

    def _log_framework_status(self, err):
        self.framework_status = FrameworkError(err)

    def _validate_schema(self):
        """
        Generate a set of input argument value configurations for the op, and
        run the op on each configuration.  The set includes all possible
        combinations of valid index ranks, input tensor dtypes, and settings
        for parameters that are not declared with arg_unchecked in the schema.
        Also generates some invalid configurations.  Checks that opcheck will
        pass the valid ones, and log the appropriate errors for the invalid
        ones. 

        It can be thought of as a systematic 'sampling procedure' from a
        generative model with a set of hidden variables (index dims and ranks,
        and others) and observed variables (values of arguments)
        """
        for config in fgraph.gen_graph_iterate(self.gen_graph):
            # extract the values from the argument nodes of the graph
            # print(vals)
            arg_dict = { p: config.get(k, None) for p,k in self.params.items() }
            # self.wrapped_op(**arg_dict)
            # print(arg_dict.keys())
            try:
                self.wrapped_op(**arg_dict)
            except Exception as e:
                if isinstance(e, SchemaError):
                    raise e
                else:
                    print('in validate_schema, got exception: ', e)

            msg, err = self._validation_report(config)
            print(f'{msg}\n{err}')

    def _passed(self):
        return (
                isinstance(self.input_status, Success) and
                isinstance(self.framework_status, Success) and
                isinstance(self.return_status, Success)
                )

    def _validation_report(self, config):
        err = ''
        if self._passed():
            err += 'None'
        else:
            err += f'Input Error: {self.input_status.message(self)}\n'
            err += f'Framework Error: {self.framework_status.message(self)}\n'
            err += f'Returns Error: {self.return_status.message(self)}\n'
        msg = ''
        dims = ', '.join(f'{k}:{v}' for k,v in config[Kind.DIMS].items())
        msg += f'\nIndexes: {dims}'
        dtypes = ', '.join(f'{k}:{v.name}' for k,v in config[Kind.DTYPES].items()) 
        msg += f'\nDTypes: ({dtypes})' 
        # sigs = ', '.join(f'{n}: {sig}' for n, sig in self.arg_sigs.items())
        # msg += f'\nShape Signatures: {sigs}'
        for kname, arg_val in config.items():
            if isinstance(arg_val, tf.Tensor):
                msg += f'\n{kname} shape: {arg_val.shape.as_list()}'
            elif base.pfx(kname) == '':
                continue
            else:
                msg += f'\n{kname}: {arg_val}'
        return msg, err

    def _report(self):
        print('Validation Report')

    def _sig_inds(self, sig):
        inds = list(self.index.keys())
        return tuple(inds.index(idx) for idx in sig)

    def _check_arg_name(self, arg_name):
        """Ensure {arg_name} is a valid argument name"""
        if arg_name not in self.params:
            raise SchemaError(
                f'\'{arg_name}\' not a known parameter. '
                f'Known parameters are: {self.params.keys()}')

    def _get_arg(self, arg_name, default=None):
        """Retrieve the value of {arg_name} argument at call-time."""
        self._check_arg_name(arg_name)
        return self.arguments[arg_name]

    def _get_arg_kind(self, arg_name):
        """Retrieve the type expected for {arg_name}"""
        self._check_arg_name(arg_name)
        kname = self.params[arg_name]
        return base.kind(kname)

    def _set_arg_kname(self, arg_name, arg_kname):
        """
        Expect {arg_name} to have type {arg_kname}
        """
        if arg_name not in self.params:
            raise SchemaError(
                f'{type(self).__name__}: Attempted to add {arg_name} parameter '
                f'but it is not found in the framework op parameters. '
                f'Valid parameters are: {self.params.keys()}')
        
        if self.params[arg_name] is not None:
            raise SchemaError(
                f'{type(self).__name__}: Attempting to add {arg_name} as type '
                f'{arg_type} to the registry, but it is already registered '
                f'as type {self.params[arg_name].__name__}')
        self.params[arg_name] = arg_kname

    def _check_sig(self, signature, name):
        if any(s not in self.index.keys() for s in signature):
            raise SchemaError(
                f'Signature "{signature}" associated with \'{name}\' '
                f'contains one or more unregistered indices. '
                f'Current known indices are: '
                f"{','.join(self.index.keys())}"
                f'Call OpSchema.add_index with the missing index.')

    def _get_return(self, idx):
        try:
            return self.returns[idx]
        except IndexError:
            raise SchemaError(
                f'{type(self).__qualname__}({idx}) called but only '
                f'{len(self.returns)} returns')
    
    @staticmethod
    def _resolve_arg_names(caller, graph_cls, arg_names):
        # resolve each arg_name uniquely.  arg_name may be a distinct prefix
        # of a node in graph_cls, or a full name that uniquely identifies the
        # node
        knames = []
        for arg_name in arg_names:
            kname = P.find_unique_name(arg_name)
            if kname is None:
                raise SchemaError(
                    f'{type(caller).__qualname__}: argument name \'{arg_name}\''
                    f' didn\'t identify a unique node in the registry.')
            knames.append(kname)
        return tuple(knames)

    def _init_pred_graph(self):
        P.clear_registry()
        P.add_node(Kind.SCHEMA, lambda: (True, self))
        P.add_node(Kind.DTYPES, pr.ValidDTypes(self.dtype_cons))
        P.add_node(Kind.RANKS, pr.IndexRanks(self, self.rank_cons))
        P.add_node(Kind.IDIMS, pr.input_index_dims, Kind.RANKS)
        P.add_node(Kind.CDIMS, pr.ComputedDims(self.comp_dims), Kind.IDIMS)

    def _init_gen_graph(self):
        G.clear_registry()
        G.add_node(Kind.SCHEMA, lambda: [self])
        G.add_node(Kind.DTYPES, ge.GenDTypes(self.dtype_cons)) 
        G.add_node(Kind.RANKS, ge.GenRanks(self, self.rank_cons)) 
        G.add_node(Kind.DIMS, ge.GenIndexDims(self.comp_dims), Kind.RANKS)

    def _add_pred_graph(self):
        pred_nodes = P.get_ordered_nodes()
        def is_return(node):
            ret_knames = (Kind.RETURN_TENSOR, Kind.VALID_RETURN)
            return base.kind(node.name) in ret_knames 

        self.input_pred_graph = [n for n in pred_nodes if not is_return(n) ]
        self.return_pred_graph = [n for n in pred_nodes if is_return(n) ]

    def _add_gen_graph(self):
        self.gen_graph = G.get_ordered_nodes()

    def _validate_constraints(self):
        """
        Called at the end of schema construction to check that schema
        constraints are self-consistent 
        """
        # Ensure that every tensor has exactly one dtype constraint
        for arg_name, arg_kname in self.params.items():
            if arg_kname is None:
                continue
            if base.kind(arg_kname) != Kind.TENSOR:
                continue
            if arg_name in self.dtype_cons.all(): 
                continue
            raise SchemaError(
                f'{type(self).__qualname__}: Error defining '
                f'\'{self.op_path}\' schema.  Tensor argument '
                f'\'{arg_name}\' has no registered dtype constraint.\n'
                f'Call tensor_equate_dtypes or tensor_valid_dtypes '
                f'for this tensor.')

        # add upper-bounds constraints for equated ranks
        for trg_sig, src_sig in self.rank_cons.equiv.items():
            limits = {}
            rank_maxs = self.rank_cons.maxs.items()
            for idx in src_sig:
                pair = next(((s,r) for s,r in rank_maxs if idx in s), None)
                if pair is None:
                    raise SchemaError(
                        f'{type(self).__qualname__}: Target signature '
                        f'\'{trg_sig}\' was equated with source signature '
                        f'\'{src_sig}\', but index \'{idx}\' in source '
                        f'signature has no limits registered with '
                        f'add_rank_limits.  All indices in source signature '
                        f'must appear in at least one add_rank_limits call.')
                max_sig, max_rank = pair
                limits[max_sig] = max_rank
            self.rank_cons.maxs[trg_sig] = sum(limits.values())

    # ============ PUBLIC API ====================
    def add_index(self, idx, description):
        self.index[idx] = description

    def arg_rank(self, arg_name, sig):
        # self._set_arg_type(arg_name, int)
        self.rank_cons.add_arg_rank(arg_name, sig)
        arg_kname = base.kname(arg_name, Kind.ARG)
        self._set_arg_kname(arg_name, arg_kname)
        P.add_node(arg_kname, pr.ArgInt(arg_name), Kind.SCHEMA) 
        G.add_node(arg_kname, ge.Rank(sig), Kind.RANKS)
        rank_node = P.get_node(Kind.RANKS)
        rank_node.maybe_append_parent(arg_kname)
        rank_node.maybe_append_parent(Kind.SCHEMA)

    def rank_constraint(self, sig, rank_func, *arg_names):
        arg_knames = self._resolve_arg_names(self, P, arg_names)
        self.rank_cons.add_sig_func(sig, rank_func, arg_knames)
        rank_node = P.get_node(Kind.RANKS)
        for arg_kname in arg_knames:
            rank_node.maybe_append_parent(arg_kname)

    def arg_unchecked(self, arg_name):
        """
        Declare {arg_name} to be an argument unchecked by OpCheck 
        """
        pass

    """
    def constraint(self, constraint_name, pred_func, *pred_arg_names):
        cons_name = base.kname(constraint_name, Kind.CONS)
        P.add_node(cons_name, pred_func, *pred_arg_names) 
    """

    def computed_dims(self, index, comp_func, *comp_arg_names):
        """
        Register {comp_func} to compute the dimensions of {index}.
        Will be called as: comp_func(idims_map, *comp_arg_vals)
        """
        comp_knames = self._resolve_arg_names(self, P, comp_arg_names)
        self.comp_dims.add(index, comp_func, comp_knames)
        cdims_pnode = P.get_node(Kind.CDIMS)
        dims_gnode = G.get_node(Kind.DIMS)
        for kname in comp_knames:
            cdims_pnode.maybe_append_parent(kname)
            if kname != Kind.IDIMS:
                dims_gnode.maybe_append_parent(kname)

    def equate_ranks(self, target_sig, source_sig):
        self._check_sig(target_sig, 'equate ranks')
        self._check_sig(source_sig, 'equate ranks')
        self.rank_cons.equate_ranks(target_sig, source_sig)

    def limit_ranks(self, sig, min_val, max_val):
        self._check_sig(sig, 'rank limits')
        self.rank_cons.add_rank_limits(sig, min_val, max_val)

    def valid_dtypes(self, tensor_name, type_list):
        """
        Declare that {tensor_name} can have any of the dtype strings in
        {type_list}.  Names in {type_list} are converted via
        tf.dtypes.as_dtype(name).  e.g. names like 'int32', 'int64', 'float32'
        """
        arg_kind = self._get_arg_kind(tensor_name)
        if arg_kind != Kind.TENSOR:
            raise SchemaError(
                f'{type(self).__name__}: Parameter \'{tensor_name}\' is '
                f'registered as {arg_kind}.  Can only call on {Kind.TENSOR}')
        if tensor_name in self.dtype_cons.valid:
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
        self.dtype_cons.add_valid(tensor_name, dtypes)

    def equate_dtypes(self, trg_tensor, src_tensor):
        """
        Declare that {trg_tensor} have the same dtype as {src_tensor}.
        Must first call arg_valid_dtypes(src_tensor, ...).
        trg_tensor must not be called in arg_valid_dtypes if it is called
        here.
        """
        src_kind = self._get_arg_kind(src_tensor)
        trg_kind = self._get_arg_kind(trg_tensor)
        if src_kind != Kind.TENSOR or trg_kind != Kind.TENSOR:
            raise SchemaError(
                f'{type(self).__name__}: Can only be called on two tensors. '
                f'Parameters \'{src_tensor}\' and \'{trg_tensor}\' are types '
                f'{src_type} and {trg_type}')
        if src_tensor not in self.dtype_cons.valid:
            raise SchemaError(
                f'{self.__qualname__}: Must first register valid types for '
                f'src_tensor (\'{src_tensor}\'')
        if trg_tensor in self.dtype_cons.all():
            raise SchemaError(
                f'{self.__qualname__}: trg_tensor (\'{trg_tensor}\') '
                f'already has a dtype constraint')
        self.dtype_cons.add_equiv(trg_tensor, src_tensor)
        pass

    def arg_pseudo(self, pseudo_name, pred_func, gen_func, arg_name):
        arg_kname = base.kname(pseudo_name, Kind.PSEUDO)
        pfunc_obj = pr.ArgFunc(arg_name, pred_func)
        P.add_node(arg_kname, pfunc_obj, Kind.SCHEMA) 
        G.add_node(arg_kname, gen_func)

    def arg_func(self, arg_name, pred_func, gen_func, *func_arg_names):
        """
        """
        knames = self._resolve_arg_names(self, P, func_arg_names)
        arg_kname = base.kname(arg_name, Kind.ARG)
        self._set_arg_kname(arg_name, arg_kname)
        pfunc_obj = pr.ArgFunc(arg_name, pred_func)
        P.add_node(arg_kname, pfunc_obj, Kind.SCHEMA, *knames)
        G.add_node(arg_kname, gen_func, *knames)

    def arg_option(self, arg_name, options):
        """
        Expect {arg_name} to take on one of the values in {options}
        """
        try:
            iter(options)
        except TypeError:
            raise SchemaError(
                f'{type(self).__qualname__}: \'options\' argument must be '
                f'iterable.  Got {type(options)}')
        def gen():
            return options

        arg_kname = base.kname(arg_name, Kind.ARG)
        self._set_arg_kname(arg_name, arg_kname)
        G.add_node(arg_kname, gen)
        def pred():
            arg_val = self._get_arg(arg_name)
            if arg_val in options:
                return True, arg_val
            else:
                return False, NonOptionError(arg_name, arg_val) 
        pred_obj = pr.ArgFunc(arg_name, pred)
        P.add_node(arg_kname, pred_obj, Kind.SCHEMA)

    @staticmethod
    def _convert_str(call_func, func_or_str, arg_names):
        if isinstance(func_or_str, str):
            if len(arg_names) != 0:
                raise SchemaError(
                    f'{type(call_func).__qualname__}: A string-valued '
                    f'arg \'{func_or_str}\' cannot have arguments. '
                    f'Got arguments \'{arg_names}\'')
            return lambda: func_or_str 
        else:
            return func_or_str

    def arg_tensor(self, arg_name, sig_func, *sig_arg_names):
        sig_func = self._convert_str(self, sig_func, sig_arg_names)
        sig_arg_knames = self._resolve_arg_names(self, P, sig_arg_names)
        tensor_kname = base.kname(arg_name, Kind.TENSOR) 
        self._set_arg_kname(arg_name, tensor_kname)

        dtype_kname = base.kname(arg_name, Kind.DTYPE)
        shape_kname = base.kname(arg_name, Kind.SHAPE)
        sig_kname = base.kname(arg_name, Kind.SIG)
        self.rank_cons.add_shape_sig(arg_name)

        P.add_node(tensor_kname, pr.GetTensor(arg_name), Kind.SCHEMA)
        dtype_pnode = P.add_node(dtype_kname, pr.dtype, tensor_kname)
        shape_pnode = P.add_node(shape_kname, pr.tensor_shape, tensor_kname)
        sig_pnode = P.add_node(sig_kname, pr.Sig(sig_func), *sig_arg_knames)

        sig_gnode = G.add_node(sig_kname, ge.Sig(sig_func), *sig_arg_knames)
        G.add_node(tensor_kname, ge.GenTensor(arg_name), sig_kname,  Kind.DIMS,
                Kind.DTYPES) 
        dims_gnode = G.get_node(Kind.DIMS)
        dims_gnode.append_parent(sig_gnode)

        # Create edges
        dtypes_pnode = P.get_node(Kind.DTYPES)
        ranks_pnode = P.get_node(Kind.RANKS)
        idims_pnode = P.get_node(Kind.IDIMS)
        dtypes_pnode.append_parent(dtype_pnode)
        ranks_pnode.append_parent(shape_pnode)
        ranks_pnode.append_parent(sig_pnode)
        idims_pnode.append_parent(shape_pnode)
        idims_pnode.append_parent(sig_pnode)

    def arg_shape(self, arg_name, sig_func, *sig_arg_names):
        sig_func = self._convert_str(self, sig_func, sig_arg_names)
        sig_arg_knames = self._resolve_arg_names(self, P, sig_arg_names)
        # self._set_arg_type(arg_name, list)
        shape_kname = base.kname(arg_name, Kind.SHAPE)
        self._set_arg_kname(arg_name, shape_kname)
        sig_kname = base.kname(arg_name, Kind.SIG)
        P.add_node(shape_kname, pr.ArgShape(arg_name), Kind.SCHEMA)
        P.add_node(sig_kname, pr.Sig(sig_func), *sig_arg_knames)
        sig_node = G.add_node(sig_kname, ge.Sig(sig_func), *sig_arg_knames)
        G.add_node(shape_kname, ge.GenShape(), Kind.DIMS, sig_kname)
        dims_gnode = G.get_node(Kind.DIMS)
        dims_gnode.append_parent(sig_node)

    def arg_sigrank(self, arg_name, sig, lo, hi):
        arg_kname = base.kname(arg_name, Kind.ARG) 
        P.add_node(arg_kname, pr.SigRank(arg_name, sig), Kind.SCHEMA,
                Kind.RANKS)
        G.add_node(arg_kname, ge.GenSigRank(sig, lo, hi), Kind.RANKS)
        self._set_arg_kname(arg_name, arg_kname)
        
    def return_tensor(self, sig_func, *arg_names):
        sig_func = self._convert_str(self, sig_func, arg_names)
        arg_knames = self._resolve_arg_names(self, P, arg_names)
        index = self.num_returns
        ret_name = str(index)
        ret_ten_name = base.kname(ret_name, Kind.RETURN_TENSOR)
        valid_return = base.kname(ret_name, Kind.VALID_RETURN)
        pshape_name = base.kname(ret_name, Kind.PSHAPE)
        sig_name = base.kname(ret_name, Kind.SIG)
        P.add_node(ret_ten_name, pr.GetReturnTensor(index), Kind.SCHEMA)
        P.add_node(sig_name, pr.Sig(sig_func), *arg_knames)  
        P.add_node(pshape_name, pr.predicted_shape, Kind.IDIMS, Kind.CDIMS,
                sig_name)
        P.add_node(valid_return, pr.ValidReturnShape(index), ret_ten_name,
                pshape_name) 
        G.add_node(sig_name, ge.Sig(sig_func), *arg_knames)
        self.num_returns += 1

