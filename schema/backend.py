import tensorflow as tf
import itertools
import numpy as np
from collections import OrderedDict, defaultdict
from functools import partial
from .error import *
from . import fgraph
from .fgraph import GenNode, PredNode
from . import util

class IName(object):
    DIMS = '__dims'
    DTYPES = '__dtypes'
    RANKS = '__ranks'
    SIGS = '__sigs'
    RETURN = '__return'

class Backend(object):

    def __init__(self, op_path):
        self.op_path = op_path

        # framework op parameter names 
        self.param_names = None

        # arguments given to the op.  these change for each call
        self.arguments = None

        # returns, set after the framework operation returns.  changes with
        # each call 
        self.returns = []

        # map of single-letter index name to index description string
        # For example:  'i' => 'input spatial'
        self.index = OrderedDict()

        # Constraints on the allowed combinations of ranks 
        # Possible TODO: rewrite these to use actual index names rather than
        # index positions
        self.rank_maxs = defaultdict(lambda: 10000)
        self.rank_mins = defaultdict(lambda: 0)
        self.rank_equiv = {}  # 

        # sig => rank_func, where rank_func(op) provides the rank at op runtime 
        self.sig_ranks = {}

        # map of arg_name => type
        self.arg_types = {}

        # arg_name => sig_func, call_args
        # stores both inputs and returns
        # sig_func(*call_vals) returns sig
        self.arg_sigs = {}

        # arg_name => dims_func, call_args
        # dims_func(*call_vals) returns dims
        self.arg_dims = {}

        # idx => (dims_func, call_args)
        # dims_func(dims_map, *call_args) defines dims for idx
        self.dims_from_dims = {}

        # idx => (dims_func, call_args)
        # dims_func(rank_map, *call_args) defines dims for idx
        self.dims_from_rank = {}

        # Declare a tensor to have a set of valid dtypes, or assign its dtype
        # equivalent to a source tensor
        self.dtype_valid = {}  # ten_name => (tf.int32, ...)
        self.dtype_equiv = {}  # trg_ten_name => src_ten_name 

        # The actual op wrapped by OpCheck
        self.wrapped_op = None

        # Errors
        self.input_status = None
        self.framework_status = None
        self.return_status = None

    def __repr__(self):
        rep = 'Index: \n'
        rep += '\n'.join(idx + ': ' + desc for idx, desc in self.index.items())
        rep += '\n\nShape Arguments: \n'
        rep += '\n'.join(f'{n}: {sig}' for n, sig in self.arg_sigs.items())
        rep += '\n\nDType Constraints: \n'
        rep += '\n'.join(f'{n}: {dt}' for n, dt in self.dtype_valid.items())
        rep += '\n\nDType Equivalences: \n'
        rep += '\n'.join(f'{trg}: {src}' for trg, src in
                self.dtype_equiv.items())
        # rep += '\n\nRank Arguments: \n'
        # rep += '\n'.join(repr(arg) for arg in self.arg_rank.values())
        # rep += '\n\nReturn signatures: \n'
        # rep += '\n'.join(repr(sig) for sig in self.return_sigs)
        return rep

    def init_schema(self, op, func_sig, init_schema_func):
        self.param_names = func_sig.parameters.keys()
        self.build_pred_graph()
        self.build_gen_graph()
        init_schema_func(op)
        self.finalize_graphs()
        self.validate_constraints()

        # print('\n\n')
        # print('Operation: ', op.p.op_path)
        # for k, val in GenNode.registry.items():
        #     print(val)

    def clear_call_state(self):
        """Clear the data members which hold data from a framework call"""
        self.returns.clear()
        self.input_status = None
        self.framework_status = None
        self.return_status = None

    # fails if any letters in signature don't exist in self.index
    def check_sig(self, signature, name):
        if any(s not in self.index.keys() for s in signature):
            raise SchemaError(
                f'Signature "{signature}" associated with \'{name}\' '
                f'contains one or more unregistered indices. '
                f'Current known indices are: '
                f"{','.join(self.index.keys())}"
                f'Call OpSchema.add_index with the missing index.')

    def check_arg_added(self, arg_name, domain):
        """Check whether {arg_name} already added to {domain}"""
        if arg_name in domain:
            raise SchemaError(
                f'{type(self).__name__} was previously called with {arg_name}.'
                f'Can only call once per argument')

    def num_returns(self):
        """
        Report the current number of registered return values
        """
        is_return = lambda k: k.startswith(IName.RETURN)
        return len(list(filter(is_return, self.arg_sigs.keys())))

    def get_return_name(self, idx):
        return f'{IName.RETURN}:{idx}'

    def get_return_index(self, name):
        return int(name.split(':')[1]) 

    def get_index_dims(self):
        """
        Gets the currently inferred index dims map
        """
        node = next((n for n in self.input_pred_graph if n.name == IName.DIMS),
                None)
        return node.get_cached_value()

    def sig_indices(self, sig):
        inds = list(self.index.keys())
        return tuple(inds.index(idx) for idx in sig)

    def add_rank_limits(self, sig, min_val, max_val):
        if min_val is not None:
            prev_min_val = self.rank_mins[sig]
            self.rank_mins[sig] = max(prev_min_val, min_val)
        if max_val is not None:
            prev_max_val = self.rank_maxs[sig]
            self.rank_maxs[sig] = min(prev_max_val, max_val)

    def equate_ranks(self, target_sig, source_sig):
        # enforce target_sig and source_sig to have equal rank
        self.rank_equiv[target_sig] = source_sig

    def check_arg_name(self, arg_name):
        """Ensure {arg_name} is a valid argument name"""
        if arg_name not in self.param_names:
            raise SchemaError(
                f'\'{arg_name}\' not a known parameter. '
                f'Known parameters are: {self.param_names}')

    def check_arg_type(self, arg_name, expected_type):
        """Check that {arg_name} is registered as having {expected_type}""" 
        arg_type = self.get_arg_type(arg_name)
        if arg_type != expected_type:
            raise SchemaError(
                f'{type(self).__name__}: Parameter \'{arg_name}\' is '
                f'registered as type {expected_type}\'')

    def get_arg(self, arg_name, default=None):
        """Retrieve the value of {arg_name} argument at call-time."""
        self.check_arg_name(arg_name)
        return self.arguments[arg_name]

    def valid_arg_type(self, arg_name):
        """Type check argument at call-time"""
        val = self.get_arg(arg_name)
        expected_type = self.arg_types[arg_name]
        if expected_type is None:
            return True
        else:
            return isinstance(val, expected_type)

    def set_arg_type(self, arg_name, arg_type):
        """Expect {arg_name} to have type {arg_type}"""
        if arg_name not in self.param_names:
            raise SchemaError(
                f'{type(self).__name__}: Attempted to add {arg_name} parameter '
                f'but it is not found in the framework op parameters. '
                f'Valid parameters are: {self.param_names}')
        if arg_name in self.arg_types:
            if self.arg_types[arg_name] != arg_type:
                raise SchemaError(
                    f'{type(self).__name__}: Attempted to add {arg_name} as type '
                    f'{arg_type} to the registry, but it is already registered '
                    f'as type {self.arg_types[arg_name]}')
        self.arg_types[arg_name] = arg_type

    def get_arg_type(self, arg_name):
        """Retrieve the type(s) expected for {arg_name}"""
        self.check_arg_name(arg_name)
        if arg_name not in self.arg_types:
            raise SchemaError(
                f'Argument \'{arg_name}\' does not have a registered type. ')
        return self.arg_types[arg_name]

    def get_return(self, idx):
        try:
            return self.returns[idx]
        except IndexError:
            raise SchemaError(
                f'{type(self).__qualname__}({idx}) called but only '
                f'{len(self.returns)} returns')

    def validate_constraints(self):
        """
        Called at the end of schema construction to check that schema
        constraints are self-consistent 
        """
        # Ensure that every tensor has exactly one dtype constraint
        for arg_name, arg_type in self.arg_types.items():
            if arg_type != tf.Tensor:
                continue
            if arg_name in (*self.dtype_valid, *self.dtype_equiv):
                continue
            raise SchemaError(
                f'{type(self).__qualname__}: Error defining '
                f'\'{self.op_path}\' schema.  Tensor argument '
                f'\'{arg_name}\' has no registered dtype constraint.\n'
                f'Call tensor_equate_dtypes or tensor_valid_dtypes '
                f'for this tensor.')

        # add upper-bounds constraints for equated ranks
        for trg_sig, src_sig in self.rank_equiv.items():
            limits = {}
            rank_maxs = self.rank_maxs.items()
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
            self.rank_maxs[trg_sig] = sum(limits.values())

    def prepare_call(self, op, bound_args):
        """Call during the framework call phase"""
        self.clear_call_state()
        self.arguments = bound_args

    def log_framework_status(self, err):
        self.framework_status = FrameworkError(err)

    def src_dtype(self, tensor_name):
        """Return the source tensor name which determines the dtype of
        {tensor_name}"""
        if tensor_name in self.dtype_valid:
            return tensor_name
        src = self.dtype_equiv.get(tensor_name, None)
        if src is None:
            raise SchemaError(f'\'{tensor_name}\' has no determined dtype')
        return src

    def build_pred_graph(self):
        """
        Build the predicate graph for validating all inputs
        """
        def wrap(func):
            return partial(func, self)

        PredNode.clear_registry()
        types = PredNode.add_node('__types', wrap(valid_types))
        dtypes = PredNode.add_node(IName.DTYPES, wrap(valid_dtypes))
        sigs = PredNode.add_node(IName.SIGS, wrap(valid_sigs))
        ranks = PredNode.add_node(IName.RANKS, wrap(valid_ranks), IName.SIGS)
        dims = PredNode.add_node(IName.DIMS, wrap(valid_dims), IName.RANKS,
                IName.SIGS)
        ranks.add_predicate_parent(types)
        dtypes.add_predicate_parent(types)

    def finalize_graphs(self):
        """
        Call this after the user-supplied schema function is executed.
        The Schema API will create more PredNodes and GenNodes on the registry,
        and some will be roots.
        """
        # add parents for DIMS nodes
        pred_dims_node = PredNode.get_node(IName.DIMS)
        for dep in valid_dims_parents(self):
            pred_dims_node.maybe_append_parent(dep)

        gen_dims_node = GenNode.get_node(IName.DIMS)
        for dep in generate_dims_parents(self):
            gen_dims_node.maybe_append_parent(dep)

        # add parents for SIGS nodes
        pred_sigs_node = PredNode.get_node(IName.SIGS)
        for arg_name in sigs_parents(self):
            pred_sigs_node.maybe_append_parent(arg_name)
        
        gen_sigs_node = GenNode.get_node(IName.SIGS)
        for arg_name in sigs_parents(self):
            gen_sigs_node.maybe_append_parent(arg_name)

        # create the graphs
        pred_nodes = PredNode.get_ordered_nodes()
        self.input_pred_graph = [n for n in pred_nodes if not
                n.name.startswith(IName.RETURN)]
        self.return_pred_graph = [n for n in pred_nodes if
                n.name.startswith(IName.RETURN) or n.name == IName.DIMS]
        self.gen_graph = GenNode.get_ordered_nodes()

    def check_args(self):
        """
        The main function to check all input arguments for all constraints
        registered on the schema
        """
        error = fgraph.pred_graph_evaluate(self.input_pred_graph)
        self.input_status = Success() if error is None else error

    def check_return(self, op_return):
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

    def calc_rank_constraints(self):
        rmins = { self.sig_indices(sig): rank for sig, rank in
            self.rank_mins.items() }
        rmaxs = { self.sig_indices(sig): rank for sig, rank in
                self.rank_maxs.items() }
        req = { self.sig_indices(sig1): self.sig_indices(sig2) for sig1, sig2
            in self.rank_equiv.items() }
        return rmins, rmaxs, req

    def generate_ranks(self):
        """
        Generate all allowed rank combinations.  Generates a list of maps.
        Each map has index => rank for each index in self.index
        """
        rmins, rmaxs, req = self.calc_rank_constraints()

        k = len(self.index)
        index_order = list(self.index.keys())
        gen = util.feasible_region(k, rmins, rmaxs, req, {})
        rank_list = list(gen)
        return [dict(zip(index_order, ranks)) for ranks in rank_list]
        
    def generate_dtypes(self):
        """
        Generate all allowed dtype combinations.  Generates a list of maps.
        Each map has a full tensor_name => dtype for each input tensor
        """
        # src_ten are tensor names which have independent dtypes
        src_ten, allowed_dtypes = zip(*self.dtype_valid.items())
        # tensor_name => index 
        equiv_map = { trg: src_ten.index(src) for trg, src in
                self.dtype_equiv.items() }
        equiv_map.update({v: i for i, v in enumerate(src_ten)})

        combos = []
        for combo in itertools.product(*allowed_dtypes):
            el = { name: combo[ind] for name,ind in equiv_map.items() }
            combos.append(el)
        return combos

    def build_gen_graph(self):
        # Reserved names for the nodes, so they don't conflict with argument
        # names of ops
        GenNode.clear_registry()
        GenNode.add_node(IName.RANKS, lambda: self.generate_ranks())
        GenNode.add_node(IName.DTYPES, lambda: self.generate_dtypes())

        def sigs_wrap(**kwargs):
            return generate_sigs(self, **kwargs)
        GenNode.add_node(IName.SIGS, sigs_wrap)

        def dims_wrap(ranks, sigs_map, **kwargs):
            return generate_dims(self, ranks, sigs_map, **kwargs)
        GenNode.add_node(IName.DIMS, dims_wrap, IName.RANKS, IName.SIGS)

    def passed(self):
        return (
                self.input_status == Success() and
                self.framework_status == Success() and
                self.return_status == Success())

    def validation_report(self, config):
        err = ''
        if self.passed():
            err += 'None'
        else:
            err += f'Input Error: {self.input_status.message(self)}\n'
            err += f'Framework Error: {self.framework_status.message(self)}\n'
            err += f'Returns Error: {self.return_status.message(self)}\n'
        msg = ''
        dims = ', '.join(f'{k}:{v}' for k,v in config[IName.DIMS].items())
        msg += f'\nIndexes: {dims}'
        dtypes = ', '.join(f'{k}:{v.name}' for k,v in config[IName.DTYPES].items()) 
        msg += f'\nDTypes: ({dtypes})' 
        # sigs = ', '.join(f'{n}: {sig}' for n, sig in self.arg_sigs.items())
        # msg += f'\nShape Signatures: {sigs}'
        for arg_name, arg_val in config.items():
            if isinstance(arg_val, tf.Tensor):
                msg += f'\n{arg_name} shape: {arg_val.shape.as_list()}'
            elif arg_name.startswith('__'):
                continue
            else:
                msg += f'\n{arg_name}: {arg_val}'
        return msg, err

    def validate_schema(self):
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
            arg_dict = { p: config.get(p, None) for p in self.param_names }
            # self.wrapped_op(**arg_dict)
            # print(arg_dict.keys())
            try:
                self.wrapped_op(**arg_dict)
            except Exception as e:
                if isinstance(e, SchemaError):
                    raise e
                else:
                    print('in validate_schema, got exception: ', e)

            msg, err = self.validation_report(config)
            print(f'{msg}\n{err}')

    # produce ['i1', 'i2', 'i3'] from 'i' for example
    def index_list(self, idx):
        rank = self.index_rank[idx]
        if rank == 1:
            return [idx]
        else:
            return [idx + str(i) for i in range(1, rank + 1)]

    # produce ['b', 'i1', 'i2', 'i3', 'k'] from 'bik' for example
    def sig_list(self, sig):
        return [ind for s in sig for ind in self.index_list(s)]

    # produce [1,4] from letter='i', sig='bik' (assuming rank 3 for i) for
    # example
    def sig_range(self, idx, sig):
        ind = sig.index(idx)
        start = sum(self.index_ranks[l] for l in sig[:ind])
        rank = self.index_ranks[idx]
        return [start, start + rank]

    def report(self):
        """
        msg = ''
        err = self.arg_check_error
        if isinstance(err, ShapeError):
            msg += self.print_indices()
            msg += '\n\n'
            msg += self.print_inputs(err.index_letter)
        elif isinstance(err, NoMatchingRanks):
            msg += err.message(self)
        if msg != '':
            print(msg)
        # return msg
        """

def calc_sig_range(rank_map, idx, sig):
    ind = sig.index(idx)
    start = sum(rank_map[l] for l in sig[:ind])
    rank = rank_map[idx] 
    return [start, start + rank]

def calc_sig_dims(dims_map, sig):
    return [d for s in sig for d in dims_map[s]]

def valid_types(op):
    """
    Check that every argument is of allowed type
    """
    valid = True
    for arg_name, expected_types in op.arg_types.items():
        arg = op.get_arg(arg_name)
        if expected_types is None:
            continue # ignored
        if not isinstance(arg, expected_types):
            return False, ArgTypeError(arg_name)
    return valid, None

def valid_dtypes(op):
    """Check that all tensor arguments have valid dtypes"""
    for ten_name, dtypes in op.dtype_valid.items():
        arg = op.get_arg(ten_name)
        assert(isinstance(arg, tf.Tensor))
        if arg.dtype not in dtypes:
            return False, DTypeNotAllowed(ten_name, arg.dtype)
    for trg_name, src_name in op.dtype_equiv.items():
        src = op.get_arg(src_name)
        trg = op.get_arg(trg_name)
        if trg.dtype != src.dtype:
            return False, DTypeNotEqual(src_name, trg_name)
    return True, None

def input_arg_dims_parents(op):
    """
    Return all dependencies of the shape-related arguments (excluding return
    tensors)
    """
    deps = set()
    for arg_name, info in op.arg_dims.items():
        if arg_name.startswith(IName.RETURN):
            continue
        dims_func, call_names = info
        deps.update(call_names)
    return deps

def calc_input_arg_dims(op, kwargs):
    """
    calculate the dims for all shape-related arguments (excluding return
    tensors).  Returns a map with arg_name => dims
    """
    arg_dims_map = {}
    for arg_name, info in op.arg_dims.items():
        if arg_name.startswith(IName.RETURN):
            continue
        dims_func, call_names = info
        call_args = { n:kwargs[n] for n in call_names }
        dims = dims_func(**call_args)
        arg_dims_map[arg_name] = dims
    return arg_dims_map

def valid_sigs(op, **kwargs):
    """
    Dummy validation (always succeeds) to provide the sigs_map (arg_name =>
    sig) to the RANKS node in the pred_graph
    """
    sigs_map = generate_sigs(op, **kwargs)
    return True, sigs_map[0]

def valid_ranks(op, sigs_map, **kwargs):
    """
    Using the rank allowed combinations and rank inference constraints,
    resolve the ranks to a single combination
    """
    const_map = {}
    for sig, rank_func in op.sig_ranks.items():
        sig_inds = op.sig_indices(sig)
        const_map[sig_inds] = rank_func(op)

    # augment the const_map using ranks of shape inputs
    arg_dims_map = calc_input_arg_dims(op, kwargs)
    for arg_name, dims in arg_dims_map.items():
        sig = sigs_map[arg_name]
        sig_inds = op.sig_indices(sig)
        const_map[sig_inds] = len(dims)

    k = len(op.index)
    rmins, rmaxs, req = op.calc_rank_constraints()
    rank_list = list(util.feasible_region(k, rmins, rmaxs, req, const_map))

    if len(rank_list) == 0:
        return False, NoMatchingRanks()
    elif len(rank_list) > 1:
        return False, AmbiguousRanks()
    else:
        index_ranks = dict(zip(op.index.keys(), rank_list[0]))
        return True, index_ranks

def valid_dims(op, rank_map, sigs_map, **kwargs):
    """
    Infer index dims from Tensor shapes, and shape-parameter values,
    signatures and inferred index ranks.

    rank_map: idx => rank
    sigs_map: arg_name => sig
    kwargs: arguments to be passed to the dims_from_dims functions
    """
    arg_dims_map = calc_input_arg_dims(op, kwargs)
    input_inds = { idx for n in arg_dims_map.keys() for idx in sigs_map[n] }
    # print('input_inds: ', input_inds)
    index_dims = {}
    for idx in input_inds:
        # find all usages of idx
        idx_shapes = set()
        for arg_name, dims in arg_dims_map.items():
            sig = sigs_map[arg_name]
            if idx not in sig:
                continue
            sub_range = calc_sig_range(rank_map, idx, sig)
            idx_shape = dims[slice(*sub_range)]
            idx_shapes.add(tuple(idx_shape))

        if len(idx_shapes) != 1:
            return False, IndexUsageError(idx)
        else:
            index_dims[idx] = idx_shapes.pop()

    # Calculate derived dimensions
    ddims_map = calc_dims_dep_dims(op, index_dims, kwargs)
    index_dims.update(ddims_map)

    return True, index_dims

def valid_dims_parents(op):
    deps1 = input_arg_dims_parents(op)
    deps2 = dims_from_dims_parents(op)
    return deps1.union(deps2)

def valid_return_dims(op, dims_map, out_index, **kwargs):
    """
    Validate tensor output shape at {out_index}.  kwargs are inspected to find
    the subset needed 

    """
    if len(op.returns) != op.num_returns():
        return False, OutputNumberMismatch(len(op.returns))
    ten = op.returns[out_index]
    if not isinstance(ten, tf.Tensor):
        return False, ReturnTypeError(tf.Tensor, type(ten))
    rank_map = { i:len(d) for i,d in dims_map.items() }
    shape = ten.shape.as_list()
    ret_name = op.get_return_name(out_index)
    sig_func, arg_names = op.arg_sigs[ret_name]
    call_args = { arg: kwargs[arg] for arg in arg_names }
    sig = sig_func(**call_args)
    for idx in sig:
        sub_range = calc_sig_range(rank_map, idx, sig)
        idx_shape = tuple(shape[slice(*sub_range)])
        expected_idx_shape = dims_map[idx]
        if idx_shape != expected_idx_shape:
            return False, ShapeError(ret_name, idx, idx_shape)
    return True, None


def pack_dims_map(rank_map, index_list, dims_list):
    """
    Computes a dims_map (idx => dims list) from the inputs.
    rank_map: idx => rank
    index_list: list of idx corresponding to the order of dims_list
    dims_list: flat dimensions.
    """
    dims_map = {}
    offset = 0
    for i, idx in enumerate(index_list):
        rank = rank_map[idx] 
        dims_map[idx] = dims_list[offset:offset+rank]
        offset += rank
    return dims_map 

def calc_rank_dep_dims(op, rank_map, kwargs):
    """
    Computes all rank-dependent dims registered with the op, and
    returns them as a dims_map of: idx => dims_list
    """
    dims_map = {}
    for idx, info in op.dims_from_rank.items():
        func, arg_names = info
        call_args = { n:kwargs[n] for n in arg_names }
        dims = func(rank_map, **call_args)
        if not util.is_iterable(dims):
            raise SchemaError(
                f'dims_from_rank \'{func}\' from '
                f'schema \'{op.op_path}\' returned a non-iterable.')
        dims_map[idx] = dims
    return dims_map

def rank_from_dims_parents(op):
    """
    Return all dependencies of the rank-dependent dims calculations
    """
    return { arg for _,l in op.dims_from_rank.values() for arg in l }

def calc_dims_dep_dims(op, dims_map, kwargs):
    """
    Compute all dims-dependent dims registered in the op
    """
    calc_dims_map = {}
    for idx, info in op.dims_from_dims.items():
        func, arg_names = info
        call_args = { n:kwargs[n] for n in arg_names }
        dims = func(dims_map, **call_args)
        calc_dims_map[idx] = dims
    return calc_dims_map

def dims_from_dims_parents(op):
    """
    Return all dependencies of the dims-dependent dims calculations
    """
    return { arg for _,l in op.dims_from_dims.values() for arg in l }

def generate_sigs(op, **kwargs):
    """
    Generates the arg_name => sig map for all input arguments.  Assumes all
    keys of kwargs have a GenNode by that name.
    """
    sig_map = {}
    for arg_name, info in op.arg_sigs.items():
        sig_func, call_names = info
        call_args = { n:kwargs[n] for n in call_names }
        sig = sig_func(**call_args)
        sig_map[arg_name] = sig
    return [sig_map]

def sigs_parents(op):
    """
    Get the set of all parent arguments for the sigs
    """
    sig_args = set()
    for arg_name, info in op.arg_sigs.items():
        _, call_names = info
        sig_args.update(call_names)
    return sig_args

def generate_dims(op, ranks, sigs_map, **kwargs):
    """
    Generates all remaining dims for indexes, such that the total number of
    elements of all tensors is between certain limits.  {ranks} is a map of
    idx => rank.  {kwargs} is the union of all arguments needed for the
    op.dims_from_dims functions.
    """
    def classify_indices(ranks):
        nonfree_inds = [ *op.dims_from_dims, *op.dims_from_rank ]
        free_inds = [ k for k in ranks.keys() if k not in nonfree_inds ]
        return free_inds, nonfree_inds

    def nelem(op, ranks, flat_dims):
        free_inds, nonfree_inds = classify_indices(ranks)
        rank_dims_map = calc_rank_dep_dims(op, ranks, kwargs)
        free_dims_map = pack_dims_map(ranks, free_inds, flat_dims)
        all_dims_map = { **rank_dims_map, **free_dims_map }
        dims_dep_map = calc_dims_dep_dims(op, all_dims_map, kwargs)
        all_dims_map.update(dims_dep_map)
        sum_nelem = 0
        for sig in sigs_map.values():
            shape = calc_sig_dims(all_dims_map, sig)
            sum_nelem += np.prod(shape)
        return sum_nelem

    nelem_wrap = partial(nelem, op, ranks)

    free_inds, nonfree_inds = classify_indices(ranks)
    k = sum(rank for idx, rank in ranks.items() if idx in free_inds)
    min_nelem = 100000
    max_nelem = 200000
    dims = util.bsearch_integers(k, min_nelem, max_nelem, nelem_wrap)
    dims_map = pack_dims_map(ranks, free_inds, dims)
    rank_dims_map = calc_rank_dep_dims(op, ranks, kwargs)
    dims_map.update(rank_dims_map)
    dims_dep_map = calc_dims_dep_dims(op, dims_map, kwargs)
    dims_map.update(dims_dep_map)
    return [dims_map]

def generate_dims_parents(op):
    deps1 = rank_from_dims_parents(op)
    deps2 = dims_from_dims_parents(op)
    return deps1.union(deps2)

def generate_tensor(name, signature, dims_map, dtype_map):
    shape = calc_sig_dims(dims_map, signature)
    ten_dtype = dtype_map[name]
    if ten_dtype.is_integer:
        ten = tf.random.uniform(shape, minval=-10, maxval=10,
                dtype=ten_dtype)
    else:
        ten = tf.random.normal(shape, dtype=ten_dtype)
    return [ten] 

def generate_shape(signature, dims):
    shape = calc_sig_dims(dims_map, signature)
    return [shape]

