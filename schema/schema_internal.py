import tensorflow as tf
import itertools
import numpy as np
from collections import OrderedDict, defaultdict
from functools import partial
from .error import *
from . import fgraph
from .fgraph import GenNode, PredNode
from . import util

class SchemaInternal(object):

    OP = '__op'
    DIMS = '__dims'
    DTYPES = '__dtypes'
    RANK = '__rank'
    RETURN = '__return'

    def __init__(self, op_path):
        self.op_path = op_path

        # framework op parameter names 
        self.parameter_names = None

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
        self.rank_equiv = []  # [(sig1, sig2), ...] sig1 and sig2 equal ranks

        # sig => rank_func, where rank_func(op) provides the rank at op runtime 
        self.sig_ranks = {}

        # map of arg_name => type
        self.arg_types = {}
        self.return_shapes = []  # Ordered (and named) Signatures for returns

        # sig => dims_func; dims_func(op) provides expected shape of sig
        self.sig_dims = {}

        # idx => (dims_func, call_args)
        # dims_func(dims_map, *call_args) defines dims for idx
        self.dims_from_dims = {}

        # idx => (dims_func, call_args)
        # dims_func(rank_map, *call_args) defines dims for idx
        self.dims_from_rank = {}

        # Declare a tensor to have a set of valid dtypes, or assign its dtype
        # equivalent to a source tensor
        self.dtype_valid = {}  # ten_name => (tf.int32, ...)
        self.dtype_equiv = []  # [(ten_name1, ten_name2), ...] tensors must have
                               # equal dtypes

        self.calltime_config = None

        # The actual op wrapped by OpCheck
        self.wrapped_op = None

        # Errors
        self.opcheck_input_error = None
        self.framework_error = None
        self.opcheck_return_error = None

    def __repr__(self):
        rep = 'Index: \n'
        rep += '\n'.join(idx + ': ' + desc for idx, desc in self.index.items())
        # rep += '\n\nShape Arguments: \n'
        # rep += '\n'.join(repr(arg) for arg in self.arg_shape.values())
        # rep += '\n\nRank Arguments: \n'
        # rep += '\n'.join(repr(arg) for arg in self.arg_rank.values())
        rep += '\n\nReturn signatures: \n'
        rep += '\n'.join(repr(sig) for sig in self.return_shapes)
        return rep

    def init_schema(self, op, func_sig, init_schema_func, calltime_config_func):
        self.parameter_names = func_sig.parameters.keys()
        self.build_pred_graph()
        self.build_gen_graph()
        init_schema_func(op)
        self.finalize_graphs()

        # print('\n\n')
        # print('Operation: ', op.p.op_path)
        # for k, val in GenNode.registry.items():
        #     print(val)
        
        if calltime_config_func is None:
            calltime_config_func = lambda op: None
        self.calltime_config = calltime_config_func

    def clear_call_state(self):
        """Clear the data members which hold data from a framework call"""
        self.returns.clear()
        self.opcheck_input_error = None
        self.framework_error = None
        self.opcheck_return_error = None

    # fails if any letters in signature don't exist in self.index
    def check_sig(self, signature, name):
        if any(s not in self.index.keys() for s in signature):
            raise RuntimeError(
                f'Signature "{signature}" associated with \'{name}\' '
                f'contains one or more unregistered indices. '
                f'Current known indices are: '
                f"{','.join(self.index.keys())}"
                f'Call OpSchema.add_index with the missing index.')

    def check_arg_added(self, arg_name, domain):
        """Check whether {arg_name} already added to {domain}"""
        if arg_name in domain:
            raise RuntimeError(
                f'{type(self).__name__} was previously called with {arg_name}.'
                f'Can only call once per argument')

    def sig_indices(self, sig):
        inds = list(self.index.keys())
        return tuple(inds.index(idx) for idx in sig)

    def add_rank_limits(self, sig, min_val, max_val):
        sig_inds = self.sig_indices(sig)
        if min_val is not None:
            prev_min_val = self.rank_mins[sig_inds]
            self.rank_mins[sig_inds] = max(prev_min_val, min_val)
        if max_val is not None:
            prev_max_val = self.rank_maxs[sig_inds]
            self.rank_maxs[sig_inds] = min(prev_max_val, max_val)

    def sig_dims(self, sig):
        return [dim for s in sig for dim in self.index_dims[s]]

    def check_arg_name(self, arg_name):
        """Ensure {arg_name} is a valid argument name"""
        if arg_name not in self.parameter_names:
            raise RuntimeError(
                f'\'{arg_name}\' not a known parameter. '
                f'Known parameters are: {self.parameter_names}')

    def check_arg_type(self, arg_name, expected_type):
        """Check that {arg_name} is registered as having {expected_type}""" 
        arg_type = self.get_arg_type(arg_name)
        if arg_type != expected_type:
            raise RuntimeError(
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
        if arg_name not in self.parameter_names:
            raise RuntimeError(
                f'{type(self).__name__}: Attempted to add {arg_name} parameter '
                f'but it is not found in the framework op parameters. '
                f'Valid parameters are: {self.parameter_names}')
        if arg_name in self.arg_types:
            if self.arg_types[arg_name] != arg_type:
                raise RuntimeError(
                    f'{type(self).__name__}: Attempted to add {arg_name} as type '
                    f'{arg_type} to the registry, but it is already registered '
                    f'as type {self.arg_types[arg_name]}')
        self.arg_types[arg_name] = arg_type

    def get_arg_type(self, arg_name):
        """Retrieve the type(s) expected for {arg_name}"""
        self.check_arg_name(arg_name)
        if arg_name not in self.arg_types:
            raise RuntimeError(
                f'Argument \'{arg_name}\' does not have a registered type. ')
        return self.arg_types[arg_name]

    def get_return(self, idx):
        try:
            return self.returns[idx]
        except IndexError:
            raise RuntimeError(
                f'get_output({idx}) called but only {len(self.returns)} '
                f'returns')

    def prepare_call(self, op, bound_args):
        """Call during the framework call phase"""
        self.clear_call_state()
        self.arguments = bound_args
        self.calltime_config(op)

    def log_framework_error(self, err):
        self.framework_error = err

    def src_dtype(self, tensor_name):
        """Return the source tensor name which determines the dtype of
        {tensor_name}"""
        if tensor_name in self.dtype_valid:
            return tensor_name
        src = next((p[0] for p in self.dtype_equiv if p[1] == tensor_name),
                   None)
        if src is None:
            raise RuntimeError(f'\'{tensor_name}\' has no determined dtype')
        return src

    def build_pred_graph(self):
        """
        Build the predicate graph for validating all inputs
        """
        def wrap(func):
            return partial(func, self)

        PredNode.clear_registry()
        types = PredNode.add_node('__types', wrap(valid_types))
        dtypes = PredNode.add_node(self.DTYPES, wrap(valid_dtypes))
        ranks = PredNode.add_node(self.RANK, wrap(valid_ranks))
        dims = PredNode.add_node(self.DIMS, wrap(valid_dims), self.RANK)
        ranks.add_predicate_parent(types)
        dtypes.add_predicate_parent(types)

    def finalize_graphs(self):
        """
        Call this after the user-supplied schema function is executed.
        The Schema API will create more PredNodes and GenNodes on the registry,
        and some will be roots.
        """
        pred_nodes = PredNode.get_ordered_nodes()
        self.input_pred_graph = [n for n in pred_nodes if not
                n.name.startswith(self.RETURN)]
        self.return_pred_graph = [n for n in pred_nodes if
                n.name.startswith(self.RETURN) or n.name == self.DIMS]
        self.gen_graph = GenNode.get_ordered_nodes()

    def check_args(self):
        """
        The main function to check all input arguments for all constraints
        registered on the schema
        """
        error = fgraph.pred_graph_evaluate(self.input_pred_graph)
        self.opcheck_input_error = error

    def check_return(self, op_return):
        """
        Check the return tensors' shapes and types against those predicted by
        the framework
        """
        if self.opcheck_input_error is not None:
            return

        if not isinstance(op_return, (list, tuple)):
            op_return = (op_return,)
        self.returns = list(op_return)
        error = fgraph.pred_graph_evaluate(self.return_pred_graph)
        self.opcheck_return_error = error

    def generate_ranks(self):
        """
        Generate all allowed rank combinations.  Generates a list of maps.
        Each map has index => rank for each index in self.index
        """
        k = len(self.index)
        index_order = list(self.index.keys())
        gen = util.feasible_region(k, self.rank_mins, self.rank_maxs,
                                        self.rank_equiv, {})
        return [dict(zip(index_order, ranks)) for ranks in gen]
        
    def generate_dtypes(self):
        """
        Generate all allowed dtype combinations.  Generates a list of maps.
        Each map has a full tensor_name => dtype for each input tensor
        """
        # src_ten are tensor names which have independent dtypes
        src_ten, allowed_dtypes = zip(*self.dtype_valid.items())
        # tensor_name => index 
        equiv_map = { p[1]: src_ten.index(p[0]) for p in self.dtype_equiv }
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
        GenNode.add_node(self.RANK, lambda: self.generate_ranks())
        GenNode.add_node(self.DTYPES, lambda: self.generate_dtypes())

        # find the dependencies for the enclosed functions
        dims_parents = set()
        for _, arg_names in self.dims_from_dims.values():
            dims_parents.update(arg_names)
        for _, arg_names in self.dims_from_rank.values():
            dims_parents.update(arg_names)

        def func(ranks, **kwargs):
            return generate_dims(self, ranks, **kwargs)
        GenNode.add_node(self.DIMS, func, self.RANK, *dims_parents)

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
        for vals in fgraph.gen_graph_iterate(self.gen_graph):
            # extract the values from the argument nodes of the graph
            # print(vals)
            arg_dict = { k: v for k, v in vals.items() if k in
                    self.parameter_names }
            # print(arg_dict.keys())
            try:
                print(vals[self.DIMS])
                self.wrapped_op(**arg_dict)
            except:
                pass

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

    def report(self):
        print(f'OpCheck Input Error: {self.opcheck_input_error}')
        print(f'Framework Error: {self.framework_error}')
        print(f'OpCheck Output Error: {self.opcheck_return_error}')
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
    for src_name, trg_name in op.dtype_equiv:
        src = op.get_arg(src_name)
        trg = op.get_arg(trg_name)
        if trg.dtype != src.dtype:
            return False, DTypeNotEqual(src_name, trg_name)
    return True, None

def valid_ranks(op):
    """
    Using the rank allowed combinations and rank inference constraints,
    resolve the ranks to a single combination
    """
    const_map = {}
    for sig, rank_func in op.sig_ranks.items():
        sig_inds = op.sig_indices(sig)
        const_map[sig_inds] = rank_func(op)

    for sig, dims_func in op.sig_dims.items():
        sig_inds = op.sig_indices(sig)
        dims = dims_func(op)
        const_map[sig_inds] = len(dims) 

    k = len(op.index)

    rank_list = list(util.feasible_region(k, op.rank_mins, op.rank_maxs,
        op.rank_equiv, const_map))

    if len(rank_list) == 0:
        return False, NoMatchingRanks()
    elif len(rank_list) > 1:
        return False, AmbiguousRanks()
    else:
        index_ranks = dict(zip(op.index.keys(), rank_list[0]))
        return True, index_ranks

def valid_dims(op, rank_map):
    """
    Infer index dims from Tensor shapes, and shape-parameter values,
    signatures and inferred index ranks
    """
    index_dims = {}
    for idx in rank_map.keys():
        # find all usages of idx
        idx_shapes = set()
        for sig, dims_func in op.sig_dims.items():
            if idx not in sig:
                continue
            shape = dims_func(op)
            sub_range = calc_sig_range(idx, sig)
            idx_shape = shape[slice(*sub_range)]
            idx_shapes.add(tuple(idx_shape))

        if len(idx_shapes) != 1:
            return False, IndexUsageError(idx)
        else:
            index_dims[idx] = idx_shapes.pop()
    return True, index_dims

def valid_return_dims(op, dims_map, out_index):
    """
    Validate tensor output shape at {out_index}
    """
    if len(self.returns) != len(self.return_shapes):
        return False, OutputNumberMismatch(len(self.returns))
    ten = op.returns[out_index]
    if not isinstance(ten, tf.Tensor):
        return False, ReturnTypeError(tf.Tensor, type(ten))
    shape = ten.shape.as_list()
    sig = op.return_shapes[out_index]
    for idx in sig:
        sub_range = calc_sig_range(idx, sig)
        idx_shape = shape[slice(*sub_range)]
        expected_idx_shape = dims_map[idx]
        if idx_shape != expected_idx_shape:
            return False, ShapeError(f'return[{out_index}]', idx, idx_shape)
    return True, None

def generate_dims(op, ranks, **kwargs):
    """
    Generates all remaining dims for indexes, such that the total number of
    elements of all tensors is between certain limits.  {partial_dims} is a map
    of idx => dims which are calculated from ranks only. {ranks} is a map of
    idx => rank.  {kwargs} is the union of all arguments needed for the
    op.dims_from_dims functions.
    """

    # calculate the rank-dependent dims
    partial_dims = {}
    for idx, info in op.dims_from_rank.items():
        func, arg_names = info
        call_args = { n:kwargs[n] for n in arg_names }
        dims = func(ranks, **call_args)
        if not util.is_iterable(dims):
            raise RuntimeError(
                f'dims_from_rank \'{func}\' from '
                f'schema \'{op.op_path}\' returned a non-iterable.')
        partial_dims[idx] = dims

    calc_dims = list(partial_dims.keys()) + list(op.dims_from_dims.keys())
    free_inds = [ i for i in ranks.keys() if i not in calc_dims ]
    sigs = list(op.sig_dims.keys())
    sigs.extend(op.return_shapes)

    def dims_map(dims):
        # construct known dims map
        known_dims = {}
        offset = 0
        for i, idx in enumerate(free_inds):
            rank = ranks[idx] 
            known_dims[idx] = dims[offset:offset+rank]
            offset += rank
        known_dims.update(partial_dims)
        return known_dims

    def nelem(dims):
        known_dims = dims_map(dims)

        # compute all dependent dims
        for idx, info in op.dims_from_dims.items():
            func, arg_names = info
            call_args = { n:kwargs[n] for n in arg_names }
            dims = func(known_dims, **call_args)
            known_dims[idx] = dims

        sum_nelem = 0
        for sig in sigs:
            shape = calc_sig_dims(known_dims, sig)
            sum_nelem += np.prod(shape)
        return sum_nelem

    k = len(free_inds)
    min_nelem = 100000
    max_nelem = 200000
    dims = util.bsearch_integers(k, min_nelem, max_nelem, nelem)
    return [dims_map(dims)]

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

