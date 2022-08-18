import tensorflow as tf
import itertools
import numpy as np
from collections import OrderedDict, defaultdict
from .error import *
from .arg import *
from . import util


class SchemaInternal(object):
    """
    An object which represents the 'Shape API' of one framework operation.
    The lifecycle of the OpSchema is as follows:

    1. When opcheck.register(framework_op_name) is called, a new OpSchema is
       instantiated and enclosed in a wrapper function for the framekwork
       operation.

    2. The OpSchema is initialized with an associated init callback function
       provided to OpSchema::set_init.  

    3. When the user invokes that framework function, the wrapper is called,
       and calls OpSchema::init, which calls the enclosed callback, providing it
       with the same arguments as the framework function receives.

    4. Usually, only parameters which affect the Shape API logic itself will 
       be used during this call.

    TODO: complete docs
    """

    def __init__(self, op_path):
        self.op_path = op_path

        # framework op parameter names 
        self.parameter_names = None

        # arguments given to the op.  these change for each call
        self.arguments = None

        # arguments for the test generation
        self.test_arguments = {} 

        # returns, set after the framework operation returns.  changes with
        # each call 
        self.returns = []

        # these are set to different values during call-time
        self.index_ranks = {}
        self.index_dims = {}

        # these members are set during schema initialization and do not change
        # during calls to the framework

        # map of single-letter index name to index description string
        # For example:  'i' => 'input spatial'
        self.index = OrderedDict()

        # Constraints on the allowed combinations of ranks 
        # Possible TODO: rewrite these to use actual index names rather than
        # index positions
        self.rank_maxs = defaultdict(lambda: 10000)
        self.rank_mins = defaultdict(lambda: 0)
        self.rank_equiv = []  # [(sig1, sig2), ...] sig1 and sig2 equal ranks

        # map of arg_name => TypedArg
        # this will be populated with all arguments.  arg_shape, arg_rank,
        # arg_rank_func and arg_option refer to these arguments and provide
        # further interpretation.  There may be more than one interpretation of
        # a TypedArg
        self.arg_types = {}
        self.arg_shape = {}  # map of arg_name => ShapeArg
        self.arg_rank = {}  # map of arg_name => RankArg
        self.arg_rank_func = {}  # map of arg_name => RankFuncArg
        self.arg_option = {}  # map of arg_name => OptionArg
        self.return_shapes = []  # Ordered (and named) Signatures for returns

        # map of idx => func.  func(op) called at end of Dims Resolution Phase
        self.index_dims_funcs = {}

        # All tensors must appear in one or the other place.
        # Every tensor not in dtype_allowed must be connected to one through
        # the graph induced by the edges of dtype_equiv
        self.dtype_valid = {}  # arg_name => (tf.int32, ...)
        self.dtype_equiv = []  # [(arg_name1, arg_name2), ...] tensors must have
        # equal dtypes

        # Function provided for initializing the schema
        self.init_schema = None
        self.calltime_config = None

        # The actual op wrapped by OpCheck
        self.wrapped_op = None

        # Errors
        self.errors = []
        self.framework_error = None
        self.can_check_return = False

    def __repr__(self):
        rep = 'Index: \n'
        rep += '\n'.join(idx + ': ' + desc for idx, desc in self.index.items())
        rep += '\n\nShape Arguments: \n'
        rep += '\n'.join(repr(arg) for arg in self.arg_shape.values())
        rep += '\n\nRank Arguments: \n'
        rep += '\n'.join(repr(arg) for arg in self.arg_rank.values())
        rep += '\n\nReturn signatures: \n'
        rep += '\n'.join(repr(sig) for sig in self.return_shapes)
        rep += '\n\nErrors: \n'
        rep += '\n'.join(repr(e) for e in self.errors)
        return rep

    def clear_call_state(self):
        """Clear the data members which hold data from a framework call"""
        self.index_ranks.clear()
        self.index_dims.clear()
        self.returns.clear()
        self.errors.clear()
        self.framework_error = None

    # fails if any letters in signature don't exist in self.index
    def check_sig(self, signature, name):
        if any(s not in self.index.keys() for s in signature):
            raise RuntimeError(
                f'Signature "{signature}" associated with \'{name}\' '
                f'contains one or more unregistered indices. '
                f'Current known indices are: '
                f"{','.join(self.index.keys())}"
                f'Call OpSchema::add_index with the missing index.')

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

    def sig_rank(self, sig):
        return sum(self.index[s].rank() for s in sig)

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

    def log_error(self, err):
        self.errors.append(err)

    def log_framework_error(self, err):
        self.framework_error = err

    def compute_index_dims(self):
        """
        Call this at the end of the Dims Resolution Phase to compute any index
        dims which have registered functions.
        """
        for idx, func in self.index_dims_funcs.items():
            self.index_dims[idx] = func()

    def set_index_dims(self, idx, dims):
        if idx not in self.index_ranks:
            raise RuntimeError(
                f'Attempted to set dims for index \'{idx}\' but it has no '
                f'rank assigned')
        if len(dims) != self.index_ranks[idx]:
            raise RuntimeError(
                f'Attempted to set {idx} with dims {dims}, but it has rank '
                f'{self.index_ranks[idx]}')
        self.index_dims[idx] = list(dims)

    def resolve_ranks(self):
        """
        Using the rank allowed combinations and rank inference constraints,
        resolve the ranks to a single combination
        """
        # create an additional const constraint functions from the arguments
        const_map = {}
        merged = [*self.arg_shape.values(), *self.arg_rank.values()]
        for arg in merged:
            sig_inds = self.sig_indices(arg.sig)
            const_map[sig_inds] = arg.rank()
        k = len(self.index)

        rank_combos = list(util.feasible_region(k, self.rank_mins,
            self.rank_maxs, self.rank_equiv, const_map))
        return rank_combos

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

    def valid_types(self):
        """
        Check that every argument is of allowed type
        """
        valid = True
        for arg_name, expected_types in self.arg_types.items():
            arg = self.get_arg(arg_name)
            if expected_types is None:
                continue # ignored
            if not isinstance(arg, expected_types):
                self.log_error(ArgTypeError(arg_name))
                valid = False
        return valid

    def valid_tensor_dtypes(self):
        """Check that all tensor arguments have valid dtypes"""
        valid = True
        for ten_name, dtypes in self.dtype_valid.items():
            arg = self.get_arg(ten_name)
            assert(isinstance(arg, tf.Tensor))
            if arg.dtype not in dtypes:
                self.log_error(DTypeNotAllowed(ten_name, arg.dtype))
                valid = False
        for src_name, trg_name in self.dtype_equiv:
            src = self.get_arg(src_name)
            trg = self.get_arg(trg_name)
            if trg.dtype != src.dtype:
                self.log_error(DTypeNotEqual(src_name, trg_name))
                valid = False
        return valid

    def valid_first_phase_hooks(self):
        """Run all registered first-phase hooks"""
        return True

    def valid_ranks(self):
        """Infer the index ranks unambiguously and set them on the schema, or return
        False.  Index ranks are inferred using the shape-input signatures and
        schema rank constraints, along with any other rank information
        registered in arg_rank or arg_rank_func 
        """
        inferred_ranks = None
        rank_list = self.resolve_ranks()
        if len(rank_list) == 0:
            self.log_error(NoMatchingRanks())
        elif len(rank_list) > 1:
            self.log_error(AmbiguousRanks())
        else:
            inferred_ranks = rank_list[0]

        if inferred_ranks is None:
            return False

        # Set the index ranks to inferred
        # TODO: make this safer
        self.index_ranks = dict(zip(self.index.keys(), inferred_ranks))
        self.index_dims = {}
        return True

    def valid_rank_hooks(self):
        """Run any rank-dependent hooks"""
        return True

    def valid_dims(self):
        """
        Infer index dims from Tensor shapes, and shape-parameter values,
        signatures and inferred index ranks
        """
        valid = True
        for idx in self.index.keys():
            # find all usages of idx
            idx_shapes = set()
            for arg in self.arg_shape.values():
                if idx not in arg.sig:
                    continue
                shape = arg.dims()
                sub_range = self.sig_range(idx, arg.sig)
                idx_shape = shape[slice(*sub_range)]
                idx_shapes.add(tuple(idx_shape))

            if len(idx_shapes) != 1:
                self.log_error(IndexUsageError(idx))
                valid = False
            else:
                self.index_dims[idx] = idx_shapes.pop()
        return valid

    def valid_dims_hooks(self):
        """Run hooks that depend on initialized index dims"""
        return True

    def check_args(self):
        """
        The main function to check all input arguments for all constraints
        registered on the schema
        """
        self.can_check_return = False

        # First Phase
        if not self.valid_types():
            return
        if not self.valid_tensor_dtypes():
            return
        if not self.valid_first_phase_hooks():
            return

        # Rank Resolution
        if not self.valid_ranks():
            return
        if not self.valid_rank_hooks():
            return

        # Dims Resolution
        if not self.valid_dims():
            return
        _ = self.valid_dims_hooks()

        # Post-validation
        self.compute_index_dims()
        self.can_check_return = True
        return

    def check_return(self, op_return):
        """
        Check the return tensors' shapes and types against those predicted by
        the framework
        """
        if not self.can_check_return:
            return

        if not isinstance(op_return, (list, tuple)):
            op_return = (op_return,)
        if len(self.return_shapes) != len(op_return):
            self.log_error(OutputNumberMismatch(len(op_return)))
        self.returns = list(op_return)

        for shape in self.return_shapes:
            if self.sig_dims(shape.sig) != shape.dims():
                err = OutputShapeError(shape.idx)
                self.log_error(err)
        """
        if err == '':
            msg += 'Indices:\n'
            msg += self.print_indices()
            msg += '\n\n'
            msg += 'Inferred Signature with actual shapes:\n\n'
            msg += self.print_inputs()
            msg += self.print_outputs() 
        """

    def generate_ranks(self):
        """
        Generate all allowed rank combinations.  Generates a list of maps.
        Each map has index => rank for each index in self.index
        """
        k = len(self.index)
        index_order = self.index.keys()
        gen = util.feasible_region(k, self.rank_mins, self.rank_maxs,
                                        self.rank_equiv, {})
        return [dict(zip(index_order, ranks)) for ranks in gen]
        
    def generate_dtypes(self):
        """
        Generate all allowed dtype combinations.  Generates a list of maps.
        Each map has a full tensor_name => dtype for each input tensor
        """
        shapes = self.arg_shapes.values()
        ten_names = [ o.name for o in shapes if isinstance(o, TensorShapeArg) ]
        # src_ten are tensor names which have independent dtypes
        src_ten, allowed_dtypes = zip(*self.dtype_valid.items())
        # tensor_name => index 
        equiv_map = { p[1]: src_ten.index(p[0]) for p in self.dtype_equiv }
        equiv_map.update({v: i for i, v in enumerate(src_ten)})

        combos = []
        for combo in itertools.product(*basis):
            el = { n: combo[equiv_map[n]] for n in ten_names }
            combos.append(el)
        return combos

    def generate_dims(self, ranks, **kwargs):
        # kwargs can be used by the registered functions in
        # dims_from_dims and dims_from_rank.  Each of these kwargs
        # represents a gen_graph node and must be added as a parent
        # to this node
        # Phase 1 - initialize rank-dependent dims
        for idx, info in self.dims_from_rank:
            func, arg_names = info
            call_args = { n:kwargs[n] for n in arg_names }
            dims = func(**call_args)
            self.set_index_dims(idx, dims)

        # Phase 2 - identify free dims
        inds_list = set(self.index.keys()).difference(
                self.dims_from_rank.keys(),
                self.dims_from_dims.keys()
                )
        k = len(inds_list)

        def calc_rank_offsets(inds_list, rank_map):
            offset = 0
            offsets = {}
            for i, idx in enumerate(inds_list):
                rank = rank_map[idx] 
                offsets[idx] = (offset, offset + rank)
                offset += rank
            return offsets

        def sum_nelem_func(rank_offsets, tensor_sigs, dims):
            # set all free dims
            for idx, offsets in rank_offsets.items():
                idx_dims = dims[slice(*offsets)]
                self.set_index_dims(idx, idx_dims)

            # now set computed dims
            # Whatever kwargs are in arg_names, these must be added
            # as parents to this node
            for idx, info in self.dims_from_dims:
                func, arg_names = info
                call_args = { n:kwargs[n] for n in arg_names }
                dims = func(**call_args)
                self.set_index_dims(idx, dims)

            sum_nelem = 0
            for sig in tensor_sigs:
                shape = self.sig_dims(sig)
                print(sig, shape)
                sum_nelem += np.prod(shape)
            # print(sum_nelem)
            return sum_nelem
        
        rank_offsets = calc_rank_offsets(free_inds, ranks)
        tensor_sigs = [o.sig for o in self.arg_shape.values()]
        tensor_sigs.extend(ret.sig for ret in self.return_shapes)
        func = lambda inds: sum_nelem_func(rank_offsets, tensor_sigs, inds)

        # sets self.index_dims via sum_nelem_func
        min_nelem = 100000
        max_nelem = 200000
        _ = util.bsearch_integers(k, min_nelem, max_nelem, func)

        # Since the function updates self.index_dims as a side-effect, it is
        # the return value
        return [self.index_dims] # Just a one-element list empty value.

    def generate_tensor(self, arg, dims, dtypes):
        shape = self.sig_dims(arg.sig)
        ten_dtype = dtypes[name]
        if ten_dtype.is_integer:
            ten = tf.random.uniform(shape, minval=-10, maxval=10,
                    dtype=ten_dtype)
        else:
            ten = tf.random.normal(shape, dtype=ten_dtype)
        return [ten] 

    def generate_shape(self, arg, dims):
        shape = self.sig_dims(arg.sig)
        return [shape]

    def build_gen_graph(self):
        # by definition, the ranks generation doesn't depend on anything
        nodes = {} 
        parent_map = defaultdict(list)

        # Reserved names for the nodes, so they don't conflict with argument
        # names of ops
        RANKS = '__ranks'
        DIMS = '__dims'
        DTYPES = '__dtypes'

        ranks = gen_graph.GenNode(RANKS, lambda: self.generate_ranks())
        nodes[RANKS] = ranks

        dtypes = gen_graph.GenNode(DTYPES, lambda: self.generate_dtypes())
        nodes[DTYPES] = dtypes

        for arg_name, info in self.arg_generators.items():
            func, call_args = info
            node = gen_graph.GenNode(arg_name, func)
            nodes[arg_name] = node
            parent_map[arg_name].extend(call_args)

        for arg_name, parents in parent_map.items():
            node = nodes[arg_name]
            for pa in parents:
                node.add_parent(nodes[arg_name])

        # find the dependencies
        dims_parents = set()
        for _, arg_names in self.dims_from_dims.values():
            dims_parents.update(arg_names)
        for _, arg_names in self.dims_from_rank.values():
            dims_parents.update(arg_names)

        dims = gen_graph.GenNode(DIMS, lambda **kwargs:
                self.generate_dims(kwargs))
        nodes[DIMS] = dims

        for pa in dims_parents:
            dims.add_parent(nodes[pa])

        # tensors and shapes
        for arg_name, obj in self.arg_shapes.items():
            if isinstance(obj, TensorShapeArg):
                def gen_ten_wrap(dims, dtypes):
                    return self.generate_tensor(obj, dims, dtypes)
                node = gen_graph.GenNode(arg_name, gen_ten_wrap)
                node.add_parent(nodes[DIMS])
                node.add_parent(nodes[DTYPES])
            elif isinstance(obj, ListShapeArg):
                def gen_shape_wrap(dims):
                    return self.generate_shape(obj, dims)
                node = gen_graph.GenNode(arg_name, gen_shape_wrap)
                node.add_parent(nodes[DIMS])

        self.inputs_gen_graph = nodes

    def validate_schema(self):
        self.test_arguments.clear()
        roots = gen_graph.get_roots(self.inputs_gen_graph)
        for vals in gen_graph.iterate(roots):
            # extract the values from the argument nodes of the graph
            arg_dict = { k: v for k, v in vals.items() if k in
                    self.parameter_names }
            self.wrapped_op(*arg_dict)


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
        self.test_arguments.clear()

        rank_combos = self.generate_ranks()
        dtype_combos = self.generate_dtypes()

        # enumerate all combos of other arguments' values
        min_nelem = 100000
        max_nelem = 200000

        # TODO: set dims that are computed from ranks
        free_inds = [i for i in self.index if i not in self.index_dims_funcs]
        k = len(free_inds)
        tensor_sigs = [o.sig for o in self.arg_shape.values()]
        tensor_sigs.extend(ret.sig for ret in self.return_shapes)

        for dtypes, ranks in itertools.product(dtype_combos, rank_combos):

            self.index_ranks = ranks 
            rank_offsets = calc_rank_offsets(free_inds)
            func = lambda inds: sum_nelem_func(rank_offsets, tensor_sigs, inds)

            # sets self.index_dims via sum_nelem_func
            _ = util.bsearch_integers(k, min_nelem, max_nelem, func)

            # here, need to generate rank-dependent and dims-dependent argument
            # values.

            # generate tensors and shapes
            for name, arg in self.arg_shape.items():
                shape = self.sig_dims(arg.sig)
                print('generating tensors: ', arg.sig, shape)
                if isinstance(arg, TensorShapeArg):
                    ten_dtype = dtypes[name]
                    if ten_dtype.is_integer:
                        ten = tf.random.uniform(shape, minval=-10, maxval=10,
                                dtype=ten_dtype)
                    else:
                        ten = tf.random.normal(shape, dtype=ten_dtype)
                    self.test_arguments[name] = ten
                elif isinstance(arg, ListShapeArg):
                    self.test_arguments[name] = shape
                else:
                    pass

            # run the test
            self.wrapped_op(**self.test_arguments)

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
        msg = ''
        for err in self.errors:
            if isinstance(err, ShapeError):
                msg += self.print_indices()
                msg += '\n\n'
                msg += self.print_inputs(err.index_letter)
            elif isinstance(err, NoMatchingRanks):
                msg += err.message(self)
        if msg != '':
            print(msg)
        # return msg
