import itertools
from collections import OrderedDict, defaultdict
from . import error
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
        self.test_arguments = None

        # returns, set after the framework operation returns.  changes with
        # each call 
        self.returns = []

        # these are set to different values during call-time
        self.index_ranks = {}
        self.index_dims = {} 

        # these members are set during schema initialization and do not change
        # during calls to the framework

        # map of EinTups, letter -> tup (with tup.name a description)
        self.index = OrderedDict()

        # Constraints on the allowed combinations of ranks 
        # Possible TODO: rewrite these to use actual index names rather than
        # index positions
        self.rank_maxs = defaultdict(lambda: 10000)
        self.rank_mins = defaultdict(lambda: 0) 
        self.rank_equiv = [] # [(sig1, sig2), ...] sig1 and sig2 equal ranks

        # map of arg_name => TypedArg
        # this will be populated with all arguments.  arg_shape, arg_rank,
        # arg_rank_func and arg_option refer to these arguments and provide
        # further interpretation.  There may be more than one interpretation of
        # a TypedArg
        self.arg_types = {}

        # map of arg_name => ShapeArg
        self.arg_shape = {}

        # map of arg_name => RankArg
        self.arg_rank = {}

        # map of arg_name => RankFuncArg
        self.arg_rank_func = {}

        # map of arg_name => OptionArg
        self.arg_option = {}

        # Ordered (and named) Signatures for returns 
        self.return_shapes = []

        # map of idx => func.  func(op) called at end of Dims Resolution Phase
        self.index_dims_funcs = {}

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
                f'{self.__qualname__} was previously called with {arg_name}.'
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

    def get_arg(self, arg_name, default=None):
        """Retrieve the value of {arg_name} argument at call-time."""
        if arg_name not in self.arg_types:
            raise RuntimeError(
                f'\'{arg_name}\' not found in registered arguments. '
                f'Arguments are: {self.arg_types.keys()}')

        if arg_name not in self.parameter_names:
            raise RuntimeError(
                f'\'{arg_name}\' not a known parameter. '
                f'Known parameters are: {self.parameter_names}')
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
                f'{self.__qualname__}: Attempted to add {arg_name} parameter '
                f'but it is not found in the framework op parameters. '
                f'Valid parameters are: {self.parameter_names}')
        if arg_name in self.arg_types:
            if self.arg_types[arg_name] != arg_type:
                raise RuntimeError(
                    f'{self.__qualname__}: Attempted to add {arg_name} as type '
                    f'{arg_type} to the registry, but it is already registered '
                    f'as type {self.arg_types[arg_name]}')
        self.arg_types[arg_name] = arg_type

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

    def generate_ranks(self):
        """Generate all allowed rank combinations"""
        k = len(self.index)
        yield from util.feasible_region(k, self.rank_mins, self.rank_maxs,
                self.rank_equiv, {})

    def check_args(self):
        """
        Evaluate the current argument values, logging any errors found.  Return
        True if no errors, False otherwise.
        """
        self.can_check_return = False

        # Type Check Phase: Check that all arguments have valid type
        type_violation = False
        for arg_name in self.arguments:
            if not self.valid_arg_type(arg_name):
                err = error.ArgTypeError(arg_name)
                self.log_error(err)
                type_violation = True

        if type_violation:
            return

        # Rank Resolution Phase
        inferred_ranks = None
        rank_list = self.resolve_ranks()
        if len(rank_list) == 0:
            self.log_error(error.NoMatchingRanks())
        elif len(rank_list) > 1:
            self.log_error(error.AmbiguousRanks())
        else:
            inferred_ranks = rank_list[0]

        if inferred_ranks is None:
            return 

        # Set the index ranks to inferred
        # TODO: make this safer
        self.index_ranks = dict(zip(self.index.keys(), inferred_ranks))
        self.index_dims = {}

        # Dims Resolution Phase:
        # Are the indices used consistently?
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
                self.log_error(error.IndexUsageError(idx))
            else:
                self.index_dims[idx] = idx_shapes.pop()

        # Now, calculate any dimensions which have registered functions
        self.compute_index_dims()

        # TODO: Are the tensor types valid?
        self.can_check_return = True

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
                err = error.OutputShapeError(shape.idx) 
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

    def validate_schema(self):
        """
        Generate a set of input argument value configurations for the op, and
        run the op on each configuration.  The set includes all possible
        combinations of valid index ranks, input tensor dtypes, and settings
        for parameters that are not declared with arg_unchecked in the schema.
        Also generates some invalid configurations.  Checks that opcheck will
        pass the valid ones, and log the appropriate errors for the invalid
        ones. 
        """
        free_idx = [ i for i in self.index if i not in self.index_dims_funcs ]
        k = len(free_idx)
        offset = 0
        offsets = {}
        for i, idx in enumerate(free_idx):
            rank = self.index_ranks[idx]
            offsets[idx] = (offset, offset+rank)
            offset += rank

        tensor_sigs = [ o.sig for o in self.arg_shape.values() ]
        tensor_sigs.extend(self.return_shapes)

        def max_nelem_func(dims):
            # set all free dims
            for idx in free_idx:
                idx_dims = dims[slice(*offsets[idx])]
                self.set_index_dims(idx, idx_dims)
            # now set computed dims
            self.compute_index_dims()

            max_nelem = 0
            for sig in tensor_sigs:
                sig_dims = [ d for idx in sig for d in self.index_dims[idx] ]
                nelem = np.prod(sig_dims)
                max_nelem = max(max_nelem, nelem)
            return max_nelem
        
        # enumerate all possible ranks
        for ranks in self.generate_ranks():
            # max_nelem_func sets index dims, no need to collect output
            self.index_ranks = dict(zip(self.index.keys(), ranks))
            _ = util.bsearch_integers(k, min_nelem, max_nelem, max_nelem_func)

            # generate tensors
            for name, arg in self.arg_shape.items():
                shape = self.sig_dims(arg.sig)
                if isinstance(arg, TensorShapeArg):
                    # TODO: handle dtypes 
                    ten_dtype = tf.int32
                    ten = tf.random.normal(shape, dtype=ten_dtype)
                    self.test_arguments[name] = ten
                elif isinstance(arg, ListShapeArg): 
                    self.test_arguments[name] = shape
                else:
                    pass

            # run the test
            self.wrapped_op(self.test_arguments)

    # produce ['i1', 'i2', 'i3'] from 'i' for example
    def index_list(self, idx):
        rank = self.index_rank[idx]
        if rank == 1:
            return [idx]
        else:
            return [idx + str(i) for i in range(1, rank+1)]

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
            if isinstance(err, error.ShapeError):
                msg += self.print_indices()
                msg += '\n\n'
                msg += self.print_inputs(err.index_letter)
            elif isinstance(err, error.NoMatchingRanks):
                msg += err.message(self)
        if msg != '':
            print(msg)
        # return msg

