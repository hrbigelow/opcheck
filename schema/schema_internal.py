import itertools
from collections import OrderedDict
from .error import *

class SchemaInternal(object):
    """
    Internal workings of the Schema object

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

        # arguments given to the op
        self.arguments = None

        # outputs, set after 
        self.outputs = None 

        # map of EinTups, letter -> tup (with tup.name a description)
        self.index = OrderedDict()

        # Array of ShapeInput
        self.input_shapes = []

        # Array of RankInput 
        self.input_ranks = []

        # Ordered (and named) Signatures for outputs 
        self.output_shapes = []

        # Function provided for initializing the schema
        self.init_schema = None
        self.calltime_config = lambda op: None 

        # Errors
        self.errors = []

    def __repr__(self):
        ind = 'Index: \n' 
        ind += '\n'.join(let + ': ' + repr(tup) for let, tup in
                self.index.items())
        sig = 'Input signatures: \n'
        sig += '\n'.join(f'{sig}' for sig in self.input_shapes)
        out = 'Output signatures: \n'
        out += '\n'.join(f'{sig}' for sig in self.output_shapes)
        err = 'Errors: \n'
        err += '\n'.join(repr(e) for e in self.errors) 
        return '\n\n'.join((ind, sig, out, err))

    def clear(self):
        self.index.clear()
        self.input_shapes.clear()
        self.input_ranks.clear()
        self.output_shapes.clear()
        self.errors.clear()

    # fails if any letters in signature don't exist in self.index
    def check_sig(self, signature, name):
        if any(s not in self.index.keys() for s in signature):
            raise RuntimeError(
                f'Signature "{signature}" associated with \'{name}\' '
                f'contains one or more unregistered indices. '
                f'Current known indices are: '
                f"{','.join(self.index.keys())}"
                f'Call OpSchema::add_index with the missing index.')


    def sig_dims(self, sig):
        return [dim for s in sig for dim in self.index[s].dims()]

    def sig_rank(self, sig):
        return sum(self.index[s].rank() for s in sig)

    def get_index(self, letter_name):
        if letter_name not in self.index:
            raise RuntimeError(
                f'No index with letter name \'{letter_name}\' exists')
        return self.index[letter_name]

    def get_arg(self, arg_name, default=None):
        if arg_name not in self.arguments:
            raise RuntimeError(
                f'\'{arg_name}\' not found in call arguments. '
                f'Arguments are: {self.arguments.keys()}')
        arg = self.arguments[arg_name]
        if arg is None:
            arg = default
        return arg

    def get_output(self, idx):
        try:
            return self.outputs[idx]
        except IndexError:
            raise RuntimeError(
                f'get_output({idx}) called but only {len(self.outputs)} '
                f'outputs')

    def init(self, op, bound_args):
        self.clear()
        self.arguments = bound_args
        self.init_schema(op)
        self.calltime_config(op)
        for idx, tup in self.index.items():
            if tup.rank_range is None and tup.rank_parent is None:
                raise RuntimeError(
                    f'Schema for {op.p.op_path} does not define any rank '
                    f'constraint for index \'{idx}\' ({tup.name})')
            tup.lift_rank_range()

    def set_outputs(self, op_return):
        """Register the op outputs {op_return} with the schema.  {op_return}
        may be a single value or iterable"""
        if isinstance(op_return_val_or_list, (list, tuple)):
            self.outputs = list(op_return_val_or_list)
        else:
            self.outputs = [op_return_val_or_list]

    def log_error(self, err):
        self.errors.append(err)

    def evaluate(self):
        # loop through all valid ranks
        tups = self.index.values()
        range_tups = [ t for t in tups if t.rank_range is not None ]
        range_list = [ t.rank_range for t in range_tups ]
        combos = itertools.product(*range_list)

        ranks_found = False
        for ranks in combos:
            for tup in self.index.values():
                tup.clear()
            for t, r in zip(range_tups, ranks):
                t.set_rank(r)
            for t in tups:
                t.calc_rank()

            # check if input ranks match signature ranks
            if all(s.valid_rank() for s in self.input_shapes +
                    self.input_ranks):
                ranks_found = True
                break
        
        if not ranks_found:
            # The rank combination was inconsistent.  Try harder to guess
            # what the user intended?
            self.log_error(NoMatchingRanks())
            return False

        # Found a rank combination matching the inputs.
        # Check all index shape <=> input shape consistency
        dims_valid = True
        for shape in self.input_shapes:
            for letter in shape.sig:
                tup = self.index[letter]
                sub_dims = shape.sub_dims(letter)
                if tup.has_dims():
                    if tup.dims() != shape.sub_dims(letter):
                        self.log_error(ShapeError(shape.name, letter, sub_dims))
                        dims_valid = False
                else:
                    tup.set_dims(sub_dims)

        # Any remaining tups without dims are by definition output-only.
        # They must have a gen_expr
        for tup in self.index.values():
            if not tup.has_dims():
                tup.gen_dims()

        return dims_valid

    # produce ['i1', 'i2', 'i3'] from 'i' for example
    def index_list(self, letter):
            tup = self.index[letter]
            if tup.rank() == 1:
                return [letter]
            else:
                return [letter + str(i) for i in range(1, tup.rank()+1)]

    # produce ['b', 'i1', 'i2', 'i3', 'k'] from 'bik' for example
    def sig_list(self, sig):
        return [ind for s in sig for ind in self.index_list(s)]

    # produce [1,4] from letter='i', sig='bik' (assuming rank 3 for i) for
    # example
    def sig_range(self, letter, sig):
        ind = sig.index(letter)
        start = sum(self.index[l].rank() for l in sig[:ind])
        rank = self.index[letter].rank()
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
        return self.print_shapes(self.output_shapes, highlight)

    # check that the framework op output shapes match those predicted from
    # opcheck.
    def validate(self, ret_val):
        err = ''
        msg = ''

        if not isinstance(ret_val, tuple):
            ret_val = (ret_val,)
        if len(self.output_shapes) != len(ret_val):
            err += f'OpSchema({self.op_path}) expected '
            f'{len(self.output_shapes)} outputs but framework returned '
            f'{len(ret_val)}\n'
        else:
            z = zip(self.output_shapes, ret_val, range(1, len(ret_val) + 1))
            for shape, ret, pos in z: 
                sig_dims = self.sig_dims(shape.sig)
                ret_dims = ret.shape.as_list()
                if sig_dims != ret_dims:
                    err += (f'Output {pos} {name} shape mismatch: '
                            f'opcheck expected {sig_dims} but framework returned '
                            f'{ret_dims}\n')
                shape.set_dims(ret_dims)

        if err == '':
            msg += 'Indices:\n'
            msg += self.print_indices()
            msg += '\n\n'
            msg += 'Inferred Signature with actual shapes:\n\n'
            msg += self.print_inputs()
            msg += self.print_outputs() 

        return err, msg

    def report(self):
        msg = ''
        for err in self.errors:
            if isinstance(err, ShapeError):
                # msg += 'Indices:\n'
                msg += self.print_indices()
                msg += '\n\n'
                msg += self.print_inputs(err.index_letter)
            elif isinstance(err, NoMatchingRanks):
                msg += err.message(self)

        return msg


