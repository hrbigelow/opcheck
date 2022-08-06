import itertools
from collections import OrderedDict
from ast_nodes import EinTup, SchemaFunctionExpr
from error import *

class OpSchema(object):
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

    5. Once configured, 

    """
    def __init__(self, op_path):
        self.op_path = op_path

        # map of EinTups, letter -> tup (with tup.name a description)
        self.index = OrderedDict()

        # Signatures for inputs (for example:  filters => 'fio'
        self.input_sig = {}

        # Actual shapes of inputs (and possibly outputs) from the function call
        self.tensor_shapes = {}

        # Ordered (and named) Signatures for outputs 
        self.output_sig = OrderedDict()

        # Function provided for initializing the schema
        self.init_schema = None

        # Errors
        self.errors = []

        self.has_outputs = False

    def __repr__(self):
        ind = 'Index: \n' 
        ind += '\n'.join(let + ': ' + repr(tup) for let, tup in
                self.index.items())
        sig = 'Input signatures: \n'
        sig += '\n'.join(f'{name}: {sig}' for name, sig in self.input_sig.items())
        out = 'Output signatures: \n'
        out += '\n'.join(f'{name}: {sig}' for name, sig in
                self.output_sig.items())
        err = 'Errors: \n'
        err += '\n'.join(repr(e) for e in self.errors) 
        return '\n\n'.join((ind, sig, out, err))

    def clear(self):
        self.index.clear()
        self.input_sig.clear()
        self.output_sig.clear()
        self.errors.clear()
        self.tensor_shapes.clear()
        self.has_outputs = False

    def clear_shapes(self):
        for tup in self.index.values():
            tup.clear()

    def add_index(self, letter_name, full_name):
        self.index[letter_name] = EinTup(full_name)

    # fails if any letters in signature don't exist in self.index
    def _check_sig(self, arg_name, signature):
        if any(s not in self.index.keys() for s in signature):
            raise RuntimeError(
                f'Signature "{signature}" given for input argument '
                f'"{arg_name}" contains one or more unregistered '
                f'letter codes.  Currently registered letter codes are: '
                f"{','.join(self.index.keys())}"
                f'Call OpSchema::add_index with the missing letter code.')

    def add_signature(self, arg_name, signature):
        self._check_sig(arg_name, signature)
        self.input_sig[arg_name] = signature

    def get_dims(self, sig):
        return [dim for s in sig for dim in self.index[s].dims()]

    def rank(self, sig):
        return sum(self.index[s].rank() for s in sig)

    def append_output_signature(self, arg_name, signature):
        self._check_sig(arg_name, signature)
        self.output_sig[arg_name] = signature

    def _get_index(self, letter_name):
        if letter_name not in self.index:
            raise RuntimeError(
                f'No index with letter name \'{letter_name}\' exists')
        return self.index[letter_name]

    def equate_rank(self, letter1, letter2):
        ind1 = self._get_index(letter1)
        ind2 = self._get_index(letter2)
        ind1.equate_rank(ind2)

    def set_rank_range(self, letter_name, rng):
        ind = self._get_index(letter_name)
        ind.set_rank_range(rng)

    # constraint is a function accepting self 
    def add_dims_constraint(self, letter_name, constraint):
        ind = self._get_index(letter_name)
        static_expr = SchemaFunctionExpr(constraint, self)
        ind.add_gen_expr(constraint)

    def set_init(self, init_func):
        self.init_schema = init_func

    def init(self, bound_args):
        self.clear()
        self.init_schema(self, bound_args)
        for tup in self.index.values():
            if tup.rank_range is None and tup.rank_parent is None:
                tup.set_rank_range(range(1, 2))
            tup.lift_rank_range()
        for name in self.input_sig.keys():
            if name not in bound_args:
                raise RuntimeError(
                    f'Schema expected input {name} but was not found in '
                    f'argument list.  Argument list contains: '
                    f'{bound_args.keys()}')
            self.tensor_shapes[name] = bound_args[name].shape.as_list() 

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
            self.clear_shapes()
            for t, r in zip(range_tups, ranks):
                t.set_rank(r)
            for t in tups:
                t.calc_rank()
            # check if input ranks match signature ranks
            for name, sig in self.input_sig.items():
                if self.rank(sig) != len(self.tensor_shapes[name]):
                    break
            else:
                ranks_found = True
                break
        
        if not ranks_found:
            # The rank combination was inconsistent.  Try harder to guess
            # what the user intended?
            self.log_error(NoMatchingRanks())
            return False

        # Found a rank combination matching the inputs.
        # Now, either set dimension of index to input sub_dims,
        # or, check sub_dims against index.dims()
        dims_valid = True
        for name, sig in self.input_sig.items():
            input_dims = self.tensor_shapes[name]
            offset = 0
            for letter in sig:
                tup = self.index[letter]
                sub_dims = input_dims[offset:offset+tup.rank()]
                if tup.has_dims():
                    if tup.dims() != sub_dims:
                        self.log_error(ShapeError(name, letter, sub_dims))
                        dims_valid = False
                else:
                    tup.set_dims(sub_dims)
                offset += tup.rank()

        if any(not tup.has_dims() for tup in self.index.values()):
            raise RuntimeError(
                f'Failed to set dimensions for some indices'
                f'Schema:\n{self}')

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

    """
    Returns (for example) a multi-line string:

    input[ b,  i1,  i2, k]   filters[f1, f2, k, l]
         [10, 100, 100, 3]          [ 9,  9, 8, 8]
                        ^                    ^

    Highlighting the extent of the indices denoted by shape_error.index_letter 
    """
    def shape_interpretation(self, with_output=False, highlight_letter=None):
        items = list(self.input_sig.items()) 
        if with_output:
            items.extend(self.output_sig.items())

        n = len(items)
        outer = [ [None] * n for _ in range(2) ]
        highlight = [None] * n
        for ii, name_sig in enumerate(items):
            name, sig = name_sig
            rows = [ self.sig_list(sig), self.tensor_shapes[name] ]
            table, coords = tabulate(rows, ', ', False)
            outer[0][ii] = f'{name}[{table[0]}]'
            outer[1][ii] = f'[{table[1]}]'

            if highlight_letter is not None:
                width = len(table[0])
                b, e = self.sig_range(highlight_letter, sig)
                rng = range(coords[b][0], coords[e-1][1])
                hi = ''.join('^' if i in rng else ' ' for i in range(width))
                highlight[ii] = f' {hi} '

        if highlight_letter is not None:
            outer.append(highlight)

        out = ''
        sep = ''
        for i in range(n):
            rows = [ [row[i]] for row in outer ]
            table, _ = tabulate(rows, '', False)
            out += sep
            out += '\n'.join(table)
            sep = '\n\n'
        return out

        # this was for all in one line.  was less readable
        # outer_table, _ = tabulate(outer, '   ', False)
        # return '\n'.join(outer_table)

    # check that the framework op output shapes match those predicted from
    # opcheck.
    def validate(self, ret_val):
        err = ''
        msg = ''
        if not isinstance(ret_val, tuple):
            ret_val = (ret_val,)
        if len(self.output_sig) != len(ret_val):
            err += f'OpSchema({self.op_path}) expected {len(self.output_sig)} outputs '
            f'but framework returned {len(ret_val)}\n'
        else:
            self.has_outputs = True
            z = zip(self.output_sig.items(), ret_val, range(1, len(ret_val) + 1))
            for name_sig, ret, pos in z: 
                name, sig = name_sig
                sig_dims = self.get_dims(sig)
                ret_dims = ret.shape.as_list()
                if sig_dims != ret_dims:
                    err += (f'Output {pos} {name} shape mismatch: '
                            f'opcheck expected {sig_dims} but framework returned '
                            f'{ret_dims}\n')
                self.tensor_shapes[name] = ret_dims

        if err == '':
            msg = f'{self.op_path} opcheck passed\n\n'
            msg += 'Indices:\n'
            msg += self.print_indices()
            msg += '\n\n'
            msg += 'Inferred Signature with actual shapes:\n\n'
            msg += self.shape_interpretation(True)

        return err, msg

    def report(self):
        msg = ''
        for err in self.errors:
            if isinstance(err, ShapeError):
                # msg += 'Indices:\n'
                msg += self.print_indices()
                msg += '\n\n'
                msg += self.print_shape_interpretation(False, err.index_letter)
            elif isinstance(err, NoMatchingRanks):
                msg += err.message(self)

        return msg

