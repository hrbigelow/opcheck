def pfx(kname):
    return kname[:kname.index(':')]

def kind(kname):
    return kname[kname.index(':'):]

def kname(prefix, kind):
    return prefix+kind

class Kind(object):
    # these cannot have prefixes
    SCHEMA = ':schema'
    DTYPES = ':dtypes'
    RANKS = ':index_ranks'
    IDIMS = ':input_index_dims'
    CDIMS = ':computed_dims'
    DIMS = ':all_dims'
    PSHAPE = ':predicated_shape'

    # these must have prefixes
    DTYPE = ':dtype'
    SIG = ':sig'
    # CONS = ':constraint'

    # these must have unique usages of prefixes
    ARG = ':arg'
    PSEUDO = ':pseudo'
    TENSOR = ':tensor'
    SHAPE = ':shape'
    RETURN_TENSOR = ':return_tensor'
    VALID_RETURN = ':valid_return'

class RankConstraints(object):
    def __init__(self, op):
        self.op = op
        self.maxs = {}
        self.mins = {}

        # sig => sig
        self.equiv = {}

        # constraints applied during predicate
        # for generation, these are not applied, instead there exist
        # inverse functions that go in the opposite direction
        self.sig_funcs = {}
        self.sig_args = {}

        # a set of prefixes.  expect inputs of pfx:sig and pfx:shape 
        self.shape_sig = set()

    def equate_ranks(self, target_sig, source_sig):
        self.equiv[target_sig] = source_sig

    def add_rank_limits(self, sig, min_val, max_val):
        if min_val is not None:
            prev_min_val = self.mins.get(sig, -1)
            self.mins[sig] = max(prev_min_val, min_val)
        if max_val is not None:
            prev_max_val = self.maxs.get(sig, 10000)
            self.maxs[sig] = min(prev_max_val, max_val)

    def add_shape_sig(self, prefix):
        self.shape_sig.add(prefix)

    def add_arg_rank(self, arg_name, sig):
        def get_arg(arg_name, op):
            return op._get_arg(arg_name)
        node_name = kname(arg_name, Kind.ARG)
        self.add_sig_func(sig, get_arg, (node_name, Kind.SCHEMA))

    def add_sig_func(self, sig, func, arg_knames):
        self.sig_funcs[sig] = func
        self.sig_args[sig] = arg_knames 

    def mins_inds(self):
        d = self.mins.items()
        return { self.op._sig_inds(sig): rank for sig,rank in d }

    def maxs_inds(self):
        d = self.maxs.items()
        return { self.op._sig_inds(sig): rank for sig,rank in d }

    def equiv_inds(self):
        d = self.equiv.items()
        return { self.op._sig_inds(s1): self.op._sig_inds(s2) for s1, s2 in d }

    def const_inds(self, kwargs):
        # evaluate each sig_func, providing the 
        const_map = {}
        for sig, func in self.sig_funcs.items():
            arg_names = self.sig_args[sig]
            call_args = tuple(kwargs[a] for a in arg_names)
            rank = func(*call_args)
            inds = self.op.sig_inds(sig)
            const_map[inds] = rank

        # process the shape_sig entries.
        for prefix in self.shape_sig:
            shape = kwargs[kname(prefix, Kind.SHAPE)]
            sig = kwargs[kname(prefix, Kind.SIG)]
            inds = self.op._sig_inds(sig)
            const_map[inds] = len(shape)

        return const_map

class DTypeConstraints(object):
    def __init__(self):
        self.valid = {}
        self.equiv = {}

    def add_valid(self, tensor_name, dtypes):
        self.valid[tensor_name] = tuple(dtypes)

    def add_equiv(self, target_tensor, source_tensor):
        self.equiv[target_tensor] = source_tensor

    def all(self):
        return (*self.valid, *self.equiv)

class CompDims(object):
    """
    Encapsulate the functions and arguments for computed index dimensions.
    """
    def __init__(self):
        # idx => func
        self.funcs = {}

        # idx => arg_names
        self.args = {}

    def add(self, index, comp_func, arg_knames):
        """
        Register {index} to be computed by {comp_func}, taking {arg_names} as
        arguments
        """
        self.funcs[index] = comp_func
        self.args[index] = arg_knames

    def get_args(self):
        return { a for l in self.args.values() for a in l }

    def __call__(self, idims_map, **kwargs):
        comp_dims_map = {}
        for index, func in self.funcs.items():
            arg_names = self.args[index]
            call_args = tuple(kwargs[a] for a in arg_names)
            comp_dims_map[index] = func(idims_map, *call_args)
        return comp_dims_map


