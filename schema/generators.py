from functools import partial
from . import util

class GenDTypes(object):
    def __init__(self, dtype_cons):
        self.dtype_cons = dtype_cons

    def __call__(self):
        """
        Generate all allowed dtype combinations.  Generates a list of maps.
        Each map has a full tensor_name => dtype for each input tensor
        """
        # src_ten are tensor names which have independent dtypes
        src_ten, allowed_dtypes = zip(*self.dtype_cons.valid.items())
        # tensor_name => index 
        equiv_map = { trg: src_ten.index(src) for trg, src in
                self.dtype_cons.equiv.items() }
        equiv_map.update({v: i for i, v in enumerate(src_ten)})

        combos = []
        for combo in itertools.product(*allowed_dtypes):
            el = { name: combo[ind] for name,ind in equiv_map.items() }
            combos.append(el)
        return combos

class GenRanks(object):
    def __init__(self, op, rank_cons):
        self.op = op
        self.rcons = rank_cons

    def __call__(self):
        """
        Generate all allowed rank combinations.  Generates a list of maps.
        Each map has index => rank for each index in self.index
        """
        rmins = self.rcons.rmins_inds()
        rmaxs = self.rcons.rmaxs_inds()
        reqs = self.rcons.req_inds()

        k = len(self.op.index)
        index_order = list(self.op.index.keys())
        gen = util.feasible_region(k, rmins, rmaxs, req, {})
        rank_list = list(gen)
        return [dict(zip(index_order, ranks)) for ranks in rank_list]

class Sig(object):
    """
    Generate a single signature using {sig_func} and any additional arguments.
    """
    def __init__(self, sig_func):
        self.sig_func = sig_func

    def __call__(self, **kwargs):
        return [self.sig_func(**kwargs)]


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

class GenIndexDims(object):
    """
    Generate all (input and return) index dims
    """
    def __init__(self):
        self.derived_func = {}
        self.derived_args = {}

    def add_derived(self, idx, func, *func_args):
        self.derived_func[idx] = func
        self.derived_args[idx] = func_args

    def calc_dims(self, rank_map, dims_map, kwargs):
        calc_dims_map = {}
        for idx, func in self.derived_func.items():
            args = self.derived_args[idx]
            arg_vals = []
            for a in args:
                if a == Kind.IRANKS:
                    v = rank_map
                elif a == Kind.IDIMS:
                    v = dims_map
                else:
                    v = kwargs[a]
                arg_vals.append(v)
            calc_dims = func(*arg_vals)
            calc_dims_map[idx] = calc_dims 
        return calc_dims_map

    def __call__(self, rank_map, **kwargs):
        sig_keys = [ k for k in kwargs.keys() if get_kind(k) == Kind.SIG ]
        sigs_map = { get_prefix(k): kwargs[k] for k in sig_keys }

        def nelem(op, rank_map, free_inds, calc_inds, flat_dims):
            free_dims_map = pack_dims_map(rank_map, free_inds, flat_dims)
            calc_dims_map = self.calc_dims(rank_map, free_dims_map, kwargs) 
            dims_map = { **free_dims_map, **calc_dims_map } 
            sum_nelem = 0
            for sig in sigs_map.values():
                shape = [d for s in sig for d in dims_map[s]]
                sum_nelem += np.prod(shape)
            return sum_nelem

        calc_inds = self.derived_funcs.keys()
        free_inds = [ k for k in rank_map.keys() if k not in calc_inds ]
        nelem_wrap = partial(nelem, op, rank_map, free_inds, calc_inds)

        k = sum(rank for idx, rank in rank_map.items() if idx in free_inds)
        min_nelem = 100000
        max_nelem = 200000
        dims = util.bsearch_integers(k, min_nelem, max_nelem, nelem_wrap)
        free_dims_map = pack_dims_map(rank_map, free_inds, dims)
        calc_dims_map = self.calc_dims(rank_map, free_dims_map, kwargs) 
        dims_map = { **free_dims_map, **calc_dims_map } 
        return [dims_map]

class GenTensor(object):
    def __init__(self, arg_name):
        self.arg_name = arg_name

    def __call__(self, sig, dims_map, dtype_map):
        dtype = dtype_map[self.arg_name]
        shape = calc_sig_dims(dims_map, signature)
        if dtype.is_integer:
            ten = tf.random.uniform(shape, minval=-10, maxval=10,
                    dtype=dtype)
        else:
            ten = tf.randot.normal(shape, dtype=dtype)
        return [ten] 


