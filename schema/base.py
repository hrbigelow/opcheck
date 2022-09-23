import tensorflow as tf
import numpy as np
from collections import namedtuple, defaultdict
from .error import SchemaError
from . import util

def kpfx(kname):
    return kname.split(':')[0]

def kind(kname):
    idx = kname.index(':')
    return kname[idx:]

def kname(prefix, kind):
    return prefix+kind

def ksig(prefix):
    return prefix + Kind.SIG

def karg(prefix):
    return prefix + Kind.ARG

class Kind(object):
    # these cannot have prefixes
    SCHEMA = ':schema'
    DTYPES = ':dtypes'
    RANKS = ':ranks'
    IDIMS = ':input_dims'
    SINGLE_DIMS = ':single_dims'
    GEN_DIMS = ':gen_dims'
    GD_DIMS = ':gd_dims'
    COMP_DIMS_TEM = ':comp_dims_tem'
    PSHAPE = ':predicated_shape'
    NONE = ':none'

    # these must have prefixes
    DTYPE = ':dtype'
    SIG = ':sig'
    SIG_MAP = ':sig_map'
    SHAPE_MAP = ':shape_map'

    ARG = ':arg'
    PSEUDO = ':pseudo'
    DATA_FORMAT = ':data_format'
    DATA_TENSOR = ':data_tensor'
    SHAPE_LIST = ':shape_list'
    SHAPE_INT = ':shape_int'
    SHAPE_TENSOR = ':shape_tensor'
    SHAPE_TENSOR2D = ':shape_tensor2d'
    SHAPE = ':shape'
    RETURN_TENSOR = ':return_tensor'
    VALID_RETURN = ':valid_return'


class RankCandidates(object):
    """
    Produce all possible rank candidates, resolving min, max, and equiv
    constraints.
    """
    def __init__(self, op):
        self.op = op

        # sig => max_rank
        self.maxs = {}

        # sig => min_rank
        self.mins = {}

        # index => index 
        self.equiv = {}
        # self.equiv = { k: k for k in self.op.index.keys() }

    def equate_ranks(self, target_index, source_index):
        self.equiv[target_index] = source_index

    def add_rank_limits(self, sig, min_val, max_val):
        if min_val is not None:
            prev_min_val = self.mins.get(sig, -1)
            self.mins[sig] = max(prev_min_val, min_val)
        if max_val is not None:
            prev_max_val = self.maxs.get(sig, 10000)
            self.maxs[sig] = min(prev_max_val, max_val)

    def index_limited(self, index):
        return index in self.mins or index in self.maxs

    def index_equated(self, index):
        return index in self.equiv
    
    def all_index_ranks(self):
        fi = [ k for k in self.op.index.keys() if k not in self.equiv ]
        min_map = { tuple(fi.index(s) for s in sig): rank for sig, rank in
                self.mins.items() } 
        max_map = { tuple(fi.index(s) for s in sig): rank for sig, rank in
                self.maxs.items() } 
        gen = util.feasible_region(len(fi), min_map, max_map)
        def add_equiv(gen):
            for ranks in gen:
                rank_map = dict(zip(fi, ranks))
                eq_map = { t: rank_map[s] for t,s in self.equiv.items() }
                rank_map.update(**eq_map)
                yield rank_map
        return add_equiv(gen)

class RankConstraint(object):
    """
    Define a constraint rank(sig) == rank_func(shape), where sig and shape are
    the run-time signature and shape associated with {shape_arg}
    """
    def __init__(self, name, shape_arg, rank_func):
        self.name = name
        self.shape_arg = shape_arg
        self.rank_func = rank_func

    def observed_rank(self, shape_map, **kwargs):
        # return the observed rank of the associated shape argument
        # this takes **kwargs because sometimes, rank information comes from
        # other sources besides the shape_map
        shape = shape_map[self.shape_arg]
        return self.rank_func(shape)

    def computed_rank(self, sig_map, rank_map):
        # return the rank of the associated signature that is implied by the
        # index ranks
        sig = sig_map[self.shape_arg]
        return sum(rank_map[s] for s in sig)

    def rank_error(self, sig_map, shape_map, rank_map, **kwargs):
        """
        Computes the difference between the predicted rank of the constraint's
        argument's signature based on the proposed set of index ranks, and the
        observed rank.
        Negative means the fix is to add to the rank
        """
        obs_rank = self.observed_rank(shape_map, **kwargs) 
        cmp_rank = self.computed_rank(sig_map, rank_map)
        return obs_rank - cmp_rank

    def highlight_map(self):
        """
        Produce a map of arg_name => [dim1, dim2, ...], where dim1 etc are
        positions of the shape that should be highlighted with '^^'.
        """
        raise NotImplementedError

    def suggestion(self):
        """
        A plain-English suggestion to the user, describing what aspect of the
        input needs to be changed.
        """
        raise NotImplementedError

class SliceRankConstraint(RankConstraint):
    def __init__(self, shape_arg, slice_index):
        """
        Represent the logical constraint:

        rank(sig) == len(shape)

        where sig and shape are the signature and shape associated with
        {shape_arg}.{slice_index}.  These special nodes are created by the API
        call arg_shape_tensor2d.
        """
        node = f'{shape_arg}.{slice_index}'
        name = f'rank(sig({node})) == len({node})'
        super().__init__(name, node, len)
        self.arg_name = shape_arg

    def highlight_map(self, sig_map, shape_map, rank_map):
        obs_rank = self.observed_rank(shape_map)
        cmp_rank = self.computed_rank(sig_map, rank_map)
        lo = min(obs_rank, cmp_rank)
        hi = max(obs_rank, cmp_rank)
        inds = list(range(lo, hi))
        return { self.shape_arg: inds }

    def suggestion(self, rank_error):
        if rank_error == 0:
            return None
        elif rank_error < 0:
            msg = f'Increase {self.arg_name}.shape[1] by {-rank_error}'
        else:
            msg = f'Decrease {self.arg_name}.shape[1] by {rank_error}'

class ShapeRankConstraint(RankConstraint):
    def __init__(self, shape_arg, arg_kind):
        """
        Represent the logical constraint:

        rank(sig) == len(shape)

        where sig and shape are the signature and shape associated with
        {shape_arg}.

        {shape_arg} Kind may be one of DATA_TENSOR, SHAPE_INT, SHAPE_LIST,
        SHAPE_TENSOR, SHAPE_TENSOR2D 
        """
        name = f'rank(sig({shape_arg})) == len({shape_arg})'
        super().__init__(name, shape_arg, len)
        allowed_kinds = (Kind.DATA_TENSOR, Kind.SHAPE_LIST, Kind.SHAPE_INT,
                Kind.SHAPE_TENSOR, Kind.SHAPE_TENSOR2D)
        if arg_kind not in allowed_kinds:
            raise RuntimeError(
                f'{type(self).__qualname__}: got illegal arg_kind '
                f'\'{arg_kind}\'')
        self.arg_kind = arg_kind
        
    def highlight_map(self, sig_map, shape_map, rank_map):
        re = self.rank_error(sig_map, shape_map, rank_map)
        shape = shape_map[self.shape_arg]
        act_len = len(shape)
        cmp_len = act_len - re
        inds = list(range(min(act_len, cmp_len), max(act_len, cmp_len)))
        return { self.shape_arg: inds }

    def suggestion(self, rank_error):
        s = 's' if abs(rank_error) > 1 else ''
        if rank_error == 0:
            return None
        elif rank_error < 0:
            if self.arg_kind == Kind.DATA_TENSOR:
                msg = f'Add {-rank_error} dimension{s} to \'{self.shape_arg}\''
            elif self.arg_kind in (Kind.SHAPE_TENSOR, Kind.SHAPE_LIST):
                msg = f'Add {-rank_error} element{s} to \'{self.shape_arg}\''
            elif self.arg_kind == Kind.SHAPE_INT:
                msg = f'Increase \'{self.shape_arg}\' by {-rank_error}'
            else:
                pass
        else:
            if self.arg_kind == Kind.DATA_TENSOR:
                msg = (f'Remove {rank_error} dimension{s} from '
                f'\'{self.shape_arg}\'')
            elif self.arg_kind in (Kind.SHAPE_TENSOR, Kind.SHAPE_LIST):
                msg = (f'Remove {rank_error} element{s} from '
                        f'\'{self.shape_arg}\'')
            elif self.arg_kind == Kind.SHAPE_INT:
                msg = f'Decrease \'{self.shape-arg}\' by {-rank_error}'
        return msg

class IntRankConstraint(RankConstraint):
    """
    Define the constraint: rank(rank_sig) == arg_val, where arg_val is the
    value of {shape_arg}
    """
    def __init__(self, name, rank_arg, rank_sig):
        super().__init__(name, None, None)
        self.rank_sig = rank_sig
        self.rank_arg = rank_arg

    def observed_rank(self, _, **kwargs):
        val = kwargs[self.rank_arg]
        return val

    def computed_rank(self, sig_map, rank_map):
        sig = self.rank_sig
        return sum(rank_map[s] for s in sig)

    def highlight_map(self, *args):
        return { self.rank_arg: [0] }

    def suggestion(self, rank_error):
        if rank_error == 0:
            return None
        elif rank_error < 0:
            return f'Increase \'{self.shape_arg}\' by {-rank_error}'
        else:
            return f'Decrease \'{self.shape_arg}\' by {rank_error}'

class DimRankConstraint(RankConstraint):
    """
    Define a constraint called {name} with the logic:

    dims(source_idx)[0] = get_dims_func(shape)

    """
    def __init__(self, name, rank_sig, shape_arg, get_dims_func, source_idx):
        super().__init__(name, shape_arg, get_dims_func)
        self.rank_sig = rank_sig 
        self.source_idx = source_idx

    def computed_rank(self, _, rank_map):
        sig = self.rank_sig
        return sum(rank_map[s] for s in sig)

    def highlight_map(self, sig_map, shape_map, rank_map):
        hl = defaultdict(list) 
        for arg, shape in shape_map.items():
            sig = sig_map[arg]
            dim = 0
            for s in sig:
                if s == self.source_idx:
                    hl[arg].extend(range(dim, dim + rank_map[s]))
                dim += rank_map[s]
        return hl

    def suggestion(self, rank_error):
        if rank_error == 0:
            return None
        elif rank_error < 0:
            return (f'Increase the dimension of index \'{self.source_idx}\' by '
                    f'{-rank_error}')
        else:
            return (f'Decrease the dimension of index \'{self.source_idx}\' by '
                    f'{rank_error}')
    
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

