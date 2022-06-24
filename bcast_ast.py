import tensorflow as tf
import enum

class AST(object):
    def __init__(self, *children):
        self.children = list(children)

    def get_inds(self):
        return {ind for c in self.children for ind in c.get_inds()}

    def split_perm(self, trg_inds, core_inds_unordered):
        core_inds = [ ind for ind in trg_inds if ind in core_inds_unordered ]
        bcast_inds = [ ind for ind in trg_inds if ind not in core_inds ]
        n_core = len(core_inds)
        if n_core != len(core_inds_unordered):
            raise RuntimeError(
                    f'split_perm: trg_inds ({trg_inds}) did not '
                    f'contain all core indices ({core_inds})')

        src_inds = core_inds + bcast_inds

        # src_inds[perm[i]] = trg_inds[i]
        perm = [ src_inds.index(ind) for ind in trg_inds ]
        # perm = [ trg_inds.index(ind) for ind in src_inds ] 
        return perm, n_core 

    def flat_dims(self, inds):
        return [ dim for ind in inds for dim in self.cfg.dims(ind) ]

    def get_cardinality(self, *inds):
        return [ self.cfg.nelem(ind) for ind in inds ]

    # core logic for broadcasting
    def broadcast_shape(self, full_inds, core_inds):
        dims = []
        for ind in full_inds:
            if ind in core_inds:
                dims.extend(self.cfg.dims(ind))
            else:
                dims.extend([1] * self.cfg.rank(ind))
        return dims


class RandomCall(AST):
    # apply a function pointwise to the materialized arguments
    # args can be: constant or array-like
    def __init__(self, cfg, min_expr, max_expr, dtype_string):
        super().__init__(min_expr, max_expr)
        self.cfg = cfg
        if dtype_string == 'INT':
            self.dtype = tf.int32
        elif dtype_string == 'FLOAT':
            self.dtype = tf.float64
        else:
            raise RuntimeError(f'dtype must be INT or FLOAT, got {dtype_string}')
        self.min_expr = min_expr
        self.max_expr = max_expr

    def evaluate(self, full_inds):
        core_inds = self.get_inds()
        perm, n_core = self.split_perm(full_inds, core_inds)
        card = self.get_cardinality(*(full_inds[p] for p in perm))
        full_dims = self.flat_dims(full_inds) 

        results = []
        # print(f'core_inds: {core_inds}, bcast_inds: {bcast_inds}')
        for _ in self.cfg.cycle(*core_inds):
            slc = tf.random.uniform(
                    shape=card[n_core:], # materialized broadcast
                    minval=self.min_expr.value(),
                    maxval=self.max_expr.value(),
                    dtype=self.dtype)
            results.append(slc)

        ten = tf.stack(results)
        ten = tf.reshape(ten, card)
        ten = tf.transpose(ten, perm)
        ten = tf.reshape(ten, full_dims) 
        return ten

class RangeExpr(AST):
    # Problem: no good way to instantiate 'children' here since
    # the eintup's are just strings
    # RANGE[s, c], with s the key_eintup, and c the 1-D last_eintup
    def __init__(self, cfg, key_eintup, last_eintup):
        super().__init__()
        self.cfg = cfg
        self.key_ind = key_eintup
        self.last_ind = last_eintup

    def prepare(self):
        if self.cfg.rank(self.last_ind) != 1:
            raise RuntimeError(f'RangeExpr: last EinTup \'{self.last_ind}\''
                    f' must have rank 1.  Got {self.cfg.rank(self.last_ind)}')
        if self.cfg.dims(self.last_ind)[0] != self.cfg.rank(self.key_ind):
            raise RuntimeError(f'RangeExpr: last EinTup \'{self.last_ind}\''
                    f' must have dimension equal to rank of key EinTup '
                    f'\'{self.key_ind}\'')

    def get_inds(self):
        return {self.key_ind, self.last_ind}

    def evaluate(self, trg_inds):
        core_inds_unordered = self.get_inds()
        perm, n_core = self.split_perm(trg_inds, core_inds_unordered)
        n_inds = len(trg_inds)
        n_bcast = n_inds - n_core
        src_inds = [None] * n_inds 
        for i in range(n_inds):
            src_inds[perm[i]] = trg_inds[i]

        core_inds = src_inds[:n_core]
        card = self.get_cardinality(*core_inds)
        ranges = [tf.range(e) for e in self.cfg.dims(self.key_ind)]
        ranges = tf.meshgrid(*ranges, indexing='ij')

        trg_dims = self.broadcast_shape(trg_inds, core_inds)

        # ndrange.shape = DIMS(self.key_ind) + DIMS(self.last_ind)
        # these two should also be consecutive in trg_inds
        ndrange = tf.stack(ranges, axis=self.cfg.rank(self.key_ind))
        ndrange = tf.reshape(ndrange, card + [1] * n_bcast)
        ndrange = tf.transpose(ndrange, perm)
        ndrange = tf.reshape(ndrange, trg_dims)
        return ndrange


class IntExpr(AST):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def value(self):
        return self.val


class Rank(AST):
    def __init__(self, cfg, arg):
        super().__init__()
        self.cfg = cfg
        self.ein_arg = arg

    def prepare(self):
        if self.ein_arg not in self.cfg.tups:
            raise RuntimeError(f'Rank arg {self.ein_arg} not a known EinTup')

    def value(self):
        return len(self.cfg.tups[self.ein_arg])

class DimKind(enum.Enum):
    Star = 0
    Int = 1
    EinTup = 2

class Dims(AST):
    def __init__(self, cfg, arg, ind_expr):
        super().__init__()
        self.cfg = cfg
        self.ein_arg = arg
        if ind_expr == ':':
            self.kind = DimKind.Star
        elif isinstance(ind_expr, int):
            self.kind = DimKind.Int
            self.index = ind_expr
        elif isinstance(ind_expr, str):
            self.kind = DimKind.EinTup
            self.ein_ind = ind_expr
        else:
            raise RuntimeError(f'index expression must be int, \:\, or EinTup')

    def __repr__(self):
        if self.kind == DimKind.Star:
            ind_str = ':'
        elif self.kind == DimKind.Int:
            ind_str = str(self.index)
        else:
            ind_str = self.ein_ind
        return f'{self.kind} Dims({self.ein_arg})[{ind_str}]'

    def prepare(self):
        if self.ein_arg not in self.cfg.tups:
            raise RuntimeError(f'Dims argument \'{self.ein_arg}\' not a known EinTup')
        if (self.kind == DimKind.Int and 
                self.index >= len(self.cfg.tups[self.ein_arg])):
            raise RuntimeError(f'Dims index \'{self.ind}\' out of bounds')
        if self.kind == DimKind.EinTup:
            if self.ein_ind not in self.cfg.tups:
                raise RuntimeError(f'Dims EinTup name \'{self.ind}\' not known EinTup')
            if len(self.cfg.tups[self.ein_ind]) != 1:
                raise RuntimeError(f'Dims EinTup index \'{self.ein_ind}\' must be '
                        f'rank 1, got \'{len(self.cfg.tups[self.ein_ind])}\'')
            if (self.cfg.dims(self.ein_ind)[0] >
                    len(self.cfg.dims(self.ein_arg))):
                raise RuntimeError(f'Dims EinTup index \'{self.ein_ind}\' must'
                f' have values in range of EinTup argument \'{self.ein_arg}\'.'
                f' {self.cfg.dims(self.ein_ind)[0]} exceeds '
                f'{len(self.cfg.dims(self.ein_arg))}')

    def value(self):
        d = self.cfg.dims(self.ein_arg)
        if self.kind == DimKind.Star:
            return d
        elif self.kind == DimKind.Int:
            return d[self.index]
        else:
            ein_val = self.cfg.tups[self.ein_ind].value()
            return d[ein_val[0]]
    
    def get_inds(self):
        if self.kind == DimKind.EinTup:
            return {self.ein_ind}
        else:
            return set()


if __name__ == '__main__':
    import config
    cfg = config.Config(5, 10)
    cfg.set_ranks({'batch': 2, 'slice': 3, 'coord': 1})
    cfg.tups['coord'].dims[0] = 3  
    dims = Dims(cfg, 'slice', 'coord')
    dims.prepare()
    rc = RandomCall(cfg, IntExpr(0), dims, 'INT')
    ten = rc.evaluate(['slice', 'coord'])
    # print(ten)
    # print(ten.shape)
    # print(cfg.tups)

    cfg.set_ranks({'batch': 3, 'slice': 3, 'coord': 1})
    cfg.tups['coord'].dims[0] = 3  
    rng = RangeExpr(cfg, 'batch', 'coord')
    rng.prepare()
    ten = rng.evaluate(['slice', 'batch', 'coord'])
    print(ten)
    print(ten.shape)
    print(cfg.tups)

