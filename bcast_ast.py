import tensorflow as tf
import enum

class AST(object):
    def __init__(self):
        pass

    def get_inds(self):
        return set()

    # core logic for broadcasting
    def broadcast_shape(self, full_inds):
        core_inds = list(filter(lambda e: e in self.get_inds(), full_inds))
        def get_dim(ind):
            if ind in core_inds:
                return self.cfg.dims(ind) 
            else:
                return [1] * self.cfg.rank(ind)
        bcast_dims = [ dim for ind in full_inds for dim in get_dim(ind) ]
        return core_inds, bcast_dims


class RandomCall(object):
    # apply a function pointwise to the materialized arguments
    # args can be: constant or array-like
    def __init__(self, cfg, min_expr, max_expr, dtype_string):
        self.cfg = cfg
        if dtype_string == 'INT':
            self.dtype = tf.int32
        elif dtype_string == 'FLOAT':
            self.dtype = tf.float64
        else:
            raise RuntimeError(f'dtype must be INT or FLOAT, got {dtype_string}')
        self.min_expr = min_expr
        self.max_expr = max_expr

    def evaluate(self, ind_list):
        slice_inds = self.min_expr.get_inds().union(self.max_expr.get_inds())
        lead_shape = [ dim for ind in slice_inds for dim in self.cfg.dims(ind) ]

        # trailing shape
        bcast_inds = set(ind_list).difference(slice_inds)
        bcast_shape = [ dim for ind in bcast_inds for dim in self.cfg.dims(ind) ]
        results = []
        print(f'slice_inds: {slice_inds}, bcast_inds: {bcast_inds}')
        ordered_inds = sorted(slice_inds)
        for _ in self.cfg.cycle(*ordered_inds):
            slc = tf.random.uniform(
                    shape=bcast_shape,
                    minval=self.min_expr.value(),
                    maxval=self.max_expr.value(),
                    dtype=self.dtype)
            results.append(slc)
        ten = tf.stack(results)
        ten = tf.reshape(ten, lead_shape + bcast_shape)
        return ten

class RangeExpr(AST):
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
                    f' must have dimension equal to rank of key EinTup.')

    def evaluate(self, full_inds):
        core_inds, bcast_shape = self.broadcast_shape(full_inds)
        ranges = tf.meshgrid(
                *[tf.range(e) for e in self.cfg.dims(self.key_ind)]
                )
        ndrange = tf.stack(ranges, axis=self.cfg.rank(self.key_ind))
        return tf.reshape(ndrange, bcast_shape)

    def get_inds(self):
        return {self.key_ind, self.last_ind}


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
    ten = rc.evaluate(['slice'])
    print(ten)
    print(ten.shape)
    print(cfg.tups)

