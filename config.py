import numpy as np
import itertools

class Shape(object):
    # simple data class
    def __init__(self):
        self.dims = None

    def set(self, dims):
        self.dims = [ int(d) for d in dims ]

    def set_elem(self, ind, dim):
        if self.dims is None:
            raise RuntimeError('Cannot call set_elem() on uninitialized Shape')
        if ind >= len(self.dims):
            raise RuntimeError(
                f'set_elem() index {ind} out of bounds for length '
                f'{len(self.dims)} dims')
        self.dims[ind] = dim

    def get(self):
        if self.dims is None:
            raise RuntimeError('Cannot call get() on uninitialized Shape')
        return self.dims


class EinTup(object):
    def __init__(self, name, shape_of=None):
        self.name = name
        self.primary = (shape_of is None)
        self.shape = Shape() if self.primary else shape_of 
        self._value = None

    def __repr__(self):
        try:
            dimstring = ','.join([str(d) for d in self.dims()])
        except RuntimeError:
            dimstring = '?'
        return f'EinTup \'{self.name}\': [{dimstring}]'

    def __len__(self):
        return len(self.dims())

    def __iter__(self):
        self.index = np.ndindex(*self.dims())
        return self

    def __next__(self):
        # intentionally silent.  simply used to advance the position
        self._value = next(self.index)
        return self.value()

    def same_shape_as(self, other):
        return self.shape is other.shape 

    def dims(self):
        return self.shape.get()

    def rank(self):
        return len(self.shape.get())

    def nelem(self):
        return np.prod(self.dims())

    def set_dims(self, dims):
        if not self.primary:
            raise RuntimeError(f'cannot call set_dims on non-primary EinTup')
        self.shape.set(dims)

    def set_dim(self, ind, val):
        self.shape.set_elem(ind, val)

    def value(self):
        if self._value is None:
            raise RuntimeError(f'{self} called value() before iteration')
        return self._value


class Config(object):
    def __init__(self, min_dim=5, max_dim=100):
        # map of eintup names to EinTup instances
        self.tups = {}
        self.array_sig = {}
        self.arrays = {}
        self.min_dim = min_dim
        self.max_dim = max_dim

    # TODO: make compatible with non-primary EinTups
    def set_dims(self, rank_map):
        for tup, rank in rank_map.items():
            if tup not in self.tups:
                raise RuntimeError(
                    f'Cannot set dims for unknown EinTup {tup}')
            dims = np.random.randint(self.min_dim, self.max_dim, rank)
            self.tups[tup].set_dims(dims)

    def set_one_dim(self, tup, ind, val):
        self.tup(tup).set_dim(ind, val)

    def maybe_add_tup(self, name, shadow_of=None):
        if name in self.tups:
            pass
        elif shadow_of is None:
            self.tups[name] = EinTup(name, None)
        elif shadow_of in self.tups:
            self.tups[name] = EinTup(name, shadow_of.shape)
        else:
            raise RuntimeError(
                f'Config::maybe_add_tup - shadow_of \'{shadow_of}\''
                f'provided but does not exist')
        return self.tups[name]

    def tup(self, eintup):
        if eintup not in self.tups:
            raise RuntimeError(
                    f'Config::tup() got unknown eintup name {eintup}')
        return self.tups[eintup]

    def dims(self, eintup):
        return self.tup(eintup).dims()

    def rank(self, eintup):
        return len(self.dims(eintup))

    def nelem(self, eintup):
        return self.tup(eintup).nelem()

    def cycle(self, *eintups):
        tups = [self.tup(e) for e in eintups]
        return itertools.product(*tups)

if __name__ == '__main__':
    cfg = Config()
    cfg.set_ranks({'batch': 2, 'slice': 1})
    for val in cfg.cycle('slice'):
        print(val)

