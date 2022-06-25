import numpy as np
import itertools

class Shape(object):
    # simple data class
    def __init__(self):
        pass

    def set(self, dims):
        self.dims = dims

    def get(self):
        return self.dims


class EinTup(object):
    def __init__(self, name, shape_of=None):
        self.name = name
        self.primary = (shape_of is None)
        self.shape = shape_of or Shape() 
        self._value = None

    def __repr__(self):
        return f'EinTup \'{self.name}\': {self.dims()}'

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

    def set_dims(self, dims):
        if not self.primary:
            raise RuntimeError(f'cannot call set_dims on non-primary EinTup')
        self.shape.set(dims)

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
    def set_ranks(self, rank_map):
        def make_tup(name, rank):
            return EinTup(name, np.random.randint(self.min_dim, self.max_dim,
                rank))
        self.tups = { name: make_tup(name, rank) for name, rank in
                rank_map.items() }

    def maybe_add_tup(self, name, shadow_of=None):
        if name not in self.tups:
            shape_of = None
            if shadow_of is not None:
                if shadow_of not in self.tups:
                    raise RuntimeError(
                        f'Config::maybe_add_tup - shadow_of \'{shadow_of}\''
                        f'provided but does not exist'
                        )
                shape_of = self.tups[shadow_of].shape 
            self.tups[name] = EinTup(name, shape_of)

    def tup(self, eintup):
        if eintup not in self.tups:
            raise RuntimeError(
                    f'Config::{func} got unknown eintup name {eintup}')
        return self.tups[eintup]

    def dims(self, eintup):
        return self.tup(eintup).dims

    def rank(self, eintup):
        return len(self.dims(eintup))

    def nelem(self, eintup):
        return np.prod(self.dims(eintup))

    def cycle(self, *eintups):
        for e in eintups:
            self._check_eintup('cycle', e)
        return itertools.product(*(self.tups[e] for e in eintups))

if __name__ == '__main__':
    cfg = Config()
    cfg.set_ranks({'batch': 2, 'slice': 1})
    for val in cfg.cycle('slice'):
        print(val)

