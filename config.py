import numpy as np
import itertools

class EinTup(object):
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims
        self._value = None

    def __repr__(self):
        return f'EinTup \'{self.name}\': {self.dims}'

    def __len__(self):
        return len(self.dims)

    def __iter__(self):
        self.index = np.ndindex(*self.dims)
        return self

    def __next__(self):
        # intentionally silent.  simply used to advance the position
        self._value = next(self.index)
        return self.value()

    def value(self):
        if self._value is None:
            raise RuntimeError(f'{self} called value() before iteration')
        return self._value


class Config(object):
    def __init__(self, min_dim=5, max_dim=100):
        # map of eintup names to EinTup instances
        self.tups = {}
        self.min_dim = min_dim
        self.max_dim = max_dim

    def set_ranks(self, rank_map):
        def make_tup(name, rank):
            return EinTup(name, np.random.randint(self.min_dim, self.max_dim,
                rank))
        self.tups = { name: make_tup(name, rank) for name, rank in
                rank_map.items() }

    def _check_eintup(self, func, eintup):
        if eintup not in self.tups:
            raise RuntimeError(f'Config::{func} got unknown eintup name {eintup}')

    def dims(self, eintup):
        self._check_eintup('dims', eintup)
        return self.tups[eintup].dims

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

