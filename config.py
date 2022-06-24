import numpy as np
import itertools

class EinTup(object):
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims
        self._value = None

    def __repr__(self):
        return f'EinTup {self.name}: {self.dims}'

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

    def cycle(self, *eintups):
        for e in eintups:
            if e not in self.tups:
                raise RuntimeError(f'Config::cycle got unknown eintup name {e}')
        return itertools.product(*(self.tups[e] for e in eintups))

    def dims(self, eintup):
        if eintup not in self.tups:
            raise RuntimeError(f'Config::dims got unknown eintup name '
            f'\'{eintup}\'')
        return self.tups[eintup].dims

    def rank(self, eintup):
        if eintup not in self.tups:
            raise RuntimeError(f'Config::dims got unknown eintup name '
            f'\'{eintup}\'')
        return len(self.tups[eintup].dims)


if __name__ == '__main__':
    cfg = Config()
    cfg.set_ranks({'batch': 2, 'slice': 1})
    for val in cfg.cycle('slice'):
        print(val)

