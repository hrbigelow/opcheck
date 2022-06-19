import numpy as np

class ShapeConfig(object):
    def __init__(self):
        self.arrays = {}
        self.tup = SliceTuple()

    def set_ranks(self, rank_map):
        self.tup.reset(rank_map)

    def init_arrays(self):
        for ary in self.arrays.values():
            ary.update_dims(self.tup.dims_map)
            ary.fill(0)

    def prepare_for_statement(self, ast):
        indices = ast.get_indices()
        self.tup.set_indices(indices)

    def maybe_add_array(self, name, sig):
        if name not in self.arrays:
            self.arrays[name] = EinArray(sig)
        return self.arrays[name]

    def shape(self, name):
        return self.tup.dims_map[name]

    def rank(self, name):
        return self.tup.rank_map[name]

    def value(self, name):
        return self.tup.value(name)

    def print_array(self, name):
        print(self.arrays[name].ary)



# simple wrapper class
class SliceTuple(object):
    # subranks is { b: 2, a: 3, ... }
    # a dictionary of the ranks of each tuple-ein index
    def __init__(self):
        self._value = None
        self.slices = {}
        self.index = None

    # call when the rank map updates 
    def reset(self, rank_map):
        self.rank_map = rank_map
        self.dims_map = {
            ind: [np.random.randint(2, 5) for _ in range(sz)] for ind, sz in
            rank_map.items()
            }

    # call for each statement.  
    def set_indices(self, indices):
        dims = []
        offset = 0
        self.slices.clear()
        for ind in indices:
            dims.extend(self.dims_map[ind])
            self.slices[ind] = slice(offset, len(dims))
            offset = len(dims)
        self.index = np.ndindex(tuple(dims))

    def advance(self):
        self._value = next(self.index, None)
        return self._value is not None

    def length(self, name):
        sl = self.slices[name]
        return sl.stop - sl.start

    def value(self, name):
        return self._value[self.slices[name]]

# Wrapper for an ndarray whose dims is defined by
# sig and dims_map
class EinArray(object):
    def __init__(self, sig):
        self.sig = sig
        self.ary = np.empty((0,), dtype=np.float32)

    # may be called before or after the array is initialized
    def update_dims(self, dims_map):
        dims = sum([dims_map[i] for i in self.sig], [])
        self.ary.resize(dims, refcheck=False)

    def fill(self, val):
        self.ary.fill(val)

    def maybe_convert(self, dtype):
        if dtype != self.ary.dtype:
            self.ary = self.ary.astype(dtype)

    def __getitem__(self, idx):
        return self.ary[idx]

    def __setitem__(self, key, rhs):
        self.ary[key] = rhs



