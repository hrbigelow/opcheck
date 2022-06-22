import numpy as np

class ShapeConfig(object):
    def __init__(self):
        self.arrays = {}
        self.tup = SliceTuple()

    # called at program start
    def set_ranks(self, rank_map):
        self.tup.reset(rank_map)

    # called at program start
    def init_arrays(self):
        for ary in self.arrays.values():
            ary.update_dims()
            ary.fill(0)

    # called at statement start
    def prepare_for_statement(self, ast):
        indices = ast.get_indices()
        self.tup.set_indices(indices)

    def maybe_add_array(self, name, index_list):
        if name not in self.arrays:
            self.arrays[name] = EinArray(name, index_list)
        return self.arrays[name]

    def dims(self, name):
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

    def value(self, name):
        return self._value[self.slices[name]]

# Wrapper for an ndarray whose dims is defined by the index_list
class EinArray(object):
    def __init__(self, name, index_list):
        self.name = name
        self.index_list = index_list
        self.ary = np.empty((0,), dtype=np.float64)
        self.start_offset = (0,) 

    # may be called before or after the array is initialized
    def update_dims(self):
        mins = self.index_list.min()
        maxs = self.index_list.max()
        dims = list(h - l + 1 for l, h in zip(mins, maxs))
        self.start_offset = mins
        self.ary.resize(dims, refcheck=False)

    def fill(self, val):
        self.ary.fill(val)

    def maybe_convert(self, dtype):
        if dtype != self.ary.dtype:
            self.ary = self.ary.astype(dtype)

    def _idxadj(self, idx):
        return tuple(None if i is None else i - o 
                for i, o in zip(idx, self.start_offset))

    def __getitem__(self, idx):
        return self.ary[self._idxadj(idx)]

    def __setitem__(self, idx, rhs):
        self.ary[self._idxadj(idx)] = rhs



