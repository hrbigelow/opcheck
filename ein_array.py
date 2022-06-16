import numpy as np

# simple wrapper class
class SliceTuple(object):
    # subranks is { b: 2, a: 3, ... }
    # a dictionary of the ranks of each tuple-ein index
    def __init__(self):
        self._value = None
        self.slices = {}
        self.index = None

    # call when the shape map updates 
    def reset(self, shape_map):
        self.shape_map = shape_map

    # call for each statement.  
    def set_indices(self, indices):
        shapes = []
        offset = 0
        self.slices.clear()
        for ind in indices:
            shapes.extend(self.shape_map[ind])
            self.slices[ind] = slice(offset, len(shapes))
            offset = len(shapes)
        self.index = np.ndindex(tuple(shapes))

    def advance(self):
        self._value = next(self.index, None)
        return self._value is not None

    def length(self, name):
        sl = self.slices[name]
        return sl.stop - sl.start

    def value(self, name):
        return self._value[self.slices[name]]

# Wrapper for an ndarray whose shape is defined by
# sig and shape_map
class EinArray(object):
    def __init__(self, sig):
        self.sig = sig
        self.ary = np.empty((0,), dtype=np.float32)

    # may be called before or after the array is initialized
    def update_shape(self, shape_map):
        shape = sum([shape_map[i] for i in self.sig], [])
        self.ary.resize(shape, refcheck=False)

    def fill(self, val):
        self.ary.fill(val)

    def maybe_convert(self, dtype):
        if dtype != self.ary.dtype:
            self.ary = self.ary.astype(dtype)

    def __getitem__(self, idx):
        return self.ary[idx]

    def __setitem__(self, key, rhs):
        self.ary[key] = rhs



