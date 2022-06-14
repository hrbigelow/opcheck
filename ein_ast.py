import numpy as np
    
class AST(object):
    def __init__(self, *args):
        pass

    def value(self):
        pass

class IntNode(AST):
    def __init__(self, val):
        super().__init__()
        self.val = int(val)

    def value(self):
        return (self.val,)

    def indices(self):
        return set()

class StarNode(AST):
    # Represents a wildcard slice
    def __init__(self, tup):
        self.tup = tup
        self.name = None

    def set_name(self, name):
        self.name = name

    def value(self):
        return tuple((None,)) * len(self.tup.shape_map[self.name])

    def indices(self):
        return set()

class EinTup(AST):
    # beg, end denote the position in the main tuple
    def __init__(self, tup, name):
        super().__init__()
        self.tup = tup 
        self.name = name

    def length(self):
        return self.tup.length(self.name)

    def value(self):
        return self.tup.value(self.name)

    def indices(self):
        return {self.name}

class ArraySlice(AST):
    def __init__(self, array_name, array, index_list_node):
        super().__init__()
        self.name = array_name
        self.array = array
        self.ind_node = index_list_node
        for node, name in zip(self.ind_node, self.array.sig):
            if isinstance(node, StarNode):
                node.set_name(name)

    def length(self):
        val = self.value()
        if len(val.shape) != 1:
            raise TypeError
        return val.shape[0]

    def maybe_convert(self, dtype):
        self.array.maybe_convert(dtype)

    def value(self):
        return self.array[self.ind_node.value()]

    def assign(self, rhs):
        self.array[self.ind_node.value()] = rhs

    def indices(self):
        return self.ind_node.indices()

# This is a polymorphic list of EinTup, IntNode, and ArraySlice instances
class IndexList(list, AST):
    def __init__(self, itr=[]):
        super(IndexList, self).__init__(itr)

    def append(self, node):
        super().append(node)

    def sig(self):
        return ''.join(e.name for e in self if isinstance(e, EinTup))

    def value(self):
        return tuple(ind for e in self for ind in e.value())

    def indices(self):
        return set(ind for e in self for ind in e.indices())

class Assign(AST):
    def __init__(self, lhs, rhs):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def evaluate(self):
        self.lhs.assign(self.rhs.value())

    # collect the unique set of indices in this assignment statement
    def indices(self):
        inds = set()
        inds.update(self.lhs.indices())
        inds.update(self.rhs.indices())
        return inds
    
class Call(AST):
    def __init__(self, func, index_list_node):
        super().__init__()
        if func == 'random':
            self.func = np.random.uniform
            self.dtype = np.float32
        elif func == 'randint':
            self.func = np.random.randint
            self.dtype = np.int32
        else:
            raise RuntimeError

        self.ind_node = index_list_node

    def value(self):
        return self.func(*self.ind_node.value())

    def indices(self):
        return self.ind_node.indices()

