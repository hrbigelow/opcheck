import operator
import numpy as np
    
class AST(object):
    def __init__(self, *args):
        pass

class IntNode(AST):
    def __init__(self, val):
        super().__init__()
        self.val = int(val)

    def to_slice(self):
        return (self.val,)

    def get_eintup_names(self):
        return set()

class StarNode(AST):
    # Represents a wildcard slice
    def __init__(self, tup):
        self.tup = tup
        self.name = None

    def set_name(self, name):
        self.name = name

    def to_slice(self):
        return tuple((None,)) * len(self.tup.shape_map[self.name])

    def get_eintup_names(self):
        return set()

class SliceBinOp(AST):
    def __init__(self, slice1, slice2, op):
        self.slice1 = slice1
        self.slice2 = slice2
        opfuncs = [ operator.add, operator.sub, operator.mul, operator.truediv ]
        self.op = dict(zip('+-*/', opfuncs))[op]

    def value(self):
        return self.op(self.slice1.value(), self.slice2.value())

    def get_eintup_names(self):
        return set.union(
                self.slice1.get_eintup_names(),
                self.slice2.get_eintup_names())
        

class EinTup(AST):
    # beg, end denote the position in the main tuple
    def __init__(self, tup, name):
        super().__init__()
        self.tup = tup 
        self.name = name

    def length(self):
        return self.tup.length(self.name)

    def to_slice(self):
        return self.tup.value(self.name)

    def get_eintup_names(self):
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

    # returns a new ndarray
    def value(self):
        return self.array[self.ind_node.value()]

    def fill_ndarray(self, val):
        self.array.fill(val)

    def add(self, rhs):
        ind = self.ind_node.value()
        self.array[ind] += rhs

    def assign(self, rhs):
        ind = self.ind_node.value()
        self.array[ind] = rhs

    def to_slice(self):
        flat = self.value().squeeze()
        if flat.ndim != 1:
            raise RuntimeError('to_slice requires a flatten-able array')
        return tuple(flat.tolist())

    def get_eintup_names(self):
        return self.ind_node.get_eintup_names()

# This is a polymorphic list of EinTup, IntNode, and ArraySlice instances
class IndexList(list, AST):
    def __init__(self, itr=[]):
        super(IndexList, self).__init__(itr)

    def append(self, node):
        super().append(node)

    def sig(self):
        return ''.join(e.name for e in self if isinstance(e, EinTup))

    def value(self):
        return tuple(ind for e in self for ind in e.to_slice())

    def get_eintup_names(self):
        return set(ind for e in self for ind in e.get_eintup_names())

class Assign(AST):
    def __init__(self, lhs, rhs):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def reset(self):
        self.lhs.fill_ndarray(0)

    def evaluate(self):
        self.lhs.add(self.rhs.value())

    # collect the unique set of indices in this assignment statement
    def get_eintup_names(self):
        inds = set()
        inds.update(self.lhs.get_eintup_names())
        inds.update(self.rhs.get_eintup_names())
        return inds

class SizeExpr(AST):
    # the AST for with 's' the name of an ein-tuple,
    # and c another ein-tuple which is evaluated in the expression
    # c is expected to be a length 1 ein-tuple
    def __init__(self, tup, dim_name, ein_tup1d):
        self.tup = tup
        self.dim_name = dim_name
        self.index = ein_tup1d

    def value(self):
        if self.index.length() != 1:
            raise RuntimeError('SizeExpr second arg must be length-1 EinTup')
        ind = self.index.to_slice()[0]
        return self.tup.shape_map[self.dim_name][ind]

    def to_slice(self):
        return (self.value(),)

    def get_eintup_names(self):
        return {self.index.name}

    
class Call(AST):
    def __init__(self, func, index_list_node):
        super().__init__()
        if func == 'RANDOM':
            self.func = np.random.uniform
            self.dtype = np.float32
        elif func == 'RANDINT':
            self.func = np.random.randint
            self.dtype = np.int32
        else:
            raise RuntimeError(f'unknown Call function {func}')

        self.ind_node = index_list_node

    def value(self):
        return self.func(*self.ind_node.value())

    def get_eintup_names(self):
        return self.ind_node.get_eintup_names()

