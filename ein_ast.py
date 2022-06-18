import operator
import numpy as np
    
class AST(object):
    def __init__(self, *args):
        pass

class IntNode(AST):
    def __init__(self, val):
        super().__init__()
        self.val = int(val)

    def value(self):
        return self.val

    def to_slice(self):
        return (self.val,)

    def get_indices(self):
        return set()

class StarNode(AST):
    # Represents a wildcard slice
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = None

    def set_name(self, name):
        self.name = name

    def to_slice(self):
        return tuple((None,)) * self.cfg.rank(self.name)

    def get_indices(self):
        return set()

class SliceBinOp(AST):
    def __init__(self, slice1, slice2, op):
        self.slice1 = slice1
        self.slice2 = slice2
        opfuncs = [ operator.add, operator.sub, operator.mul, operator.truediv ]
        self.op = dict(zip('+-*/', opfuncs))[op]

    def value(self):
        return self.op(self.slice1.value(), self.slice2.value())

    def get_indices(self):
        return set.union(
                self.slice1.get_indices(),
                self.slice2.get_indices())
        

class EinTup(AST):
    # beg, end denote the position in the main tuple
    def __init__(self, cfg, name):
        super().__init__()
        self.cfg = cfg 
        self.name = name

    def length(self):
        return self.cfg.rank(self.name)

    def to_slice(self):
        return self.cfg.value(self.name)

    def get_indices(self):
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

    def get_indices(self):
        return self.ind_node.get_indices()

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

    def get_indices(self):
        return set(ind for e in self for ind in e.get_indices())

class Assign(AST):
    def __init__(self, lhs, rhs):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def evaluate(self):
        self.lhs.add(self.rhs.value())

    # collect the unique set of indices in this assignment statement
    def get_indices(self):
        inds = set()
        inds.update(self.lhs.get_indices())
        inds.update(self.rhs.get_indices())
        return inds

    
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

    def get_indices(self):
        return self.ind_node.get_indices()

class Rank(AST):
    def __init__(self, cfg, ind_name):
        self.cfg = cfg
        self.ind_name = ind_name

    def value(self):
        return self.cfg.rank(self.ind_name)

    def get_indices(self):
        return {self.ind_name}

class DimsAccess(AST):
    # DIMS(a)[0] for example
    def __init__(self, cfg, ind, pos):
        self.cfg = cfg
        self.ind = ind
        self.pos = int(pos)

    def value(self):
        if self.cfg.rank(self.ind) != 1:
            raise RuntimeError('DimsAccess second arg must be length-1')
        return self.cfg.shape(self.ind)[self.pos]

    def get_indices(self):
        return {self.ind}

class SizeExpr(AST):
    # Represents DIMS(a)[b] for example
    def __init__(self, cfg, ind1, ind2):
        self.cfg = cfg
        self.ind1 = ind1
        self.ind2 = ind2

    def value(self):
        if self.cfg.rank(self.ind2) != 1:
            raise RuntimeError('SizeExpr second arg must be rank-1 EinTup')
        ind = self.cfg.value(self.ind2)[0] # should be a length-1 tuple
        return self.cfg.shape(self.ind1)[ind]

    def to_slice(self):
        return (self.value(),)

    def get_indices(self):
        return {self.ind1, self.ind2}

class LogicalOp(AST):
    def __init__(self, lhs, rhs, op):
        self.lhs = lhs
        self.rhs = rhs
        ops = [ operator.lt, operator.le, operator.eq, operator.ge, operator.gt
                ]
        ops_strs = [ '<', '<=', '==', '>=', '>' ]
        self.op = dict(zip(ops_strs, ops))[op]
        
    def value(self):
        return self.op(self.lhs.value(), self.rhs.value())

    def get_indices(self):
        return set.union(
                self.lhs.get_indices(),
                self.rhs.get_indices())


class ShapeAccessBinOp(AST):
    def __init__(self, arg1, arg2, op):
        self.arg1 = arg1
        self.arg2 = arg2
        ops = [ operator.add, operator.sub ]
        self.op = dict(zip('+-', ops))[op]

    def value(self):
        return self.op(self.arg1.value(), self.arg2.value())

    def get_indices(self):
        return set.union(
                self.arg1.get_indices(),
                self.arg2.get_indices())


