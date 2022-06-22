import tensorflow as tf
import numpy as np
import operator

class OutOfBoundsError(Exception):
    pass
    
class AST(object):
    def __init__(self, *children):
        # print(f'in AST init with {children}')
        self.children = list(children)

    def add_range_constraints(self, constraints):
        for ch in self.children:
            ch.add_range_constraints(constraints)

    def get_indices(self):
        return {ind for c in self.children for ind in c.get_indices()}

    # if true, only evaluate at compile time.  if false,
    # evaluate the op for every EinTup iteration for relevant
    # statements.
    def is_static(self):
        return len(self.get_indices()) == 0


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
        super().__init__()
        self.cfg = cfg
        self.name = None

    def set_name(self, name):
        self.name = name

    def to_slice(self):
        return tuple((None,)) * self.cfg.rank(self.name)

    def get_indices(self):
        return set()

class ScalarBinOp(AST):
    def __init__(self, arg1, arg2, op):
        super().__init__(arg1, arg2)
        self.arg1 = arg1
        self.arg2 = arg2
        opfuncs = [ operator.add, operator.sub, operator.mul, operator.truediv ]
        self.op = dict(zip('+-*/', opfuncs))[op]

    def value(self):
        return self.op(self.arg1.value(), self.arg2.value())

class StaticBinOp(AST):
    def __init__(self, arg1, arg2, op):

        super().__init__(arg1, arg2)
        self.arg1 = arg1
        self.arg2 = arg2
        opfuncs = [ operator.add, operator.sub, operator.mul, operator.truediv,
                operator.floordiv ]
        self.op = dict(zip(['+', '-', '*', '/', '//'], opfuncs))[op]

    def value(self):
        if (isinstance(self.arg1, Dims) 
                and isinstance(self.arg2, Dims) 
                and len(self.arg1.value()) != len(self.arg2.value())
                ):
            raise RuntimeError(f'Dims sizes must match for StaticBinOp.'
                    f'Got {self.arg1.value()} and {self.arg2.value()}')
        vals1 = self.arg1.value()
        vals2 = self.arg2.value()
        sz1 = len(vals1)
        sz2 = len(vals2)
        if sz1 != sz2 and sz1 != 1 and sz2 != 1:
            raise RuntimeError(f'StaticBinOp only supports simple broadcasting'
                    f' with lengths == 1 and != 1.  Got {sz1} and {sz2}'
                    f' (values {vals1} and {vals2}')
        if sz1 == 1:
            vals1 = [vals1[0]] * sz2
        elif sz2 == 1:
            vals2 = [vals2[0]] * sz1

        return tuple(map(self.op, vals1, vals2))

class RangeConstraint(AST):
    def __init__(self, signature_string, kind, value_expr):
        super().__init__(value_expr)
        self.signature_string = signature_string
        self.kind = kind
        self.value_expr = value_expr

    def value(self):
        return self.value_expr.value()

class EinTup(AST):
    # beg, end denote the position in the main tuple
    def __init__(self, cfg, name):
        super().__init__()
        self.cfg = cfg 
        self.name = name

    def signature(self):
        return self.name

    def min(self):
        return (0,) * self.cfg.rank(self.name)

    def max(self):
        return tuple(m - 1 for m in self.cfg.dims(self.name))

    def to_slice(self):
        return self.cfg.value(self.name)

    def get_indices(self):
        return {self.name}

class EinTupBinOp(AST):
    def __init__(self, lhs, rhs, op):
        super().__init__(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs
        opfuncs = [ operator.add, operator.sub ]
        self.op_string = op
        self.op = dict(zip('+-', opfuncs))[op]
        self.min_expr = None
        self.max_expr = None

    def signature(self):
        return self.lhs.signature() + self.op_string + self.rhs.signature()

    def min(self):
        if self.op_string == '+':
            return tuple(map(self.op, self.lhs.min(), self.rhs.min()))
        elif self.op_string == '-':
            return tuple(map(self.op, self.lhs.min(), self.rhs.max()))

    def max(self):
        if self.op_string == '+':
            return tuple(map(self.op, self.lhs.max(), self.rhs.max()))
        elif self.op_string == '-':
            return tuple(map(self.op, self.lhs.max(), self.rhs.min()))

    def add_range_constraints(self, constraints):
        # do not recurse to non-top-level EinTupBinOps
        for c in constraints:
            if not isinstance(c, RangeConstraint):
                continue
            if c.signature_string != self.signature():
                continue
            if c.kind == 'MIN':
                self.min_expr = c.value_expr
            elif c.kind == 'MAX':
                self.max_expr = c.value_expr

    # Fix IntNode broadcasting issue
    def to_slice(self):
        slc = tuple(map(self.op, self.lhs.to_slice(), self.rhs.to_slice()))
        if self.min_expr is not None:
            if any(map(operator.lt, self.min_expr.value(), slc)):
                raise OutOfBoundsError
        if self.max_expr is not None:
            if any(map(operator.gt, self.max_expr.value(), slc)):
                raise OutOfBoundsError
        return slc

class ArraySlice(AST):
    def __init__(self, array, index_list):
        super().__init__(index_list)
        self.array = array
        self.ind_node = index_list
        for this_node, orig_node in zip(self.ind_node, self.array.index_list):
            if isinstance(this_node, StarNode):
                this_node.set_name(orig_node.signature())

    def maybe_convert(self, dtype):
        self.array.maybe_convert(dtype)

    # returns a new ndarray
    def value(self):
        return self.array[self.ind_node.value()]

    def add(self, rhs):
        ind = self.ind_node.value()
        # print(f'ArraySlice::add {self.name}[{ind}]')
        self.array[ind] += rhs

    def assign(self, rhs):
        ind = self.ind_node.value()
        self.array[ind] = rhs

    def to_slice(self):
        flat = self.value().squeeze()
        if flat.ndim != 1:
            raise RuntimeError('to_slice requires a flatten-able array')
        return tuple(flat.tolist())

# polymorphic list of EinTup, EinTupBinOp, IntNode, and ArraySlice
# instances
class IndexList(AST):
    def __init__(self, *args):
        # print(f'in IndexList init with {args}')
        super().__init__(*args)

    def __iter__(self):
        return iter(self.children)

    def append(self, node):
        self.children.append(node)

    def min(self):
        return tuple(m for t in self for m in t.min())

    def max(self):
        return tuple(m for t in self for m in t.max())

    def value(self):
        return tuple(ind for e in self for ind in e.to_slice())

    def get_indices(self):
        return set(ind for e in self for ind in e.get_indices())

class Assign(AST):
    def __init__(self, cfg, lhs, rhs):
        super().__init__(lhs, rhs)
        self.cfg = cfg
        self.lhs = lhs
        self.rhs = rhs

    def evaluate(self):
        indices = self.get_indices()
        self.cfg.tup.set_indices(indices)
        while self.cfg.tup.advance():
            try:
                self.lhs.add(self.rhs.value())
            except OutOfBoundsError:
                pass
    
class Call(AST):
    def __init__(self, func, index_list_node):
        super().__init__(index_list_node)
        if func == 'RANDOM':
            self.func = np.random.uniform
            self.dtype = np.float64
        elif func == 'RANDINT':
            self.func = np.random.randint
            self.dtype = np.int32
        else:
            raise RuntimeError(f'unknown Call function {func}')

        self.ind_node = index_list_node

    def value(self):
        args = self.ind_node.value()
        v = self.func(*args)
        # print(f'{self.func}({args}) -> {v}')
        return v

class Rank(AST):
    def __init__(self, cfg, ind_name):
        super().__init__()
        self.cfg = cfg
        self.ind_name = ind_name

    def value(self):
        return self.cfg.rank(self.ind_name)

    def get_indices(self):
        return set() 

class Dims(AST):
    def __init__(self, cfg, ind_name):
        super().__init__()
        self.cfg = cfg
        self.ind_name = ind_name

    def value(self):
        return self.cfg.dims(self.ind_name)

    def get_indices(self):
        return set()

class DimsAccess(AST):
    # DIMS(a)[0] for example
    def __init__(self, cfg, ind, pos):
        super().__init__()
        self.cfg = cfg
        self.ind = ind
        self.pos = int(pos)

    def value(self):
        if self.cfg.rank(self.ind) != 1:
            raise RuntimeError('DimsAccess second arg must be length-1')
        v = self.cfg.dims(self.ind)[self.pos]
        # print(f'returning {v} from DimsAccess')
        return v

    def get_indices(self):
        return set()

class SizeExpr(AST):
    # Represents DIMS(a)[b] for example.  In this example, 'a' is not
    # part of the indices for get_indices because it only identifies the
    # dimensions, not the tuple setting
    def __init__(self, cfg, dims_ind, live_ind):
        super().__init__()
        self.cfg = cfg
        self.dims_ind = dims_ind
        self.live_ind = live_ind

    def value(self):
        if self.cfg.rank(self.live_ind) != 1:
            raise RuntimeError('SizeExpr second arg must be rank-1 EinTup')
        ind = self.cfg.value(self.live_ind)[0] # should be a length-1 tuple
        return self.cfg.dims(self.dims_ind)[ind]

    def to_slice(self):
        return (self.value(),)

    def get_indices(self):
        return {self.live_ind}

class LogicalOp(AST):
    def __init__(self, lhs, rhs, op):
        super().__init__(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs
        ops = [ operator.lt, operator.le, operator.eq, operator.ge, operator.gt
                ]
        ops_strs = [ '<', '<=', '==', '>=', '>' ]
        self.op = dict(zip(ops_strs, ops))[op]

    def value(self):
        return self.op(self.lhs.value(), self.rhs.value())

class TensorArg(AST):
    def __init__(self, cfg, name):
        if name not in cfg.arrays:
            raise RuntimeError(f'argument must be array name, got {p[0]}')
        self.cfg = cfg
        self.name = name

    def value(self):
        return tf.convert_to_tensor(self.cfg.arrays[self.name].ary)

