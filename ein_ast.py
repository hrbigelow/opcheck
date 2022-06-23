import tensorflow as tf
import numpy as np
import operator

class OutOfBoundsError(Exception):
    pass
    
class AST(object):
    def __init__(self, *children):
        # print(f'in AST init with {children}')
        self.children = list(children)

    # call once, just after parsing program
    def add_range_constraints(self, constraints):
        for ch in self.children:
            ch.add_range_constraints(constraints)

    # call just before evaluation on new rank_map
    def prepare(self):
        for ch in self.children:
            ch.prepare()

    # returns a set of all 'live' eintup indices in this expression.
    # 'live' means the values of the eintup indices are being instantiated.
    def live_indices(self):
        return {ind for c in self.children for ind in c.live_indices()}

    def is_static(self):
        return len(self.live_indices()) == 0

    # returns a single array element as a scalar.  Only implemented
    # for ArraySlice and Call, to be used in a right-hand-side assignment
    def element(self):
        raise NotImplementedError

    # for returns a tuple of instantiated values of the live indices.
    def to_slice(self):
        raise NotImplementedError


class IntNode(AST):
    def __init__(self, cfg, val):
        super().__init__()
        self.cfg = cfg
        self.val = int(val)
        self.name = None

    # call this if this IntNode should be used in the role of the
    # `name` EinTup.  For example, in the expression DIMS(s) + 1,
    # the '1' is in the role of EinTup s, and we call set_name(s)
    def set_name(self, name):
        self.name = name

    def rank(self):
        if self.name is None:
            return 1
        else:
            return self.cfg.rank(self.name)

    def value(self):
        if self.name is None:
            return self.val
        else:
            return (self.val,) * self.cfg.rank(self.name)

    def to_slice(self):
        if self.name is None:
            return (self.val,)
        else:
            return (self.val,) * self.cfg.rank(self.name)

    def live_indices(self):
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

    def live_indices(self):
        return set()

class ScalarBinOp(AST):
    def __init__(self, arg1, arg2, op):
        super().__init__(arg1, arg2)
        if not (arg1.is_scalar and arg2.is_scalar):
            raise RuntimeError('ScalarBinOp can only operate on two scalars')
        self.arg1 = arg1
        self.arg2 = arg2
        opfuncs = [ operator.add, operator.sub, operator.mul, operator.truediv ]
        self.op = dict(zip('+-*/', opfuncs))[op]

    def element(self):
        return self.op(self.arg1.element(), self.arg2.element())

class StaticBinOpBase(AST):
    # Accepts Rank, Dims, DimsAccess expressions.  
    def __init__(self, arg1, arg2):
        super().__init__(arg1, arg2)
        self.arg1 = arg1
        self.arg2 = arg2
        if isinstance(self.arg1, Dims):
            self.set_name(self.arg1.get_name())
        if isinstance(self.arg2, Dims):
            self.set_name(self.arg2.get_name())

    def set_name(self, name):
        if not isinstance(self.arg1, Dims):
            self.arg1.set_name(name)
        if not isinstance(self.arg2, Dims):
            self.arg2.set_name(name)

    def _same_kind(self, vals1, vals2):
        return isinstance(vals1, (tuple, list)) == isinstance(vals2, (tuple, list))

    def value(self):
        raise NotImplementedError

class ArithmeticBinOp(StaticBinOpBase):
    def __init__(self, arg1, arg2, op):
        super().__init__(arg1, arg2)
        opfuncs = [ operator.add, operator.sub, operator.mul, operator.truediv,
                operator.floordiv ]
        self.op = dict(zip(['+', '-', '*', '/', '//'], opfuncs))[op]

    def prepare(self):
        super().prepare()
        if self.arg1.rank() != self.arg2.rank():
            raise RuntimeError(f'Dims sizes must match for ArithmeticBinOp.'
                    f'Got {self.arg1.value()} and {self.arg2.value()}')
    def rank(self):
        # arg1 and arg2 must have same rank
        return self.arg1.rank()

    def value(self):
        vals1 = self.arg1.value()
        vals2 = self.arg2.value()
        if not self._same_kind(vals1, vals2):
            raise RuntimeError('ArithmeticBinOp got list and scalar')
        if isinstance(vals1, (tuple, list)):
            return tuple(map(self.op, vals1, vals2))
        else:
            return self.op(vals1, vals2)

class LogicalOp(StaticBinOpBase):
    def __init__(self, arg1, arg2, op):
        super().__init__(arg1, arg2)
        ops = [ operator.lt, operator.le, operator.eq, operator.ge, operator.gt
                ]
        ops_strs = [ '<', '<=', '==', '>=', '>' ]
        self.op = dict(zip(ops_strs, ops))[op]

    def value(self):
        vals1 = self.arg1.value()
        vals2 = self.arg2.value()
        if self.arg1.rank() != self.arg2.rank():
            return False
        if not self._same_kind(vals1, vals2):
            return False
        if isinstance(vals1, (tuple, list)):
            return all(self.op(v1, v2) for v1, v2 in zip(vals1, vals2))
        else:
            return self.op(vals1, vals2)

class RangeConstraint(AST):
    def __init__(self, eintup_binop, kind, value_expr):
        super().__init__(value_expr)
        self.signature_string = eintup_binop.signature()
        self.kind = kind
        self.value_expr = value_expr
        self.value_expr.set_name(eintup_binop.lhs.signature())

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

    def rank(self):
        return self.cfg.rank(self.name)

    def min(self):
        return (0,) * self.rank()

    def max(self):
        return tuple(m - 1 for m in self.cfg.dims(self.name))

    def to_slice(self):
        return self.cfg.value(self.name)

    def live_indices(self):
        return {self.name}

class EinTupBinOp(AST):
    def __init__(self, lhs, rhs, op):
        super().__init__(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs
        opfuncs = [ operator.add, operator.sub ]
        self.op_string = op
        self.op = dict(zip('+-', opfuncs))[op]
        self.min_expr = lambda : (-float('inf'),) * self.rank()
        self.max_expr = lambda : (float('inf'),) * self.rank()

    def signature(self):
        return self.lhs.signature() + self.op_string + self.rhs.signature()

    def rank(self):
        # both lhs and rhs have same rank
        return self.lhs.rank()

    def min(self):
        return self._min

    def max(self):
        return self._max

    # called once, just after program parsing 
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

    # call just before running the program with new rank settings
    def prepare(self):
        super().prepare()
        if self.min_expr is not None:
            self.min_expr.prepare()

        if self.max_expr is not None:
            self.max_expr.prepare()

        lmin, lmax = self.lhs.min(), self.lhs.max()
        rmin, rmax = self.rhs.min(), self.rhs.max()

        min_cons = self.min_expr.value()
        max_cons = self.max_expr.value()

        def maxop(cons, l, r): return max(cons, self.op(l, r))
        def minop(cons, l, r): return min(cons, self.op(l, r))

        if self.op_string == '+':
            self._min = tuple(map(maxop, min_cons, lmin, rmin)) 
            self._max = tuple(map(minop, max_cons, lmax, rmax))
        elif self.op_string == '-':
            self._min = tuple(map(maxop, min_cons, lmin, rmax))
            self._max = tuple(map(minop, max_cons, lmax, rmin))

        if any(l > h for l, h in zip(self._min, self._max)):
            raise RuntimeError('invalid min/max range for EinTupBinOp')

    def to_slice(self):
        slc = tuple(map(self.op, self.lhs.to_slice(), self.rhs.to_slice()))
        if any(map(operator.lt, self._min, slc)):
            raise OutOfBoundsError
        if any(map(operator.gt, self._max, slc)):
            raise OutOfBoundsError
        return slc

class ArraySlice(AST):
    def __init__(self, array, index_list):
        super().__init__(index_list)
        self.array = array
        self.index_list = index_list
        self.is_scalar = True
        for this_node, orig_node in zip(self.index_list, self.array.index_list):
            if isinstance(this_node, StarNode):
                self.is_scalar = False
                this_node.set_name(orig_node.signature())

    def maybe_convert(self, dtype):
        self.array.maybe_convert(dtype)

    def _ind(self):
        return self.index_list.to_slice()

    # return a scalar valued single element of the array
    def element(self):
        if not self.is_scalar:
            raise RuntimeError(
            'Cannot call element on non-scalar ArraySlice'
            )
        return self.array[self._ind()]

    def add(self, rhs):
        # print(f'ArraySlice::add {self.name}[{ind}]')
        self.array[self._ind()] += rhs

    def assign(self, rhs):
        self.array[self._ind()] = rhs

    def to_slice(self):
        flat = self.array[self._ind()].squeeze()
        if flat.ndim != 1:
            raise RuntimeError('to_slice requires a flatten-able array')
        return tuple(flat.tolist())

# polymorphic list of StarNode, EinTup, EinTupBinOp, IntNode, and ArraySlice
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

    def to_slice(self):
        return tuple(ind for e in self for ind in e.to_slice())

    def live_indices(self):
        return set(ind for e in self for ind in e.live_indices())

class Assign(AST):
    def __init__(self, cfg, lhs, rhs, fill_zero):
        super().__init__(lhs, rhs)
        self.cfg = cfg
        self.lhs = lhs
        self.rhs = rhs
        self.fill_zero = fill_zero

    def prepare(self):
        super().prepare()
        self.lhs.array.update_dims()
        if self.fill_zero:
            self.lhs.array.fill(0)

    def evaluate(self):
        indices = self.live_indices()
        self.cfg.tup.set_indices(indices)
        self.prepare()
        while self.cfg.tup.advance():
            try:
                self.lhs.add(self.rhs.element())
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

        self.index_list = index_list_node

    def element(self):
        args = self.index_list.to_slice()
        v = self.func(*args)
        # print(f'{self.func}({args}) -> {v}')
        return v

class Rank(AST):
    def __init__(self, cfg, ind_name):
        super().__init__()
        self.cfg = cfg
        self.ind_name = ind_name

    def rank(self):
        return 1

    def value(self):
        return self.cfg.rank(self.ind_name)

    def live_indices(self):
        return set() 

class Dims(AST):
    def __init__(self, cfg, ind_name):
        super().__init__()
        self.cfg = cfg
        self.ind_name = ind_name

    def get_name(self):
        return self.ind_name

    def rank(self):
        return self.cfg.rank(self.ind_name)

    def value(self):
        return self.cfg.dims(self.ind_name)

    def live_indices(self):
        return set()

class DimsAccess(AST):
    # DIMS(a)[0] for example
    def __init__(self, cfg, ind, pos):
        super().__init__()
        self.cfg = cfg
        self.ind = ind
        self.pos = int(pos)

    def rank(self):
        return 1

    def value(self):
        if self.cfg.rank(self.ind) != 1:
            raise RuntimeError('DimsAccess second arg must be length-1')
        v = self.cfg.dims(self.ind)[self.pos]
        # print(f'returning {v} from DimsAccess')
        return v

    def live_indices(self):
        return set()

class SizeExpr(AST):
    # Represents DIMS(a)[b] for example.  In this example, 'a' is not
    # part of the indices for live_indices because it only identifies the
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

    def live_indices(self):
        return {self.live_ind}

class TensorArg(AST):
    def __init__(self, cfg, name):
        if name not in cfg.arrays:
            raise RuntimeError(f'argument must be array name, got {name}')
        self.cfg = cfg
        self.name = name

    def value(self):
        return tf.convert_to_tensor(self.cfg.arrays[self.name].ary)

