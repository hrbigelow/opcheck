import operator
import math

class Broadcastable(object):
    """
    Types
    bscalar: Broadcastable with val a scalar
    blist: Broadcastable with val a list

    Rules
    bscalar op (scalar|bscalar) -> bscalar
    bscalar op (list|blist) -> blist (broadcast the scalar)
    blist op (scalar|bscalar) -> blist (broadcast the scalar)
    blist op (list|blist) -> (raise if unequal length), return list

    scalars are broadcast to the appropriate length, then the operation
    is done element-wise.
    """
    def __init__(self, int_or_intlist):
        self.val = int_or_intlist

    def __repr__(self):
        return f'Broadcastable({self.val})'

    @staticmethod
    def getval(obj):
        scalar = not Broadcastable.islist(obj)
        if isinstance(obj, Broadcastable):
            v = obj.val
        else:
            v = obj
        if scalar:
            v = [v]
        return v

    @staticmethod
    def islist(obj):
        return isinstance(obj, list) or (
                isinstance(obj, Broadcastable) and
                isinstance(obj.val, list))

    @staticmethod
    def _ceildiv_op(a, b):
        return math.ceil(a / b)

    def _op(self, oper, o):
        a_is_list = Broadcastable.islist(self)
        b_is_list = Broadcastable.islist(o)
        aval = Broadcastable.getval(self)
        bval = Broadcastable.getval(o)
        if a_is_list and b_is_list:
            if len(aval) != len(bval):
                raise RuntimeError(
                    f'Broadcastable got unequal length lists '
                    f'{aval} and {bval}')
            else:
                return Broadcastable([oper(a, b) for a, b in zip(aval, bval)])
        elif a_is_list:
            return Broadcastable([oper(a, bval[0]) for a in aval])
        elif b_is_list:
            return Broadcastable([oper(aval[0], b) for b in bval])
        else:
            return Broadcastable(oper(aval[0], bval[0]))

    def __add__(self, o):
        return self._op(operator.add, o)

    def __sub__(self, o):
        return self._op(operator.sub, o)

    def __mul__(self, o):
        return self._op(operator.mul, o)

    def __floordiv__(self, o):
        return self._op(operator.floordiv, o)

    def ceildiv(self, o):
        return self._op(Broadcastable._ceildiv_op, o)

