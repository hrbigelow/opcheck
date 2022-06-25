import tensorflow as tf
import enum
import re

def valid_eintup_name(name):
    return re.fullmatch('[a-z]+', name)

class AST(object):
    def __init__(self, *children):
        self.children = list(children)

    def __repr__(self):
        indent_str = '  '
        child_reprs = [repr(c) for c in self.children]
        def indent(mls):
            return '\n'.join(indent_str + l for l in mls.split('\n'))
        this = self.__class__.__name__
        if hasattr(self, 'name'):
            this += f'({self.name})'

        child_repr = '\n'.join(indent(cr) for cr in child_reprs)
        return this + '\n' + child_repr

    def get_inds(self):
        return {ind for c in self.children for ind in c.get_inds()}

    def split_perm(self, trg_inds, core_inds_unordered):
        core_inds = [ ind for ind in trg_inds if ind in core_inds_unordered ]
        bcast_inds = [ ind for ind in trg_inds if ind not in core_inds ]
        n_core = len(core_inds)
        if n_core != len(core_inds_unordered):
            raise RuntimeError(
                    f'split_perm: trg_inds ({trg_inds}) did not '
                    f'contain all core indices ({core_inds})')

        src_inds = core_inds + bcast_inds

        # src_inds[perm[i]] = trg_inds[i]
        perm = [ src_inds.index(ind) for ind in trg_inds ]
        # perm = [ trg_inds.index(ind) for ind in src_inds ] 
        return perm, n_core 

    def flat_dims(self, inds):
        return [ dim for ind in inds for dim in self.cfg.dims(ind) ]

    def get_cardinality(self, *inds):
        return [ self.cfg.nelem(ind) for ind in inds ]

    # core logic for broadcasting
    def broadcast_shape(self, full_inds, core_inds):
        dims = []
        for ind in full_inds:
            if ind in core_inds:
                dims.extend(self.cfg.dims(ind))
            else:
                dims.extend([1] * self.cfg.rank(ind))
        return dims


class RandomCall(AST):
    # apply a function pointwise to the materialized arguments
    # args can be: constant or array-like
    def __init__(self, cfg, min_expr, max_expr, dtype_string):
        super().__init__(min_expr, max_expr)
        self.cfg = cfg
        if dtype_string == 'INT':
            self.dtype = tf.int32
        elif dtype_string == 'FLOAT':
            self.dtype = tf.float64
        else:
            raise RuntimeError(f'dtype must be INT or FLOAT, got {dtype_string}')
        self.min_expr = min_expr
        self.max_expr = max_expr

    def evaluate(self, full_inds):
        core_inds = self.get_inds()
        perm, n_core = self.split_perm(full_inds, core_inds)
        card = self.get_cardinality(*(full_inds[p] for p in perm))
        full_dims = self.flat_dims(full_inds) 

        results = []
        # print(f'core_inds: {core_inds}, bcast_inds: {bcast_inds}')
        for _ in self.cfg.cycle(*core_inds):
            slc = tf.random.uniform(
                    shape=card[n_core:], # materialized broadcast
                    minval=self.min_expr.value(),
                    maxval=self.max_expr.value(),
                    dtype=self.dtype)
            results.append(slc)

        ten = tf.stack(results)
        ten = tf.reshape(ten, card)
        ten = tf.transpose(ten, perm)
        ten = tf.reshape(ten, full_dims) 
        return ten

class RangeExpr(AST):
    # Problem: no good way to instantiate 'children' here since
    # the eintup's are just strings
    # RANGE[s, c], with s the key_eintup, and c the 1-D last_eintup
    def __init__(self, cfg, key_eintup, last_eintup):
        super().__init__(key_eintup, last_eintup)
        self.cfg = cfg
        self.key_ind = key_eintup
        self.last_ind = last_eintup

    def prepare(self):
        if self.cfg.rank(self.last_ind) != 1:
            raise RuntimeError(f'RangeExpr: last EinTup \'{self.last_ind}\''
                    f' must have rank 1.  Got {self.cfg.rank(self.last_ind)}')
        if self.cfg.dims(self.last_ind)[0] != self.cfg.rank(self.key_ind):
            raise RuntimeError(f'RangeExpr: last EinTup \'{self.last_ind}\''
                    f' must have dimension equal to rank of key EinTup '
                    f'\'{self.key_ind}\'')

    def get_inds(self):
        return {self.key_ind, self.last_ind}

    def evaluate(self, trg_inds):
        core_inds_unordered = self.get_inds()
        perm, n_core = self.split_perm(trg_inds, core_inds_unordered)
        n_inds = len(trg_inds)
        n_bcast = n_inds - n_core
        src_inds = [None] * n_inds 
        for i in range(n_inds):
            src_inds[perm[i]] = trg_inds[i]

        core_inds = src_inds[:n_core]
        card = self.get_cardinality(*core_inds)
        ranges = [tf.range(e) for e in self.cfg.dims(self.key_ind)]
        ranges = tf.meshgrid(*ranges, indexing='ij')

        trg_dims = self.broadcast_shape(trg_inds, core_inds)

        # ndrange.shape = DIMS(self.key_ind) + DIMS(self.last_ind)
        # these two should also be consecutive in trg_inds
        ndrange = tf.stack(ranges, axis=self.cfg.rank(self.key_ind))
        ndrange = tf.reshape(ndrange, card + [1] * n_bcast)
        ndrange = tf.transpose(ndrange, perm)
        ndrange = tf.reshape(ndrange, trg_dims)
        return ndrange

class ArrayBinOp(AST):
    def __init__(self, lhs, rhs, op_string):
        super().__init__(lhs, rhs)
        # TODO: expand to include Dims, Rank, IntExpr, and add evaluate()
        # methods to those classes
        assert(isinstance(lhs, (RangeExpr, RandomCall, RightArray)))
        assert(isinstance(rhs, (RangeExpr, RandomCall, RightArray)))
        self.lhs = lhs
        self.rhs = rhs
        ops = [ tf.add, tf.subtract, tf.multiply, tf.divide ]
        self.op = dict(zip('+-*/', ops))[op_string]

    def evaluate(self):
        # TODO: optimize index ordering
        trg_inds = self.get_inds()
        lval = self.lhs.evaluate(trg_inds)
        rval = self.rhs.evaluate(trg_inds)
        return self.op(lval, rval)

class RValueArray(AST):
    # Represents a mention of a persistent array on the right-hand-side
    def __init__(self, cfg, name, index_list):
        super().__init__(*index_list)
        self.cfg = cfg
        self.name = name
        self.index_list = index_list

    def evaluate(self, trg_inds):
        pass

class LValueArray(AST):
    # represents an array expression being assigned to
    def __init__(self, cfg, array_name, index_list):
        super().__init__(*index_list)
        self.name = array_name
        self.cfg = cfg
        self.has_slice_index = False
        is_first_init = (array_name not in self.cfg.array_sig)

        # check validity of all names in the index_list
        seen = set()
        for et in index_list:
            if not isinstance(et, str):
                if is_first_init:
                    raise RuntimeError(
                        f'first initialization of \'{array_name}\' must have '
                        f'an index list of only EinTup names.  Got \'{et}\'')
                continue
            if not valid_eintup_name(et):
                raise RuntimeError(
                    f'first mention of array \'{array_name}\' has invalid eintup '
                    f'name \'{et}\'')
            if et in seen:
                raise RuntimeError(
                    f'Eintup name \'{et}\' appears twice in index list')
            seen.add(et)

        if is_first_init:
            self.cfg.array_sig = index_list
            for et in index_list:
                self.cfg.maybe_add_tup(et)

        else:
            # check for consistent shapes.  only an eintup or an RValueArray
            # slice is allowed
            for sig, call in zip(self.cfg.array_sig[name], index_list):
                if isinstance(call, RValueArray):
                    self.has_slice_index = True
                    # check consistent shape.  how to check for out-of-bounds?
                    pass
                else:
                    if call not in self.cfg.tups:
                        raise RuntimeError(
                            f'LValueArray index expression \'{call}\' not '
                            f'an existing EinTup name, but this is not '
                            'the first array call')
                    if not self.cfg.tup(call).same_shape_as(self.cfg.tup(sig)):
                        raise RuntimeError(
                            f'Signature of array \'{self.name}\' is \'{sig}\''
                            f' but attempting to use \'{call}\' in place of it'
                            f' which does not have the same shape')

        self.index_list = index_list

    def assign(self, rhs):
        if self.has_slice_index:
            raise NotImplementedError
            # need to use tf.scatter
        else:
            trg_inds = self.index_list
            val = rhs.evaluate(trg_inds)
            self.cfg.arrays[self.array_name] = val

    def add(self, rhs):
        if self.array_name not in self.cfg.arrays:
            raise RuntimeError(
                f'Cannot do += on first mention of array \'{self.array_name}\''
                )

        if self.has_slice_index:
            raise NotImplementedError
            # use tf.scatter
        else:
            trg_inds = self.index_list
            val = rhs.evaluate(trg_inds)
            prev = self.cfg.arrays[self.array_name]
            self.cfg.arrays[self.array_name] = tf.add(prev, val)

class Assign(AST):
    def __init__(self, lhs, rhs, do_accum=False):
        super().__init__(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs
        self.do_accum = do_accum

    def evaluate(self):
        if self.do_accum:
            self.lhs.add(self.rhs)
        else:
            self.lhs.assign(self.rhs)


class IntExpr(AST):
    def __init__(self, val):
        super().__init__()
        self.val = int(val)

    def value(self):
        return self.val


class Rank(AST):
    def __init__(self, cfg, arg):
        super().__init__(arg)
        self.cfg = cfg
        self.ein_arg = arg

    def prepare(self):
        if self.ein_arg not in self.cfg.tups:
            raise RuntimeError(f'Rank arg {self.ein_arg} not a known EinTup')

    def value(self):
        return len(self.cfg.tups[self.ein_arg])

class DimKind(enum.Enum):
    Star = 0
    Int = 1
    EinTup = 2

class Dims(AST):
    def __init__(self, cfg, arg, ind_expr):
        super().__init__()
        self.cfg = cfg
        self.ein_arg = arg
        if ind_expr == ':':
            self.kind = DimKind.Star
        elif isinstance(ind_expr, int):
            self.kind = DimKind.Int
            self.index = ind_expr
        elif isinstance(ind_expr, str):
            self.kind = DimKind.EinTup
            self.ein_ind = ind_expr
        else:
            raise RuntimeError(f'index expression must be int, \:\, or EinTup')

    def __repr__(self):
        if self.kind == DimKind.Star:
            ind_str = ':'
        elif self.kind == DimKind.Int:
            ind_str = str(self.index)
        else:
            ind_str = self.ein_ind
        return f'{self.kind} Dims({self.ein_arg})[{ind_str}]'

    def prepare(self):
        if self.ein_arg not in self.cfg.tups:
            raise RuntimeError(f'Dims argument \'{self.ein_arg}\' not a known EinTup')
        if (self.kind == DimKind.Int and 
                self.index >= len(self.cfg.tups[self.ein_arg])):
            raise RuntimeError(f'Dims index \'{self.ind}\' out of bounds')
        if self.kind == DimKind.EinTup:
            if self.ein_ind not in self.cfg.tups:
                raise RuntimeError(f'Dims EinTup name \'{self.ind}\' not known EinTup')
            if len(self.cfg.tups[self.ein_ind]) != 1:
                raise RuntimeError(f'Dims EinTup index \'{self.ein_ind}\' must be '
                        f'rank 1, got \'{len(self.cfg.tups[self.ein_ind])}\'')
            if (self.cfg.dims(self.ein_ind)[0] >
                    len(self.cfg.dims(self.ein_arg))):
                raise RuntimeError(f'Dims EinTup index \'{self.ein_ind}\' must'
                f' have values in range of EinTup argument \'{self.ein_arg}\'.'
                f' {self.cfg.dims(self.ein_ind)[0]} exceeds '
                f'{len(self.cfg.dims(self.ein_arg))}')

    def value(self):
        d = self.cfg.dims(self.ein_arg)
        if self.kind == DimKind.Star:
            return d
        elif self.kind == DimKind.Int:
            return d[self.index]
        else:
            ein_val = self.cfg.tups[self.ein_ind].value()
            return d[ein_val[0]]
    
    def get_inds(self):
        if self.kind == DimKind.EinTup:
            return {self.ein_ind}
        else:
            return set()

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

class TensorArg(AST):
    def __init__(self, cfg, name):
        if name not in cfg.arrays:
            raise RuntimeError(f'argument must be array name, got {name}')
        self.cfg = cfg
        self.name = name

    def value(self):
        return self.cfg.arrays[self.name]


if __name__ == '__main__':
    import config
    cfg = config.Config(5, 10)
    """
    cfg.set_ranks({'batch': 2, 'slice': 3, 'coord': 1})
    cfg.tups['coord'].dims[0] = 3  
    dims = Dims(cfg, 'slice', 'coord')
    dims.prepare()
    rc = RandomCall(cfg, IntExpr(0), dims, 'INT')
    ten = rc.evaluate(['slice', 'coord'])
    # print(ten)
    # print(ten.shape)
    # print(cfg.tups)
    """

    cfg.set_ranks({'batch': 3, 'slice': 3, 'coord': 1})
    cfg.tups['coord'].dims[0] = 3  
    rng = RangeExpr(cfg, 'batch', 'coord')
    rng.prepare()
    ten = rng.evaluate(['slice', 'batch', 'coord'])
    print(ten)
    print(ten.shape)
    print(cfg.tups)

