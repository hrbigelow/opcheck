import tensorflow as tf
import enum
import operator
import re

# check validity of all names in the index_list
def get_signature(tup_or_rval):
    if isinstance(tup_or_rval, str):
        tup = tup_or_rval
        if not re.fullmatch('[a-z]+', tup):
            raise RuntimeError(f'Got invalid index name {tup}')
        return [tup]
    elif isinstance(tup_or_rval, RValueArray):
        rval = tup_or_rval
        sig = rval.signature()
        if len(sig) == 0:
            raise RuntimeError(
                f'Cannot use a non-slice RValueArray as an index')
        return sig

def check_duplicate_use(sig_list):
    seen = set()
    for tup in sig_list:
        if tup in seen:
            raise RuntimeError(
                f'Eintup name \'{tup}\' appears twice in index list')
        seen.add(tup)

STAR = ':'

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

    # call just before evaluation on new rank_map
    def prepare(self):
        for ch in self.children:
            ch.prepare()


    def get_inds(self):
        return {ind for c in self.children for ind in c.get_inds()}

    def split_perm(self, trg_sig, core_sig_unordered):
        core_sig = [ ind for ind in trg_sig if ind in core_sig_unordered ]
        bcast_sig = [ ind for ind in trg_sig if ind not in core_sig ]
        n_core = len(core_sig)
        if n_core != len(core_sig_unordered):
            raise RuntimeError(
                f'split_perm: trg_sig ({trg_sig}) did not '
                f'contain all core indices ({core_sig})')

        src_sig = core_sig + bcast_sig

        # trg_sig[i] = src_sig[perm[i]], maps src to trg 
        perm = [ src_sig.index(ind) for ind in trg_sig ]
        # src_sig[i] = trg_sig[perm[i]], reverse mapping 
        perm_rev = [ trg_sig.index(ind) for ind in src_sig ] 
        return perm, perm_rev, n_core 

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

    # reshape / transpose ten, with starting shape src_sig,
    # to be broadcastable to trg_sig 
    def layout_to_sig(self, ten, src_sig, trg_sig):
        # TODO: This shouldn't be an error - if src_sig has extra, they
        # just need to be marginalized out
        if not set(src_sig).issubset(trg_sig):
            raise RuntimeError(
                f'Source signature must be a subset of target signature.'
                f'Got src_sig {src_sig} and trg_sig {trg_sig}')

        perm, perm_rev, n_core = self.split_perm(trg_sig, src_sig)
        n_bcast = len(trg_sig) - n_core

        # get cardinality of reordered src_sig
        card = self.get_cardinality(*[trg_sig[p] for p in perm_rev])
        trg_dims = self.broadcast_shape(trg_sig, src_sig) 

        if ten.shape != self.flat_dims(src_sig):
            raise RuntimeError(
                f'Tensor shape {ten.shape} not consistent with '
                f'signature shape {self.flat_dims(src_sig)}')

        # condense tensor dim groups to src_sig cardinality 
        ten = tf.reshape(ten, card[:n_core] + [1] * n_bcast)

        # permute to desired target signature
        ten = tf.transpose(ten, perm)

        # expand cardinalities to individual dims
        ten = tf.reshape(ten, trg_dims) 

        return ten



class RandomCall(AST):
    # apply a function pointwise to the materialized arguments
    # args can be: constant or array-like
    def __init__(self, cfg, min_expr, max_expr, dtype_string):
        super().__init__(min_expr, max_expr)
        self.cfg = cfg
        self.dtype_string = dtype_string
        if dtype_string == 'INT':
            self.dtype = tf.int32
        elif dtype_string == 'FLOAT':
            self.dtype = tf.float64
        else:
            raise RuntimeError(f'dtype must be INT or FLOAT, got {dtype_string}')
        self.min_expr = min_expr
        self.max_expr = max_expr

    def __repr__(self):
        return (f'RandomCall({repr(self.min_expr)}, ' +
                f'{repr(self.max_expr)}, {self.dtype_string})')

    def evaluate(self, trg_sig):
        mins = self.min_expr.evaluate(trg_sig)
        mins = tf.cast(mins, self.dtype)
        maxs = self.max_expr.evaluate(trg_sig)
        maxs = tf.cast(maxs, self.dtype)
        trg_dims = self.flat_dims(trg_sig) 
        rnd = tf.random.uniform(trg_dims, 0, 2**31-1, dtype=self.dtype)
        ten = rnd % (maxs - mins) + mins
        return ten

class RangeExpr(AST):
    # Problem: no good way to instantiate 'children' here since
    # the eintup's are just strings
    # RANGE[s, c], with s the key_eintup, and c the 1-D last_eintup
    def __init__(self, cfg, key_eintup, last_eintup):
        super().__init__()
        self.cfg = cfg
        self.key_ind = key_eintup
        self.last_ind = last_eintup

    def __repr__(self):
        return f'RangeExpr({self.key_ind}, {self.last_ind})'

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

    def evaluate(self, trg_sig):
        src_sig = [self.key_ind, self.last_ind]
        ten = [tf.range(e) for e in self.cfg.dims(self.key_ind)]
        ten = tf.meshgrid(*ten, indexing='ij')
        ten = tf.stack(ten, axis=self.cfg.rank(self.key_ind))
        ten = self.layout_to_sig(ten, src_sig, trg_sig)
        return ten

class ArrayBinOp(AST):
    def __init__(self, cfg, lhs, rhs, op_string):
        super().__init__(lhs, rhs)
        # TODO: expand to include Rank, IntExpr, and add evaluate()
        # methods to those classes
        allowed_types = (RangeExpr, RandomCall, RValueArray, Dims, IntExpr,
                ArrayBinOp)
        if not isinstance(lhs, allowed_types):
            raise RuntimeError(
                f'left-hand-side argument is type {type(lhs)}, not allowed')
        if not isinstance(rhs, allowed_types):
            raise RuntimeError(
                f'right-hand-side argument is type {type(rhs)}, not allowed')

        self.cfg = cfg
        self.lhs = lhs
        self.rhs = rhs
        # TODO: fix problem with tf.int32 vs tf.float64 tensors
        ops = [ tf.add, tf.subtract, tf.multiply, tf.divide, tf.math.floordiv ]
        self.op = dict(zip(['+', '-', '*', '/', '//'], ops))[op_string]
        self.op_string = op_string

    def __repr__(self):
        return f'ArrayBinOp({repr(self.lhs)} {self.op_string} {repr(self.rhs)})'

    def evaluate(self, trg_sig):
        # TODO: optimize index ordering
        sub_sig = self.get_inds()
        sub_sig = list(sub_sig)
        lval = self.lhs.evaluate(sub_sig)
        rval = self.rhs.evaluate(sub_sig)
        ten = self.op(lval, rval)
        ten = self.layout_to_sig(ten, sub_sig, trg_sig)
        return ten

class RValueArray(AST):
    # Represents a mention of a persistent array on the right-hand-side
    def __init__(self, cfg, name, index_list):
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.index_list = index_list

    def __repr__(self):
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'RValueArray({self.name})[{ind_list}]'

    def get_inds(self):
        # TODO: what to do if this is nested?
        inds = { tup for tup in self.index_list if isinstance(tup, str) }
        return inds


    def signature(self):
        z = zip(self.cfg.array_sig[self.name], self.index_list) 
        return [ tup for tup, call in z if call == STAR ]

    # TODO: marginalize extra indices in src_sig 
    def evaluate(self, trg_sig):
        if self.name not in self.cfg.arrays:
            raise RuntimeError(
                f'RValueArray {self.name} called evaluate() but '
                f'array is not materialized')
        src_sig = self.cfg.array_sig[self.name]
        ten = self.cfg.arrays[self.name]
        ten = self.layout_to_sig(ten, src_sig, trg_sig)
        return ten

class LValueArray(AST):
    # represents an array expression being assigned to
    def __init__(self, cfg, array_name, index_list):
        super().__init__()
        self.name = array_name
        self.cfg = cfg
        self.has_slice_index = False

        # in the first appearance of this array in the list of statements, the
        # array signature is defined from its index_list 
        is_first_use = (array_name not in self.cfg.array_sig)
        sig_list = [ tup for call in index_list for tup in get_signature(call) ]

        check_duplicate_use(sig_list)

        def same_shape(tup1, tup2):
            return self.cfg.tup(tup1).same_shape_as(self.cfg.tup(tup2))

        if is_first_use:
            self.cfg.array_sig[self.name] = sig_list
            for et in sig_list:
                self.cfg.maybe_add_tup(et)
        else:
            z = zip(self.cfg.array_sig[self.name], sig_list)
            bad_pair = next((pair for pair in z if not same_shape(*pair)), None)
            if bad_pair is not None:
                raise RuntimeError(
                    f'Usage of {bad_pair[1]} not the same shape as signature '
                    f'{bad_pair[0]}')

        self.index_list = index_list

    def __repr__(self):
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'LValueArray({self.name})[{ind_list}]'

    def assign(self, rhs):
        if self.has_slice_index:
            raise NotImplementedError
            # need to use tf.scatter
        else:
            trg_inds = self.index_list
            val = rhs.evaluate(trg_inds)
            self.cfg.arrays[self.name] = val

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

    def __repr__(self):
        op_str = '+=' if self.do_accum else '='
        return f'Assign: {repr(self.lhs)} {op_str} {repr(self.rhs)}'

    def evaluate(self):
        if self.do_accum:
            self.lhs.add(self.rhs)
        else:
            self.lhs.assign(self.rhs)

class ScalarExpr(AST):
    def __init__(self, *children):
        super().__init__(*children)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.value()})'

    def evaluate(self, trg_sig):
        src_sig = []
        ten = tf.constant(self.value())
        return self.layout_to_sig(ten, src_sig, trg_sig)

class IntExpr(ScalarExpr):
    def __init__(self, cfg, val):
        super().__init__()
        self.cfg = cfg
        self.val = int(val)

    def value(self):
        return self.val

class Rank(ScalarExpr):
    def __init__(self, cfg, arg):
        super().__init__()
        self.cfg = cfg
        self.ein_arg = arg

    def prepare(self):
        if self.ein_arg not in self.cfg.tups:
            raise RuntimeError(f'Rank arg {self.ein_arg} not a known EinTup')

    def value(self):
        return len(self.cfg.tups[self.ein_arg])

class DimKind(enum.Enum):
    Star = 'Star'
    Int = 'Int' 
    Index = 'Index'

class Dims(AST):
    def __init__(self, cfg, arg, kind, index_expr=None):
        super().__init__()
        self.cfg = cfg
        self.ein_arg = arg
        self.kind = kind

        if self.kind == DimKind.Int:
            self.index = int(index_expr) 
        elif self.kind == DimKind.Index:
            self.index = index_expr

    def __repr__(self):
        return f'Dims({self.ein_arg})[{self.index or ":"}]'

    def prepare(self):
        if self.ein_arg not in self.cfg.tups:
            raise RuntimeError(f'Dims argument \'{self.ein_arg}\' not a known Index')
        if (self.kind == DimKind.Int and 
                self.index >= len(self.cfg.tups[self.ein_arg])):
            raise RuntimeError(f'Dims index \'{self.ind}\' out of bounds')
        if self.kind == DimKind.Index:
            if self.index not in self.cfg.tups:
                raise RuntimeError(
                    f'Dims Index name \'{self.index}\' not known Index')
            if len(self.cfg.tups[self.index]) != 1:
                raise RuntimeError(
                    f'Dims Index index \'{self.index}\' must be '
                    f'rank 1, got \'{len(self.cfg.tups[self.index])}\'')
            if (self.cfg.dims(self.index)[0] >
                    len(self.cfg.dims(self.ein_arg))):
                raise RuntimeError(
                    f'Dims Index index \'{self.index}\' must'
                    f' have values in range of Index argument \'{self.ein_arg}\'.'
                    f' {self.cfg.dims(self.index)[0]} exceeds '
                    f'{len(self.cfg.dims(self.ein_arg))}')

    def evaluate(self, trg_sig):
        if self.kind != DimKind.Index:
            raise RuntimeError(
                f'Only {DimKind.Index.value} Dims can call evaluate()')
        src_sig = self.get_inds()
        ten = tf.constant(self.cfg.dims(self.ein_arg))
        ten = self.layout_to_sig(ten, src_sig, trg_sig)
        return ten

    def value(self):
        if self.kind == DimKind.Index:
            raise RuntimeError(
                f'Cannot call value() on a {DimKind.Index.value} Dims')

        d = self.cfg.dims(self.ein_arg)
        if self.kind == DimKind.Star:
            return d
        elif self.kind == DimKind.Int:
            return d[self.index]
    
    def get_inds(self):
        if self.kind == DimKind.Index:
            return {self.index}
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
        self.op_string = op
        self.op = dict(zip(ops_strs, ops))[op]

    def __repr__(self):
        return (f'LogicalOp({repr(self.arg1)} {self.op_string} ' +
                f'{repr(self.arg2)})')

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

    cfg.maybe_add_tup('batch')
    cfg.maybe_add_tup('slice')
    cfg.maybe_add_tup('coord')
    cfg.set_dims({'batch': 3, 'slice': 3, 'coord': 1})
    cfg.tups['coord'].set_dim(0, 3)
    # rng = RangeExpr(cfg, 'batch', 'coord')
    # rng.prepare()
    # ten = rng.evaluate(['slice', 'batch', 'coord'])
    
    # d1 = Dims(cfg, 'batch', DimKind.Index, 'coord')
    # d2 = Dims(cfg, 'slice', DimKind.Index, 'coord')
    # rk = Rank(cfg, 'batch')
    # rnd = RandomCall(cfg, IntExpr(cfg, 0), rk, 'INT')
    # rnd.prepare()
    # ten = rnd.evaluate(['slice', 'batch', 'coord'])

    print(ten)
    print(ten.shape)
    print(cfg.tups)

