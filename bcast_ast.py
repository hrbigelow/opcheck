import tensorflow as tf
import enum
import operator
import re
import scatter_gather_funcs as sg

def define_sig(cfg, use_list):
    not_str = next((use for use in use_list if not isinstance(use, str)), None)
    if not_str is not None:
        raise RuntimeError(
            f'define_sig can take only string indices (EinTup names) '
            f'found {not_str}')
    from collections import Counter
    dup = next((use for use, ct in Counter(use_list).items() if ct > 1), None)
    if dup is not None:
        raise RuntimeError(
            f'define_sig must have all distinct indices.  Found duplicate '
            f'\'{dup}\'')

    return [ cfg.maybe_add_tup(use) for use in use_list ]

# Call at instantiation of an Array with established sig
def check_sig(cfg, sig_list, use_list):
    if len(sig_list) != len(use_list):
        raise RuntimeError(
            f'check_sig expected {len(sig_list)} indices but found '
            f'{len(use_list)}')

    nslices = len([use for use in use_list if isinstance(use, Slice)])
    if nslices > 1:
        raise RuntimeError(
            f'check_sig expected 0 or 1 Slice arguments.  Found {nslices}')

    for sig_tup, use in zip(sig_list, use_list):
        if isinstance(use, str):
            use_tup = cfg.maybe_add_tup(use, sig_tup)
            if not sig_tup.same_shape_as(use_tup):
                raise RuntimeError(
                    f'check_sig found incompatible shapes.  Expecting '
                    f'{sig_tup} but found {use_tup}')
        elif isinstance(use, Slice):
            pass # nothing to check during instantiation
        else:
            raise RuntimeError(
                f'check_sig expected string or Slice argument.  Found '
                f'{type(use)}')

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

    def get_tups(self):
        return {tup for c in self.children for tup in c.get_tups()}

    def split_perm(self, trg_sig, core_sig_unordered):
        core_sig = [ tup for tup in trg_sig if tup in core_sig_unordered ]
        bcast_sig = [ tup for tup in trg_sig if tup not in core_sig ]
        n_core = len(core_sig)
        if n_core != len(core_sig_unordered):
            raise RuntimeError(
                f'split_perm: trg_sig ({trg_sig}) did not '
                f'contain all core indices ({core_sig})')

        src_sig = core_sig + bcast_sig

        # trg_sig[i] = src_sig[perm[i]], maps src to trg 
        perm = [ src_sig.index(tup) for tup in trg_sig ]
        # src_sig[i] = trg_sig[perm[i]], reverse mapping 
        perm_rev = [ trg_sig.index(tup) for tup in src_sig ] 
        return perm, perm_rev, n_core 

    def flat_dims(self, tups):
        return [ dim for tup in tups for dim in tup.dims()]

    def get_cardinality(self, tups):
        return [ tup.nelem() for tup in tups ]

    # core logic for broadcasting
    def broadcast_shape(self, full_tups, core_tups):
        dims = []
        for tup in full_tups:
            if tup in core_tups:
                dims.extend(tup.dims())
            else:
                dims.extend([1] * tup.rank())
        return dims

    # reshape / transpose ten, with starting shape src_sig,
    # to be broadcastable to trg_sig 
    def layout_to_sig(self, ten, src_sig, trg_sig):
        if ten.shape != self.flat_dims(src_sig):
            raise RuntimeError(
                f'Tensor shape {ten.shape} not consistent with '
                f'signature shape {self.flat_dims(src_sig)}')

        marg_ex = set(src_sig).difference(trg_sig)
        if len(marg_ex) != 0:
            marg_pos = [ i for i, tup in enumerate(src_sig) if tup in marg_ex ]
            ten = tf.reduce_sum(ten, marg_pos)

        trg_sig = [ tup for tup in trg_sig if tup not in marg_ex ]

        perm, perm_rev, n_core = self.split_perm(trg_sig, src_sig)
        n_bcast = len(trg_sig) - n_core

        # get cardinality of reordered src_sig
        card = self.get_cardinality([trg_sig[p] for p in perm_rev])
        trg_dims = self.broadcast_shape(trg_sig, src_sig) 

        # condense tensor dim groups to src_sig cardinality 
        ten = tf.reshape(ten, card[:n_core] + [1] * n_bcast)

        # permute to desired target signature
        ten = tf.transpose(ten, perm)

        # expand cardinalities to individual dims
        ten = tf.reshape(ten, trg_dims) 

        return ten

class Slice(AST):
    # Represents an expression ary[a,b,c,:,e,...] with exactly one ':' and
    # the rest of the indices simple eintup names.  During the prepare() call,
    # the ':' index must resolve to a RANK 1 index with DIMS equal to the RANK
    # of the place where it is used.  
    def __init__(self, cfg, array_name, index_list):
        super().__init__()
        if array_name not in cfg.array_sig:
            raise RuntimeError(
                f'Cannot instantiate Slice as first appearance of array name '
                f'\'{array_name}\'')
        sig = cfg.array_sig[array_name]
        if len(sig) != len(index_list):
            raise RuntimeError(
                f'Slice instantiated with incorrect number of indices. '
                f'Expecting {len(sig)} but got {len(index_list)}')

        found_star = False
        for sig_tup, call in zip(sig, index_list):
            if call == STAR: 
                if found_star:
                    raise RuntimeError(
                        f'Found a second \':\' index.  Only one wildcard is '
                        f'allowed in a Slice instance')
                found_star = True
                self.star_tup = sig_tup
                continue
            elif not isinstance(call, str):
                raise RuntimeError(
                    f'Slice object only accepts simple tup names or \':\' as '
                    f'indices.  Got \'{call}\'')
            call_tup = cfg.maybe_add_tup(call, shadow_of=sig_tup)
            if not sig_tup.same_shape_as(call_tup):
                raise RuntimeError(
                    f'Slice called with incompatible shape. '
                    f'{call_tup} called in slot of {sig_tup}')

        if not found_star:
            raise RuntimeError(
                f'Slice must contain at least one \':\' in index_list. '
                f'Got \'{index_list}\'') 

        # passed all checks
        self.cfg = cfg
        self.name = array_name
        self.index_list = index_list

    def __repr__(self):
        ind_list = ','.join(self.index_list)
        return f'Slice({self.name})[{ind_list}]'

    def slice_dim(self):
        return self.star_tup.dims()[0]

    def prepare(self):
        rank = self.star_tup.rank() 
        if rank != 1:
            raise RuntimeError(
                f'Slice wildcard index must be rank 1.  Got {rank}')

class Array(AST):
    def __init__(self, cfg, array_name, index_list):
        super().__init__()
        array_exists = (array_name in cfg.array_sig)
        if array_exists:
            sig_list = cfg.array_sig[array_name]
            check_sig(cfg, sig_list, index_list)
        else:
            sig = define_sig(cfg, index_list)
            cfg.array_sig[array_name] = sig

        self.cfg = cfg
        self.name = array_name
        self.index_list = index_list

        en = enumerate(self.index_list)
        self.slice_pos = next((p for p, i in en if isinstance(i, Slice)), None)

        if self.slice_pos is not None:
            slc = self.index_list[self.slice_pos]
            self.children.append(slc)
    
    def maybe_get_slice_index(self):
        if self.slice_pos is not None:
            return self.index_list[self.slice_pos]
        return None

    def prepare(self):
        super().prepare()
        sig = self.cfg.array_sig[self.name]
        if self.slice_pos is not None:
            slc = self.index_list[self.slice_pos]
            target_sig_tup = sig[self.slice_pos]
            if target_sig_tup.rank() != slc.slice_dim():
                raise RuntimeError(
                    f'Array contains a Slice index with size {slc.slice_dim()}'
                    f' for target {target_sig_tup}, with incompatible rank')

class LValueArray(Array):
    def __init__(self, cfg, array_name, index_list):
        super().__init__(cfg, array_name, index_list)

    def __repr__(self):
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'LValueArray({self.name})[{ind_list}]'

    def assign(self, rhs):
        if self.slice_pos is not None:
            raise NotImplementedError
            # need to use tf.scatter
        else:
            trg_inds = self.index_list
            val = rhs.evaluate(trg_inds)
            self.cfg.arrays[self.name] = val

    def add(self, rhs):
        if self.name not in self.cfg.arrays:
            raise RuntimeError(
                f'Cannot do += on first mention of array \'{self.name}\'')
        if self.slice_pos is not None:
            raise NotImplementedError
            # use tf.scatter
        else:
            trg_inds = self.index_list
            val = rhs.evaluate(trg_inds)
            prev = self.cfg.arrays[self.name]
            self.cfg.arrays[self.name] = tf.add(prev, val)
    
class RValueArray(Array):
    def __init__(self, cfg, array_name, index_list):
        super().__init__(cfg, array_name, index_list)

    def __repr__(self):
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'RValueArray({self.name})[{ind_list}]'

    def get_tups(self):
        # TODO: what to do if this is nested?
        tups = { tup for tup in self.index_list if isinstance(tup, str) }
        return tups

    def evaluate(self, trg_sig):
        if self.name not in self.cfg.arrays:
            raise RuntimeError(
                f'RValueArray {self.name} called evaluate() but '
                f'array is not materialized')

        array_slice = self.maybe_get_slice_index()

        if array_slice is not None:
            top_sig = self.cfg.array_sig[self.name]
            sub_sig = self.cfg.array_sig[array_slice.name]

            top_ten = self.cfg.arrays[self.name]
            sub_ten = self.cfg.arrays[array_slice.name]

            # group tuple dimensions together
            top_card = self.get_cardinality(top_sig)
            sub_card = self.get_cardinality(sub_sig)

            top_ten = tf.reshape(top_ten, top_card)
            sub_ten = tf.reshape(sub_ten, sub_card)

            sub_sig = [ i.name for i in sub_sig ] 
            top_sig = [ p.name for p in top_sig ]

            slice_sig = top_sig.pop(self.slice_pos)
            star_sig = sub_sig.pop(array_slice.star_tup.name)

            batch_sig = set(sub_sig).intersection(top_sig)
            fetch_sig = set(sub_sig).difference(top_sig)
            other_sig = set(top_sig).difference(sub_sig)

            # target shapes
            target_sub = batch_sig + fetch_sig + [star_sig]
            target_top = batch_sig + slice_sig + other_sig
            target_res = batch_sig + fetch_sig + other_sig

            top_ten = self.layout_to_sig(top_ten, top_sig, target_top)
            sub_ten = self.layout_to_sig(sub_ten, sub_sig, target_sub)

            star_dims = self.cfg.dims(star_sig)

            out_of_bounds = sg.range_check(sub_ten, star_dims)
            sub_ten = sg.flatten(sub_ten, star_dims)
            result = tf.gather_nd(top_ten, sub_ten, batch_dims=1)

            # 
            target_res_dims = self.flat_dims(target_res)
            result = self.layout_to_sig(result, target_res_dims)
            return result

        else:
            src_sig = self.cfg.array_sig[self.name]
            ten = self.cfg.arrays[self.name]
            ten = self.layout_to_sig(ten, src_sig, trg_sig)
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
        self.key_tup = cfg.maybe_add_tup(key_eintup)
        self.last_tup = cfg.maybe_add_tup(last_eintup)

    def __repr__(self):
        return f'RangeExpr({self.key_tup}, {self.last_tup})'

    def prepare(self):
        if self.last_tup.rank() != 1:
            raise RuntimeError(f'RangeExpr: last EinTup \'{self.last_tup}\''
                    f' must have rank 1.  Got {self.last_tup.rank()}')
        ind_dims = self.last_tup.dims()[0] 
        key_rank = self.key_tup.rank()
        if ind_dims != key_rank:
            raise RuntimeError(
                    f'RangeExpr: last EinTup \'{self.last_tup}\' has dimension '
                    f'{ind_dims} but must be equal to key EinTup '
                    f'\'{self.key_tup}\' rank {key_rank}')

    def get_tups(self):
        return {self.key_tup, self.last_tup}

    def evaluate(self, trg_sig):
        src_sig = [self.key_tup, self.last_tup]
        ten = [tf.range(e) for e in self.key_tup.dims()]
        ten = tf.meshgrid(*ten, indexing='ij')
        ten = tf.stack(ten, axis=self.key_tup.rank())
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
        sub_sig = self.get_tups()
        sub_sig = list(sub_sig)
        lval = self.lhs.evaluate(sub_sig)
        rval = self.rhs.evaluate(sub_sig)
        ten = self.op(lval, rval)
        ten = self.layout_to_sig(ten, sub_sig, trg_sig)
        return ten

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
        self.base_tup = None
        self.ind_tup = None

        if self.kind == DimKind.Int:
            self.index = int(index_expr) 
        elif self.kind == DimKind.Index:
            self.index = index_expr

    def __repr__(self):
        return f'Dims({self.ein_arg})[{self.index or ":"}]'

    def prepare(self):
        if self.ein_arg not in self.cfg.tups:
            raise RuntimeError(f'Dims argument \'{self.ein_arg}\' not a known Index')
        else:
            self.base_tup = self.cfg.tup(self.ein_arg)

        if (self.kind == DimKind.Int and 
                self.index >= len(self.cfg.tups[self.ein_arg])):
            raise RuntimeError(f'Dims index \'{self.ind}\' out of bounds')

        if self.kind == DimKind.Index:
            if self.index not in self.cfg.tups:
                raise RuntimeError(
                    f'Dims Index name \'{self.index}\' not known Index')
            else:
                self.ind_tup = self.cfg.tup(self.index)

            if self.ind_tup.rank() != 1:
                raise RuntimeError(
                    f'Dims Index index \'{self.ind_tup}\' must be '
                    f'rank 1, got \'{self.ind_tup.rank()}\'')
            if self.ind_tup.dims()[0] > self.base_tup.rank():
                raise RuntimeError(
                    f'Dims Index index \'{self.index}\' must'
                    f' have values in range of Index argument \'{self.base_tup}\'.'
                    f' {self.ind_tup.dims()[0]} exceeds {self.base_tup.rank()}')

    def evaluate(self, trg_sig):
        if self.kind != DimKind.Index:
            raise RuntimeError(
                f'Only {DimKind.Index.value} Dims can call evaluate()')
        src_sig = self.get_tups()
        ten = tf.constant(self.base_tup.dims())
        ten = self.layout_to_sig(ten, src_sig, trg_sig)
        return ten

    def value(self):
        if self.kind == DimKind.Index:
            raise RuntimeError(
                f'Cannot call value() on a {DimKind.Index.value} Dims')

        dims = self.base_tup.dims()
        if self.kind == DimKind.Star:
            return dims
        elif self.kind == DimKind.Int:
            return dims[self.index]
    
    def get_tups(self):
        if self.kind == DimKind.Index:
            return {self.ind_tup}
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

