import tensorflow as tf
import enum
import operator
import re
import util

def define_sig(runtime, use_list):
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

    return [ runtime.maybe_add_tup(use) for use in use_list ]

# Call at instantiation of an Array with established sig
def check_sig(runtime, sig_list, use_list):
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
            use_tup = runtime.maybe_add_tup(use, sig_tup)
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

def flat_dims(tups):
    # tup.dims() may be empty, but this still works correctly
    return [ dim for tup in tups for dim in tup.dims()]

def packed_dims(tups):
    # tup.nelem() returns 1 for a zero-rank tup.  this
    # seems to work correctly.
    return [ tup.nelem() for tup in tups ]

# reshape / transpose ten, with starting shape src_sig,
# to be broadcastable to trg_sig.  if is_packed, consume
# and produce the packed form of the signature
def to_sig(ten, src_sig, trg_sig, is_packed=False):
    expect_dims = packed_dims(src_sig) if is_packed else flat_dims(src_sig)
    if ten.shape != expect_dims:
        desc = 'packed' if is_packed else 'flat'
        raise RuntimeError(
            f'Tensor shape {ten.shape} not consistent with '
            f'signature {desc} shape {expect_dims}')
    if not is_packed:
        src_dims = packed_dims(src_sig)
        ten = tf.reshape(ten, src_dims)

    marg_ex = set(src_sig).difference(trg_sig)
    if len(marg_ex) != 0:
        marg_pos = [ i for i, tup in enumerate(src_sig) if tup in marg_ex ]
        ten = tf.reduce_sum(ten, marg_pos)

    src_sig = [ tup for tup in src_sig if tup not in marg_ex ]
    card = packed_dims(src_sig)
    augmented = list(src_sig)
    trg_dims = []

    for ti, trg in enumerate(trg_sig):
        if trg not in src_sig:
            card.append(1)
            augmented.append(trg)
            trg_dims.extend([1] * trg.rank())
        else:
            trg_dims.extend(trg.dims())

    # trg_sig[i] = augmented[perm[i]], maps augmented to trg_sig
    perm = []
    for trg in trg_sig:
        perm.append(augmented.index(trg))

    ten = tf.reshape(ten, card)
    ten = tf.transpose(ten, perm)

    if not is_packed:
        ten = tf.reshape(ten, trg_dims)

    return ten

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

    # Call after all parsing is finished
    def post_parse_init(self):
        for ch in self.children:
            ch.post_parse_init()

    def get_tups(self):
        return {tup for c in self.children for tup in c.get_tups()}

class Slice(AST):
    # Represents an expression ary[a,b,c,:,e,...] with exactly one ':' and
    # the rest of the indices simple eintup names.  
    def __init__(self, runtime, array_name, index_list):
        super().__init__()
        if array_name not in runtime.array_sig:
            raise RuntimeError(
                f'Cannot instantiate Slice as first appearance of array name '
                f'\'{array_name}\'')
        sig = runtime.array_sig[array_name]
        if len(sig) != len(index_list):
            raise RuntimeError(
                f'Slice instantiated with incorrect number of indices. '
                f'Expecting {len(sig)} but got {len(index_list)}')

        found_star = False
        z = zip(range(len(sig)), sig, index_list)
        for pos, sig_tup, call in z:
            if call == STAR: 
                if found_star:
                    raise RuntimeError(
                        f'Found a second \':\' index.  Only one wildcard is '
                        f'allowed in a Slice instance')
                found_star = True
                self.star_tup = sig_tup
                self.star_pos = pos
                continue
            elif not isinstance(call, str):
                raise RuntimeError(
                    f'Slice object only accepts simple tup names or \':\' as '
                    f'indices.  Got \'{call}\'')
            call_tup = runtime.maybe_add_tup(call, shadow_of=sig_tup)
            if not sig_tup.same_shape_as(call_tup):
                raise RuntimeError(
                    f'Slice called with incompatible shape. '
                    f'{call_tup} called in slot of {sig_tup}')

        if not found_star:
            raise RuntimeError(
                f'Slice must contain at least one \':\' in index_list. '
                f'Got \'{index_list}\'') 

        # passed all checks
        self.runtime = runtime
        self.name = array_name
        self.index_list = [ self.runtime.tup(name) if name != STAR else name 
                for name in index_list ]

    def __repr__(self):
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'Slice({self.name})[{ind_list}]'

    def slice_dim(self):
        return self.star_tup.dims()[0]

    def get_array(self):
        rank = self.star_tup.rank() 
        if rank != 1:
            raise RuntimeError(
                f'Slice wildcard index must be rank 1.  Got {rank}')
        if self.name not in self.runtime.arrays:
            raise RuntimeError(
                f'Slice is not materialized yet.  Cannot call evaluate()')
        z = zip(self.runtime.array_sig[self.name], self.index_list)
        use_sig = [ sig if use == STAR else use for sig, use in z]
        return (self.runtime.arrays[self.name], use_sig)

class Array(AST):
    def __init__(self, runtime, array_name, index_list):
        super().__init__()
        array_exists = (array_name in runtime.array_sig)
        if array_exists:
            sig_list = runtime.array_sig[array_name]
            check_sig(runtime, sig_list, index_list)
        else:
            sig = define_sig(runtime, index_list)
            runtime.array_sig[array_name] = sig

        self.runtime = runtime
        self.name = array_name
        self.index_list = [ self.runtime.tup(name) if isinstance(name, str)
                else name for name in index_list ]

        en = enumerate(self.index_list)
        self.slice_pos = next((p for p, i in en if isinstance(i, Slice)), None)

        if self.slice_pos is not None:
            slc = self.index_list[self.slice_pos]
            self.children.append(slc)

    def __repr__(self):
        cls_name = self.__class__.__name__
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'{cls_name}({self.name})[{ind_list}]'

    def has_slice(self):
        return self.slice_pos is not None
    
    def maybe_get_slice_node(self):
        if self.has_slice():
            return self.index_list[self.slice_pos]
        return None

    def get_sig(self):
        if self.name not in self.runtime.array_sig:
            raise RuntimeError(
                f'Array {self.name} does not have a registered signature')
        return self.runtime.array_sig[self.name]

    def _get_array(self):
        if self.name not in self.runtime.arrays:
            raise RuntimeError(
                f'Array {self.name} called evaluate() but not materialized')
        # replace the Slice with the tup if it exists
        z = zip(self.runtime.array_sig[self.name], self.index_list)
        use_sig = [ sig if isinstance(use, Slice) else use for sig, use in z]
        return (self.runtime.arrays[self.name], use_sig)

    def get_array(self):
        if self.has_slice():
            raise RuntimeError(
                f'Cannot call Array::get_array() on slice-containing array')
        return self._get_array()

    def _shape_check(self):
        # check that 
        sig = self.runtime.array_sig[self.name]
        slice_node = self.maybe_get_slice_node()
        target_sig_tup = sig[self.slice_pos]
        if target_sig_tup.rank() != slice_node.slice_dim():
            raise RuntimeError(
                f'Array contains Slice of size {slice_node.slice_dim()} '
                f'for target {target_sig_tup} of rank {target_sig_tup.rank()}.'
                f' Size and rank must match')

    def get_array_and_slice(self):
        if not self.has_slice():
            raise RuntimeError(
                f'Cannot call Array:get_array_and_slice() on non-slice array')
        self._shape_check()
        array = self._get_array()
        slice_node = self.maybe_get_slice_node()
        slice_array = slice_node.get_array()
        return (array, slice_array) 

class LValueArray(Array):
    def __init__(self, runtime, array_name, index_list):
        super().__init__(runtime, array_name, index_list)

    def __repr__(self):
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'LValueArray({self.name})[{ind_list}]'

    def _evaluate_sliced(self, trg_sig, rhs):
        # see ops/scatter_nd.et
        """
        Approach:
        1. pack indices (ind), updates (upd) and output (out) sigs
        2. calculate target sigs for idx, upd and out using util.union_ixn
        3. construct target 
        """
        array, slice_array = self.get_array_and_slice()
        _, out_sig = array
        idx_ten, idx_sig = slice_array
        idx_ten = tf.reshape(idx_ten, packed_dims(idx_sig))

        slice_node = self.maybe_get_slice_node()
        out_sig_orig = list(out_sig)
        idx_sig_orig = list(idx_sig)

        slice_tup = out_sig.pop(self.slice_pos)
        star_tup = idx_sig.pop(slice_node.star_pos)

        ixn_union = util.union_ixn(idx_sig, out_sig)
        fetch_sig, batch_sig, other_sig = ixn_union
        if len(batch_sig) > 0:
            raise RuntimeError(f'cannot support batched scatter')

        target_idx = fetch_sig + [star_tup] 
        target_upd = fetch_sig + other_sig
        target_out = [slice_tup] + other_sig

        upd_ten  = rhs.evaluate(target_upd)
        upd_ten = tf.reshape(upd_ten, packed_dims(target_upd))
        idx_ten = to_sig(idx_ten, idx_sig_orig, target_idx, is_packed=True)

        in_bounds = util.range_check(idx_ten, slice_tup.dims())
        idx_ten = util.flatten(idx_ten, slice_tup.dims())
        idx_ten = tf.where(in_bounds, idx_ten, -1)

        shape_ten = tf.constant(packed_dims(target_out))
        out_ten = tf.scatter_nd(idx_ten, upd_ten, shape_ten)

        out_ten = to_sig(out_ten, target_out, trg_sig, is_packed=True)
        out_ten = tf.reshape(out_ten, flat_dims(trg_sig))
        return out_ten

    def assign(self, rhs):
        trg_sig = self.get_sig()

        if self.has_slice():
            val = self._evaluate_sliced(trg_sig, rhs)
        else:
            val = rhs.evaluate(trg_sig)

        self.runtime.arrays[self.name] = val

    def add(self, rhs):
        if self.name not in self.runtime.arrays:
            raise RuntimeError(
                f'Cannot do += on first mention of array \'{self.name}\'')

        trg_sig = self.index_list
        if self.has_slice():
            val = self._evaluate_sliced(trg_sig)
        else:
            val = rhs.evaluate(trg_sig)

        prev = self.runtime.arrays[self.name]
        self.runtime.arrays[self.name] = tf.add(prev, val)
    
class RValueArray(Array):
    def __init__(self, runtime, array_name, index_list):
        super().__init__(runtime, array_name, index_list)

    def __repr__(self):
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'RValueArray({self.name})[{ind_list}]'

    def get_tups(self):
        # TODO: what to do if this is nested?
        tups = { tup for tup in self.index_list if not isinstance(tup, Slice) }
        return tups

    def _evaluate_sliced(self, trg_sig):
        # see 'gather' test in ops/gather_nd.et
        """
        Overall approach:
        1.  pack all tuple dims in the param (par) and indices (idx) tensors
        2.  calculate the 3-way intersection/difference signatures between
            par and idx.
        3.  construct target signatures for par, idx, and result, to conform
            to expected signature for native tf.gather_nd
        4.  conform par and idx tensors to these signatures
        5.  calculate the bounds mask for idx tensor
        6.  flatten idx tensor values along the last dimension (util.flatten)
        7.  call tf.gather_nd
        8.  reform to the target signature
        9.  flatten the shape and return

        """
        array, slice_array = self.get_array_and_slice()
        par_ten, par_sig = array
        idx_ten, idx_sig = slice_array

        par_ten = tf.reshape(par_ten, packed_dims(par_sig))
        idx_ten = tf.reshape(idx_ten, packed_dims(idx_sig))

        slice_node = self.maybe_get_slice_node()
        par_sig_orig = list(par_sig)
        idx_sig_orig = list(idx_sig)

        # See ops/gather_nd.et
        # After slice_tup is removed from par_sig, and star_tup is removed from
        # idx_sig, we have:
        # par_sig = batch, other
        # idx_sig = batch, slice
        slice_tup = par_sig.pop(self.slice_pos)
        star_tup = idx_sig.pop(slice_node.star_pos)
        ixn_union_triplet = util.union_ixn(idx_sig, par_sig)
        fetch_sig, batch_sig, other_sig = ixn_union_triplet 

        # target shapes
        target_idx = batch_sig + fetch_sig + [star_tup]
        target_par = batch_sig + [slice_tup] + other_sig
        target_res = batch_sig + fetch_sig + other_sig

        par_ten = to_sig(par_ten, par_sig_orig, target_par, is_packed=True)
        idx_ten = to_sig(idx_ten, idx_sig_orig, target_idx, is_packed=True)

        in_bounds = util.range_check(idx_ten, slice_tup.dims())
        idx_ten = util.flatten(idx_ten, slice_tup.dims())
        idx_ten = tf.where(in_bounds, idx_ten, -1)

        num_batch_dims = len(batch_sig)
        result = tf.gather_nd(par_ten, idx_ten, batch_dims=num_batch_dims)

        result = to_sig(result, target_res, trg_sig, is_packed=True)
        result = tf.reshape(result, flat_dims(trg_sig))
        return result

    def evaluate(self, trg_sig):
        if self.has_slice():
            return self._evaluate_sliced(trg_sig)
        else:
            ten, src_sig = self.get_array()
            ten = to_sig(ten, src_sig, trg_sig)
            return ten

class RandomCall(AST):
    # apply a function pointwise to the materialized arguments
    # args can be: constant or array-like
    def __init__(self, runtime, min_expr, max_expr, dtype_string):
        super().__init__(min_expr, max_expr)
        self.runtime = runtime
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
        trg_dims = flat_dims(trg_sig) 
        rnd = tf.random.uniform(trg_dims, 0, 2**31-1, dtype=self.dtype)
        ten = rnd % (maxs - mins) + mins
        return ten

class RangeExpr(AST):
    # Problem: no good way to instantiate 'children' here since
    # the eintup's are just strings
    # RANGE[s, c], with s the key_eintup, and c the 1-D last_eintup
    def __init__(self, runtime, key_eintup, last_eintup):
        super().__init__()
        self.runtime = runtime
        self.key_tup = runtime.maybe_add_tup(key_eintup)
        self.last_tup = runtime.maybe_add_tup(last_eintup)

    def __repr__(self):
        return f'RangeExpr({self.key_tup}, {self.last_tup})'

    def get_tups(self):
        return {self.key_tup, self.last_tup}

    def evaluate(self, trg_sig):
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
        src_sig = [self.key_tup, self.last_tup]
        ten = [tf.range(e) for e in self.key_tup.dims()]
        ten = tf.meshgrid(*ten, indexing='ij')
        ten = tf.stack(ten, axis=self.key_tup.rank())
        ten = to_sig(ten, src_sig, trg_sig)
        return ten

class ArrayBinOp(AST):
    def __init__(self, runtime, lhs, rhs, op_string):
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

        self.runtime = runtime
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
        ten = to_sig(ten, sub_sig, trg_sig)
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
        return to_sig(ten, src_sig, trg_sig)

class IntExpr(ScalarExpr):
    def __init__(self, runtime, val):
        super().__init__()
        self.runtime = runtime
        self.val = int(val)

    def value(self):
        return self.val

class FloatExpr(ScalarExpr):
    def __init__(self, runtime, val):
        super().__init__()
        self.runtime = runtime
        self.val = val 

    def value(self):
        return self.val

class Rank(ScalarExpr):
    def __init__(self, runtime, tup_name_list):
        super().__init__()
        for name in tup_name_list:
            if name not in runtime.tups:
                raise RuntimeError(f'Rank tup {name} not a known EinTup')
        self.runtime = runtime
        self.tups = [ self.runtime.tup(name) for name in tup_name_list ]

    def __repr__(self):
        return f'Rank({repr(self.tups)})'

    def value(self):
        return sum(tup.rank() for tup in self.tups)

class DimKind(enum.Enum):
    Star = 'Star'
    Int = 'Int' 
    Index = 'Index'

class Dims(AST):
    """
    Dims can exist in three sub-types.  
    Expression             Type       Usage
    Dims(tupname)[0]       Int        statement, constraint
    Dims(tupname)[ind_tup] Index      statement
    Dims(tupname)[:]       Star       constraint

    """
    def __init__(self, runtime, kind, tup_name_list, index_expr=None):
        super().__init__()
        self.runtime = runtime
        self.tup_names = tup_name_list
        self.kind = kind
        self.base_tups = []
        self.ind_tup = None

        if self.kind == DimKind.Int:
            self.index = int(index_expr) 
        elif self.kind == DimKind.Index:
            self.ind_tup_name = index_expr

    def __repr__(self):
        return f'{self.kind} Dims({self.base_tups})[{self.ind_tup}]'

    def post_parse_init(self):
        for name in self.tup_names:
            if name not in self.runtime.tups:
                raise RuntimeError(f'Dims argument \'{name}\' not a known Index')
            else:
                self.base_tups.append(self.runtime.tup(name))

        if self.kind == DimKind.Index:
            if self.ind_tup_name not in self.runtime.tups:
                raise RuntimeError(
                    f'Dims Index name \'{self.ind_tup_name}\' not known Index')
            else:
                self.ind_tup = self.runtime.tup(self.ind_tup_name)


    def evaluate(self, trg_sig):
        if self.kind != DimKind.Index:
            raise RuntimeError(
                f'Only {DimKind.Index.value} Dims can call evaluate()')

        if self.kind == DimKind.Int and self.index >= self.rank():
            raise RuntimeError(
                f'Dims index \'{self.index}\' must be less than '
                f'rank {self.rank()}')

        if self.kind == DimKind.Index:
            if self.ind_tup.rank() != 1:
                raise RuntimeError(
                    f'Dims Index index \'{self.ind_tup}\' must be '
                    f'rank 1, got \'{self.ind_tup.rank()}\'')
            if self.ind_tup.dims()[0] != self.rank():
                tup_list = [tup.name for tup in self.base_tups]
                raise RuntimeError(
                    f'Index Dims index {self.ind_tup} first value '
                    f'({self.ind_tup.dims()[0]}) must be equal to rank of '
                    f'base tup list {tup_list} ({self.rank()})')

        src_sig = self.get_tups()
        ten = tf.constant(flat_dims(self.base_tups))
        ten = to_sig(ten, src_sig, trg_sig)
        return ten

    def rank(self):
        return sum(tup.rank() for tup in self.base_tups)

    def value(self):
        if self.kind == DimKind.Index:
            raise RuntimeError(
                f'Cannot call value() on a {DimKind.Index.value} Dims')

        dims = flat_dims(self.base_tups)
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
    """
    A Binary operator for use only in constraints.
    Accepts IntExpr, Rank, Dims (Int and Star) types
    """
    # Dims types   
    def __init__(self, arg1, arg2):
        accepted_classes = (IntExpr, Rank, Dims, StaticBinOpBase)
        cls_name = super().__class__.__name__
        if not (isinstance(arg1, accepted_classes) and
                isinstance(arg2, accepted_classes)):
            raise RuntimeError(
                f'{cls_name} only IntExpr, Rank, Dims and StaticBinOpBase '
                'accepted')
        if ((isinstance(arg1, Dims) and arg1.kind == DimKind.Index) or
                (isinstance(arg2, Dims) and arg2.kind == DimKind.Index)):
            raise RuntimeError(
                f'{cls_name} does not support Index Dims')

        super().__init__(arg1, arg2)
        self.arg1 = arg1
        self.arg2 = arg2

    # map the op to broadcasted values
    def _map_op(self):
        vals1 = self.arg1.value()
        vals2 = self.arg2.value()
        is_list1 = isinstance(vals1, (list, tuple))
        is_list2 = isinstance(vals2, (list, tuple))
        if is_list1 and not is_list2:
            vals2 = [vals2] * len(vals1)
        elif not is_list1 and is_list2:
            vals1 = [vals1] * len(vals2)
        elif is_list1 and is_list2:
            if len(vals1) != len(vals2):
                cls_name = super().__class__.__name__
                raise RuntimeError(
                    f'{cls_name} got unequal length values')
        else:
            vals1 = [vals1]
            vals2 = [vals2]
        return [ self.op(el1, el2) for el1, el2 in zip(vals1, vals2) ]

    def value(self):
        return self.reduce(self._map_op())

class ArithmeticBinOp(StaticBinOpBase):
    def __init__(self, arg1, arg2, op):
        super().__init__(arg1, arg2)
        opfuncs = [ operator.add, operator.sub, operator.mul, operator.truediv,
                operator.floordiv ]
        self.op = dict(zip(['+', '-', '*', '/', '//'], opfuncs))[op]

    def reduce(self, op_vals):
        return op_vals

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

    def reduce(self, op_vals):
        return all(op_vals)

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
    def __init__(self, runtime, name):
        super().__init__()
        if name not in runtime.array_sig:
            raise RuntimeError(f'argument must be array name, got {name}')
        self.runtime = runtime
        self.name = name

    def __repr__(self):
        return f'TensorArg({self.name})'

    def value(self):
        # materializes the value to full dimensions.
        # runtime.arrays may be stored in a form merely broadcastable to the
        # signature.
        sig = self.runtime.array_sig[self.name]
        full_dims = flat_dims(sig)
        ten = self.runtime.arrays[self.name]
        ten = tf.broadcast_to(ten, full_dims)
        return ten

class TensorWrap(AST):
    """
    Wraps a static value and produces a constant tensor
    """
    def __init__(self, runtime, node):
        if not isinstance(node, (Dims, Rank)):
            raise RuntimeError(
                f'TensorWrap can only wrap a Dims or Rank instance')
        super().__init__(node)
        self.node = node

    def value(self):
        return tf.constant(self.node.value())

class TFCall(AST):
    """
    Represents a python function call of a TensorFlow function.
    Arguments can be TensorArg, Dims, Rank, or python literals.
    Python literals are wrapped with 'L(...)'
    """
    def __init__(self, func_name, tf_call_list):
        ast_nodes = [ el for el in tf_call_list if isinstance(el, AST) ]
        super().__init__(*ast_nodes)
        try:
            self.func = eval(func_name)
        except NameError as ne:
            raise RuntimeError(
                f'TFCall could not find function {func_name}: {ne}')
        self.func_name = func_name
        self.tf_call_list = tf_call_list

    def __repr__(self):
        return f'TFCall({self.func_name})[{self.tf_call_list}]'

    def value(self):
        def v(a):
            return a.value() if isinstance(a, AST) else a
        calls = self.tf_call_list
        kwargs = dict((a[0], v(a[1])) for a in calls if isinstance(a, tuple))
        args = [ v(a) for a in calls if not isinstance(a, tuple) ]
        result = self.func(*args, **kwargs)

        # promote to list
        if not isinstance(result, (list, tuple)):
            result = [result]
        return result

if __name__ == '__main__':
    from runtime import Runtime
    rt = Runtime(5, 10)

    rt.maybe_add_tup('batch')
    rt.maybe_add_tup('slice')
    rt.maybe_add_tup('coord')
    rt.set_dims({'batch': 3, 'slice': 3, 'coord': 1})
    rt.tups['coord'].set_dim(0, 3)
    # rng = RangeExpr(rt, 'batch', 'coord')
    # ten = rng.evaluate(['slice', 'batch', 'coord'])
    
    # d1 = Dims(rt, 'batch', DimKind.Index, 'coord')
    # d2 = Dims(rt, 'slice', DimKind.Index, 'coord')
    # rk = Rank(rt, 'batch')
    # rnd = RandomCall(rt, IntExpr(rt, 0), rk, 'INT')
    # ten = rnd.evaluate(['slice', 'batch', 'coord'])

    print(ten)
    print(ten.shape)
    print(rt.tups)

