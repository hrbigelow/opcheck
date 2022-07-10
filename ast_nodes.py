import tensorflow as tf
import numpy as np
import enum
import operator
import util

# Call at instantiation of an Array with established sig
# TODO: move to util (how to resolve circular dependency with ast_nodes.Slice?)
def check_sig(runtime, sig_list, use_list):
    if len(sig_list) != len(use_list):
        raise RuntimeError(
            f'check_sig expected {len(sig_list)} indices but found '
            f'{len(use_list)}')

    nslices = len([u for u in use_list if isinstance(u, SliceExpr)])
    if nslices > 1:
        raise RuntimeError(
            f'check_sig expected 0 or 1 non-EinTup arguments.  Found {nslices}')

    """
    for sig_tup, use in zip(sig_list, use_list):
        if isinstance(use, str):
            use_tup = runtime.maybe_add_tup(use, sig_tup)
            if not sig_tup.same_shape_as(use_tup):
                raise RuntimeError(
                    f'check_sig found incompatible shapes.  Expecting '
                    f'{sig_tup} but found {use_tup}')
        elif isinstance(use, SliceExpr):
            pass # nothing to check during instantiation
        else:
            raise RuntimeError(
                f'check_sig expected string or SliceExpr argument.  '
                f'Found {type(use)}')
    """

def define_sig(runtime, use_list):
    allowed_types = (str, ShapeExpr)
    not_str = next((use for use in use_list if 
        not isinstance(use, allowed_types)), None)
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

STAR = ':'

class ShapeExpr(object):
    """
    A ShapeExpr is the base component for an array signature which can be used
    in the define_sig 'use_list'
    """
    def __init__(self):
        pass

    def dims(self):
        raise NotImplementedError

    def nelem(self):
        return np.prod(self.dims(), dtype=np.int32)

    def rank(self):
        return len(self.dims())

class EinTup(ShapeExpr):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.shape = Shape() 
        # TODO: parameterize these
        self.rank_expr = (0, 10)
        self.gen_expr = RangeConstraint(0, 100, self)

    def __repr__(self):
        try:
            dimstring = ','.join([str(d) for d in self.dims()])
        except RuntimeError:
            dimstring = '?'
        try:
            rankstring = self.rank()
        except RuntimeError:
            rankstring = '?'
        return f'EinTup \'{self.name}\' |{rankstring}| [{dimstring}]'

    def __len__(self):
        return len(self.dims())

    def initialize(self, dims):
        self.shape.initialize(dims)

    def clear(self):
        self.shape.clear()

    def set_rank(self, rank):
        self.shape.set_rank(rank)

    def set_dims(self, dims):
        self.shape.set_dims(dims)

    # calculate the rank from the rank_expr
    def calc_rank(self):
        if not self.has_rank():
            self.set_rank(self.rank_expr.value())
        return self.rank()

    def add_gen_expr(self, gen_expr):
        self.gen_expr = gen_expr

    def add_rank_expr(self, rank_expr):
        self.rank_expr = rank_expr

    def gen_dims(self):
        if not self.has_dims():
            dims = self.gen_expr.value()
            if isinstance(dims, int):
                dims = [dims] * self.rank()
            self.set_dims(dims)
        return self.dims()

    def has_dims(self):
        return self.shape.has_dims()

    def has_rank(self):
        return self.shape.has_rank()

    def dims(self):
        return self.shape.get_dims()

    def rank(self):
        return self.shape.get_rank()

class Shape(object):
    # simple data class
    def __init__(self):
        self.dims = None
        self.rank = None

    def __repr__(self):
        return (f'Shape: rank {self.rank}, dims {self.dims}')

    def initialize(self, dims):
        self.rank = len(dims)
        self.dims = list(dims)

    def clear(self):
        self.rank = None
        self.dims = None

    def set_rank(self, rank):
        self.dims = None
        self.rank = rank

    def get_rank(self):
        if self.rank is None:
            raise RuntimeError(
                f'Cannot call Shape::get_rank() before rank is set')
        return self.rank

    def has_dims(self):
        return self.dims is not None

    def has_rank(self):
        return self.rank is not None

    def get_dims(self):
        if self.dims is None:
            raise RuntimeError('Cannot call get_dims() on uninitialized Shape')
        return self.dims

    def set_dims(self, dims):
        if not self.has_rank():
            raise RuntimeError(
                f'Cannot call set_dims when rank is not set')
        if len(dims) != self.get_rank():
            raise RuntimeError(
                f'set_dims received {dims} but rank is {self.rank()}')
        self.dims = list(dims)

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
        tuplists = [ ch.get_tups() for ch in self.children ]
        merged = tuplists[0] if len(tuplists) else []
        for l in tuplists[1:]:
            merged = util.merge_tup_lists(merged, l)
        return merged

class Slice(AST):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        raise NotImplementedError

    def rank(self):
        raise NotImplementedError

    def get_array(self):
        raise NotImplementedError

    def get_tups(self):
        raise NotImplementedError

    def ind_pos(self):
        raise NotImplementedError

class SliceExpr(AST):
    def __init__(self, runtime, basis, *children):
        super().__init__(*children)
        self.runtime = runtime
        self.basis = basis

    def __repr__(self):
        return f'{self.__class__.__name__}({self.basis})'

    # The full signature is basis + [rank_sig]
    def get_basis(self):
        return self.basis

    def rank_sig(self):
        raise NotImplementedError

    def rank(self):
        return self.rank_sig().dims()[0]

    def evaluate(self, trg_basis):
        # return a tensor whose shape is broadcastable to 
        # trg_basis + [self.rank_sig()]
        raise NotImplementedError

class IntSlice(SliceExpr, ShapeExpr):
    def __init__(self, runtime, val):
        super().__init__(runtime, basis=list())
        self.val = val

    # to satisfy ShapeExpr (not sure why we want to use IntSlice as a ShapeExpr
    # though
    def dims(self):
        return [self.val]

    def rank_sig(self):
        return Rank(self.runtime, [])

    def evaluate(self, trg_basis): 
        ten = tf.constant(self.val, dtype=tf.int32)
        ten = util.to_sig(ten, [], trg_basis + [self.rank_sig()])
        return ten

class RankSlice(SliceExpr, ShapeExpr):
    def __init__(self, runtime, rank):
        super().__init__(runtime, basis=list())
        self.rank = rank

    # Using RankSlice as a shape
    def dims(self):
        return [self.rank.value()]

    def rank_sig(self):
        return Rank(self.runtime, [])

    def evaluate(self, trg_basis):
        ten = tf.constant(self.rank.value(), dtype=tf.int32)
        ten = util.to_sig(ten, [], trg_basis + [1])
        return ten

class DimsSlice(SliceExpr):
    def __init__(self, runtime, tup_names):
        super().__init__(runtime, basis=list())
        self.tup_names = tup_names
        self.base_tups = [ runtime.maybe_add_tup(n) for n in tup_names ]

    def rank_sig(self):
        return Rank(self.runtime, self.tup_names)

    def value(self):
        return util.flat_dims(self.base_tups)

    def evaluate(self, trg_basis):
        src_basis = self.get_basis()
        rank_sig = self.rank_sig()
        ten = tf.constant(self.value(), dtype=tf.int32)
        ten = util.to_sig(ten, src_basis + [rank_sig], trg_basis + [rank_sig])
        return ten

class EinTupSlice(SliceExpr):
    def __init__(self, runtime, eintup_name):
        self.tup = runtime.maybe_add_tup(eintup_name)
        super().__init__(runtime, [self.tup])

    def __repr__(self):
        return f'EinTupSlice({self.basis})'

    def rank_sig(self):
        return Rank(self.runtime, [self.tup.name])

    def evaluate(self, trg_basis):
        src_basis = self.get_basis()
        rank_sig = self.rank_sig()
        ten = util.ndrange(self.tup.dims()) 
        ten = util.to_sig(ten, src_basis + [rank_sig], trg_basis + [rank_sig])
        return ten

class ArraySlice(SliceExpr):
    # Represents an expression ary[a,b,c,:,e,...] with exactly one ':' and
    # the rest of the indices simple eintup names.  
    def __init__(self, runtime, array_name, index_list):
        if array_name not in runtime.array_sig:
            raise RuntimeError(
                f'Cannot instantiate ArraySlice as first appearance of array '
                f'\'{array_name}\'')
        sig = runtime.array_sig[array_name]
        if len(sig) != len(index_list):
            raise RuntimeError(
                f'ArraySlice instantiated with incorrect number of indices. '
                f'Expecting {len(sig)} but got {len(index_list)}')

        found_star = False
        z = zip(range(len(sig)), sig, index_list)
        for pos, sig_tup, call in z:
            if call == STAR: 
                if found_star:
                    raise RuntimeError(
                        f'Found a second \':\' index.  Only one wildcard is '
                        f'allowed in a ArraySlice instance')
                found_star = True
                self.ind_tup = sig_tup
                self._ind_pos = pos
                continue
            elif not isinstance(call, str):
                raise RuntimeError(
                    f'ArraySlice object only accepts simple tup names or \':\' as '
                    f'indices.  Got \'{call}\'')
            elif call != sig_tup.name:
                raise RuntimeError(
                    f'ArraySlice called with wrong EinTup. '
                    f'{call_tup} used instead of {sig_tup}')

        if not found_star:
            raise RuntimeError(
                f'ArraySlice must contain at least one \':\' in index_list. '
                f'Got \'{index_list}\'') 

        # passed all checks
        self.name = array_name
        self.index_list = [ runtime.tup(name) if name != STAR else name 
                for name in index_list ]

        basis = [ t for t in self.index_list if isinstance(t, EinTup) ]
        super().__init__(runtime, basis=basis)

    def __repr__(self):
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'ArraySlice({self.name})[{ind_list}]'

    def rank_sig(self):
        sig = self.runtime.array_sig[self.name]
        return sig[self._ind_pos]

    def evaluate(self, trg_basis):
        src_basis = self.get_basis()
        rank_sig = self.rank_sig()
        ten = self.runtime.arrays[self.name]
        ten = util.to_sig(ten, src_basis + [rank_sig], trg_basis + [rank_sig])
        return ten

    def rank(self):
        # a little odd, but the 'rank' of a slice is thought of as the rank
        # of the EinTup which it replaces.
        return self.ind_tup.dims()[0]

class SliceBinOp(SliceExpr):
    def __init__(self, runtime, lhs, rhs, op_string):
        if not isinstance(lhs, SliceExpr) or not isinstance(rhs, SliceExpr):
            raise RuntimeError(
                f'SliceBinOp can only operate on two SliceExpr instances. '
                f'Got {type(lhs)} and {type(rhs)}')
        super().__init__(runtime, basis=list())
        self.lhs = lhs
        self.rhs = rhs
        ops = [ tf.add, tf.subtract, tf.multiply, tf.math.floordiv ]
        self.op = dict(zip(['+', '-', '*', '//'], ops))[op_string]
        self.op_string = op_string

    def __repr__(self):
        return f'SliceBinOp({self.lhs} {self.op_string} {self.rhs})' 

    def get_basis(self):
        lbasis = self.lhs.get_basis()
        rbasis = self.rhs.get_basis()
        return util.merge_tup_lists(lbasis, rbasis)

    def rank_sig(self):
        lrank_sig = self.lhs.rank_sig()
        rrank_sig = self.rhs.rank_sig()
        lrank = lrank_sig.dims()
        rrank = rrank_sig.dims()
        if lrank != [0] and rrank != [0] and lrank != rrank:
            raise RuntimeError(
                f'SliceBinOp has incompatible ranks for lhs and rhs: '
                f'Got {lrank} and {rrank}.  Should be equal or broadcastable')
        return lrank_sig if lrank != [0] else rrank_sig 

    def evaluate(self, trg_basis):
        src_basis = self.get_basis()
        lten = self.lhs.evaluate(src_basis)
        rten = self.rhs.evaluate(src_basis)
        ten = self.op(lten, rten)
        rank_sig = self.rank_sig()
        ten = util.to_sig(ten, src_basis + [rank_sig], trg_basis + [rank_sig])
        return ten

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

    def __repr__(self):
        cls_name = self.__class__.__name__
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'{cls_name}({self.name})[{ind_list}]'

    def _get_slice_pos(self):
        en = enumerate(self.index_list)
        return next((p for p, i in en if isinstance(i, SliceExpr)), None)

    def get_slice(self):
        pos = self._get_slice_pos()
        return None if pos is None else self.index_list[pos]

    def get_sig(self):
        if self.name not in self.runtime.array_sig:
            raise RuntimeError(
                f'Array {self.name} does not have a registered signature')
        return self.runtime.array_sig[self.name]

    def nonslice_tups(self):
        return [ use for use in self.index_list if not isinstance(use,
            SliceExpr) ]

    def slice_tup(self):
        sig = self.get_sig()
        pos = self._get_slice_pos()
        return sig[pos]

    def get_array(self, trg_sig):
        if self.name not in self.runtime.arrays:
            raise RuntimeError(
                f'Array {self.name} called evaluate() but not materialized')
        ten = self.runtime.arrays[self.name]
        sig = self.runtime.array_sig[self.name]
        z = zip(sig, self.index_list)
        use_sig = [ sig if isinstance(use, SliceExpr) else use for sig, use in z]
        ten = util.to_sig(ten, use_sig, trg_sig)
        return ten

    def _rank_check(self):
        # check that the rank of the slice matches the expected rank
        sig = self.runtime.array_sig[self.name]
        slice_node = self.maybe_get_slice_node()
        target_sig_tup = sig[self.slice_pos]
        if target_sig_tup.rank() != slice_node.rank():
            raise RuntimeError(
                f'Array contains ArraySlice of rank {slice_node.rank()} '
                f'for target {target_sig_tup} of rank {target_sig_tup.rank()}.'
                f'ranks must match')

class LValueArray(Array):
    def __init__(self, runtime, array_name, index_list):
        super().__init__(runtime, array_name, index_list)

    def __repr__(self):
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'LValueArray({self.name})[{ind_list}]'

    def _evaluate_sliced(self, trg_sig, slice_node, rhs):
        # see ops/scatter_nd.et
        """
        Approach:
        1. pack indices (ind), updates (upd) and output (out) sigs
        2. calculate target sigs for idx, upd and out using util.union_ixn
        3. construct target 
        """
        # defines output signature in the top-level array where the slice resides
        out_sig = self.nonslice_tups()
        slice_tup = self.slice_tup()
        idx_sig = slice_node.get_basis()
        idx_rank = slice_node.rank_sig()
        idx_ten = slice_node.evaluate(idx_sig)
        idx_ten = util.pack(idx_ten, idx_sig + [idx_rank])

        ixn_union = util.union_ixn(idx_sig, out_sig)
        fetch_sig, batch_sig, other_sig = ixn_union
        if len(batch_sig) > 0:
            raise RuntimeError(f'cannot support batched scatter')

        target_idx = fetch_sig + [idx_rank] 
        target_upd = fetch_sig + other_sig
        target_out = [slice_tup] + other_sig

        upd_ten  = rhs.evaluate(target_upd)
        upd_ten = tf.reshape(upd_ten, util.packed_dims(target_upd))

        in_bounds = util.range_check(idx_ten, slice_tup.dims())
        idx_ten = util.flatten(idx_ten, slice_tup.dims())
        idx_ten = tf.where(in_bounds, idx_ten, -1)

        shape_ten = tf.constant(util.packed_dims(target_out))
        with tf.device('/GPU:0'):
            out_ten = tf.scatter_nd(idx_ten, upd_ten, shape_ten)

        out_ten = util.to_sig(out_ten, target_out, trg_sig, in_packed=True,
                out_packed=False)
        return out_ten

    def assign(self, rhs):
        trg_sig = self.get_sig()
        slice_expr = self.get_slice()
        if slice_expr is None:
            val = rhs.evaluate(trg_sig)
        else:
            val = self._evaluate_sliced(trg_sig, slice_expr, rhs)

        trg_dims = util.flat_dims(trg_sig)
        val_dims = val.shape.as_list()
        if not util.broadcastable(val_dims, trg_dims):
            raise RuntimeError(
                f'Actual array shape {val_dims} not broadcastable to '
                f'signature-based shape {trg_dims}')
        self.runtime.arrays[self.name] = val

    def add(self, rhs):
        if self.name not in self.runtime.arrays:
            raise RuntimeError(
                f'Cannot do += on first mention of array \'{self.name}\'')

        trg_sig = self.get_sig()
        slice_expr = self.get_slice()
        if slice_expr is None:
            val = rhs.evaluate(trg_sig)
        else:
            val = self._evaluate_sliced(trg_sig, slice_expr, rhs)

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
        tups = []
        for item in self.index_list:
            if isinstance(item, SliceExpr):
                slice_tups = item.get_basis()
                tups.extend(slice_tups)
            else:
                tups.append(item)
        return tups

    def _evaluate_sliced(self, trg_sig, slice_node):
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
        # TODO: perform the rank check here
        slice_tup = self.slice_tup()
        par_sig = self.nonslice_tups()
        idx_sig = slice_node.get_basis()
        idx_rank = slice_node.rank_sig()

        # See ops/gather_nd.et
        ixn_union_triplet = util.union_ixn(idx_sig, par_sig)
        fetch_sig, batch_sig, other_sig = ixn_union_triplet 

        # target shapes
        target_idx = batch_sig + fetch_sig
        target_par = batch_sig + [slice_tup] + other_sig
        target_res = batch_sig + fetch_sig + other_sig

        par_ten = self.get_array(target_par)
        par_ten = util.pack(par_ten, target_par)

        idx_ten = slice_node.evaluate(target_idx)
        idx_ten = util.pack(idx_ten, target_idx + [idx_rank]) 

        in_bounds = util.range_check(idx_ten, slice_tup.dims())
        idx_ten = util.flatten(idx_ten, slice_tup.dims())
        idx_ten = tf.where(in_bounds, idx_ten, -1)

        # all tensors are packed, so one dim per EinTup in batch_sig
        num_batch_dims = len(batch_sig)
        with tf.device('/GPU:0'):
            # TODO: this still happily executes on CPU when the GPU is not
            # available.  hmm...
            # gather_nd uses zero for out-of-bounds on GPU, but throws for CPU 
            result = tf.gather_nd(par_ten, idx_ten, batch_dims=num_batch_dims)

        result = util.to_sig(result, target_res, trg_sig, in_packed=True,
                out_packed=False)
        # need a method for flattening the dims but that respects broadcasting
        return result

    def evaluate(self, trg_sig):
        slice_expr = self.get_slice()
        if slice_expr is None:
            ten = self.get_array(trg_sig)
            return ten
        else:
            return self._evaluate_sliced(trg_sig, slice_expr)

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
        trg_dims = util.flat_dims(trg_sig) 
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
        return [self.key_tup, self.last_tup]

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
        src_sig = self.get_tups() 
        ten = ndrange(self.key_tup.dims())
        ten = util.to_sig(ten, src_sig, trg_sig)
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

    def get_tups(self):
        a = self.lhs.get_tups()
        b = self.rhs.get_tups()
        return util.merge_tup_lists(a, b)

    def evaluate(self, trg_sig):
        sub_sig = self.get_tups()
        lval = self.lhs.evaluate(sub_sig)
        rval = self.rhs.evaluate(sub_sig)
        ten = self.op(lval, rval)
        ten = util.to_sig(ten, sub_sig, trg_sig)
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

class StaticExpr(object):
    # An interface supporting the value() call, for use in StaticBinOpBase
    def value():
        raise NotImplementedError

class ScalarExpr(AST):
    def __init__(self, *children):
        super().__init__(*children)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.value()})'

    def rank(self):
        return 0

    def evaluate(self, trg_sig):
        src_sig = []
        val = self.value()
        dtype = tf.int32 if isinstance(val, int) else tf.float64
        ten = tf.constant(val, dtype=dtype)
        ten = util.to_sig(ten, src_sig, trg_sig)
        return ten

class IntExpr(ScalarExpr, ShapeExpr, StaticExpr):
    def __init__(self, runtime, val):
        super().__init__()
        self.runtime = runtime
        self.val = int(val)

    def value(self):
        return self.val

    def dims(self):
        return [self.value()]

class FloatExpr(ScalarExpr):
    def __init__(self, runtime, val):
        super().__init__()
        self.runtime = runtime
        self.val = val 

    def value(self):
        return self.val

class Rank(ScalarExpr, ShapeExpr, StaticExpr):
    def __init__(self, runtime, tup_name_list):
        super().__init__()
        for name in tup_name_list:
            if name not in runtime.tups:
                raise RuntimeError(
                    f'Rank tup {name} not a known EinTup.  Only EinTups '
                    f'instantiated in the program may be used as constraints. '
                    f'Known EinTups are: {runtime.tups.keys()}')

        self.runtime = runtime
        self.tups = [ self.runtime.tup(name) for name in tup_name_list ]

    def __repr__(self):
        return f'Rank({repr(self.tups)})'

    # needed to support ShapeExpr
    def dims(self):
        return [self.value()]

    def value(self):
        return sum(tup.rank() for tup in self.tups)

    def get_tups(self):
        return self.tups

class DimKind(enum.Enum):
    Star = 'Star'
    Index = 'Index'

class Dims(AST, StaticExpr):
    """
    Dims can exist in three sub-types.  
    Expression             Type       Usage
    Dims(tupname)[ind_tup] Index      statement
    Dims(tupname)          Star       constraint or tup expression
    """
    def __init__(self, runtime, kind, tup_name_list, index_expr=None):
        super().__init__()
        self.runtime = runtime
        self.tup_names = tup_name_list
        self.kind = kind
        self.base_tups = []
        self.ind_tup = None

        if self.kind == DimKind.Index:
            self.ind_tup_name = index_expr

    def __repr__(self):
        return f'{self.kind} Dims({self.base_tups})[{self.ind_tup}]'

    def set_index_tup(self, ind_tup):
        if self.kind != DimKind.Star:
            raise RuntimeError(
                f'Can only call set_index_tup on Star Dims')
        self.ind_tup = ind_tup

    def post_parse_init(self):
        for name in self.tup_names:
            if name not in self.runtime.tups:
                raise RuntimeError(
                    f'Dims argument \'{name}\' not a known EinTup.  Only EinTups'
                    f'instantiated in the program may be used as constraints. '
                    f'Known EinTups are: {[*self.runtime.tups.keys()]}')
            else:
                self.base_tups.append(self.runtime.tup(name))

        if self.kind == DimKind.Index:
            if self.ind_tup_name not in self.runtime.tups:
                raise RuntimeError(
                    f'Dims Index name \'{self.ind_tup_name}\' not known Index')
            else:
                self.ind_tup = self.runtime.tup(self.ind_tup_name)

    def evaluate(self, trg_sig):
        if self.kind == DimKind.Star:
            if len(self.base_tups) != 1:
                raise RuntimeError(
                    f'Only single-tup Star Dims can call evaluate()')
            src_sig = self.get_tups()
            ten = tf.constant(self.value(), dtype=tf.int32)
            ten = util.to_sig(ten, src_sig, trg_sig)
            return ten

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
        ten = tf.constant(util.flat_dims(self.base_tups))
        ten = util.to_sig(ten, src_sig, trg_sig)
        return ten

    def rank(self):
        return sum(tup.rank() for tup in self.base_tups)

    def value(self):
        if self.kind == DimKind.Index:
            raise RuntimeError(
                f'Cannot call value() on a {DimKind.Index.value} Dims')

        dims = util.flat_dims(self.base_tups)
        if self.kind == DimKind.Star:
            return dims
    
    def get_tups(self):
        return [self.ind_tup]

class DimsConstraint(AST, StaticExpr):
    def __init__(self, runtime, eintup_name):
        super().__init__()
        if eintup_name not in runtime.tups:
            raise RuntimeError(
                f'DimsConstraint must be initialized with known EinTup. '
                f'Got {eintup_name}.  Known EinTups:\n'
                f'{runtime.tups.keys()}')
        self.tup = runtime.tups[eintup_name]

    def value(self):
        if not self.tup.has_dims():
            self.tup.gen_dims()
        return self.tup.dims()

class RangeConstraint(AST, StaticExpr):
    def __init__(self, lo, hi, tup=None):
        self.tup = tup
        self.min = lo
        self.max = hi

    def __repr__(self):
        return f'RangeConstraint({self.min, self.max}) for {self.tup.name}'

    def value(self):
        if self.tup is None:
            return np.random.randint(self.min, self.max+1)
        else:
            vals = np.random.randint(self.min, self.max+1, self.tup.rank()) 
            return vals.tolist()

class StaticArraySlice(AST, StaticExpr):
    def __init__(self, runtime, array_name):
        super().__init__()
        if array_name not in runtime.array_sig:
            raise RuntimeError(
                f'Cannot instantiate ArraySlice as first appearance of array '
                f'\'{array_name}\'')
        sig = runtime.array_sig[array_name]
        if len(sig) != 1:
            raise RuntimeError(
                f'A StaticArraySlice must be Rank 1.  Got signature {sig}')
        self.name = array_name
        self.runtime = runtime
        self.ind_tup = sig[0]

    def __repr__(self):
        return f'StaticArraySlice({self.name})[{self.ind_tup}]'

    def value(self):
        if self.ind_tup.rank() != 1:
            raise RuntimeError(
                f'StaticArraySlice must be rank 1.  Got {self.ind_tup.rank()}')
        ten = self.runtime.arrays[self.name]
        return ten.numpy().tolist()

class StaticBinOpBase(AST, StaticExpr):
    """
    A Binary operator for use only in constraints.
    Accepts IntExpr, Rank, Dims Star types.
    If both arguments are scalar, returns a scalar.  Otherwise, returns
    a list, broadcasting one argument if necessary
    """
    # Dims types   
    def __init__(self, arg1, arg2):
        cls_name = super().__class__.__name__
        if not (isinstance(arg1, StaticExpr) and
                isinstance(arg2, StaticExpr)):
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
            return [ self.op(el, vals2) for el in vals1 ]
        elif not is_list1 and is_list2:
            return [ self.op(vals1, el) for el in vals2 ]
        elif is_list1 and is_list2:
            if len(vals1) != len(vals2):
                cls_name = super().__class__.__name__
                raise RuntimeError(f'{cls_name} got unequal length values')
            return [ self.op(el1, el2) for el1, el2 in zip(vals1, vals2) ]
        else:
            return self.op(vals1, vals2)

    def value(self):
        return self.reduce(self._map_op())

class ArithmeticBinOp(StaticBinOpBase):
    def __init__(self, arg1, arg2, op):
        super().__init__(arg1, arg2)

        opfuncs = [ operator.add, operator.sub, operator.mul, operator.truediv,
                operator.floordiv, util.ceildiv, min, max ]
        self.op_string = op
        self.op = dict(zip(['+', '-', '*', '/', '//', '//^', 'min', 'max'], opfuncs))[op]

    def __repr__(self):
        return f'ArithmeticBinOp {self.arg1} {self.op_string} {self.arg2}'

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
        if isinstance(op_vals, list):
            return all(op_vals)
        else:
            return op_vals

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
        full_dims = util.flat_dims(sig)
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
        return tf.constant(self.node.value(), dtype=tf.int32)

class TFCall(AST):
    """
    Represents a python function call of a TensorFlow function.
    Arguments can be TensorArg, Dims, Rank, or python literals.
    Python literals are wrapped with 'L(...)'
    """
    def __init__(self, func_name, call_list):
        all_nodes = [ a[1] if isinstance(a, tuple) else a for a in call_list ]
        ast_nodes = [ el for el in all_nodes if isinstance(el, AST) ]
        super().__init__(*ast_nodes)
        try:
            self.func = eval(func_name)
        except NameError as ne:
            raise RuntimeError(
                f'TFCall could not find function {func_name}: {ne}')
        self.func_name = func_name
        self.tf_call_list = call_list

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

