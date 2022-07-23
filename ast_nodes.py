import tensorflow as tf
import numpy as np
import enum
import operator
import util

def merge_bases(slice_list):
    bases = [ s.get_basis() for s in slice_list ]
    merged_basis = bases[0]
    for basis in bases[1:]: 
        merged_basis = util.merge_tup_lists(merged_basis, basis)
    return merged_basis

def combine_slices(slice_list):
    n = len(slice_list)
    elem_shapes = [ s.elem_shape for s in slice_list ]
    basis = merge_bases(slice_list)
    elem_shape = ElemShape(slice_list)
    tens = [None] * n
    for i in range(n):
        ten = slice_list[i].evaluate(basis)
        shape = util.single_dims(basis + [elem_shapes[i]])
        tens[i] = tf.broadcast_to(ten, shape)
    ten = tf.concat(tens, -1)
    util.check_shape(ten, basis + [elem_shape], False)
    return ten, basis, elem_shape

STAR = ':'

class ShapeExpr(object):
    """
    A ShapeExpr is the base component for an array signature which can be used
    in the define_sig 'use_list'
    """
    def __init__(self):
        pass

    # see notes.txt for SliceExpr (which derives from ShapeExpr).  This returns
    # the exclusive upper bound for each component in the elements of this
    # SliceExpr.  It is used for calculating backing tensor shapes
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
        self._dims = None
        self._rank = None

        # Must be set to a parent tup (for equality constraint) or a
        # tuple range
        self.rank_parent = None
        self.rank_range = None
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
        self._rank = len(dims)
        self._dims = list(dims)

    def clear(self):
        self._rank = None
        self._dims = None

    def set_rank(self, rank):
        self._dims = None
        self._rank = rank

    def set_dims(self, dims):
        if not self.has_rank():
            raise RuntimeError( f'Cannot call set_dims when rank is not set')
        if len(dims) != self.rank(): 
            raise RuntimeError(
                f'dims received {dims} but rank is {self.rank()}')
        self._dims = list(dims)

    # calculate the rank from the rank_parent
    def calc_rank(self):
        if not self.has_rank():
            if self.rank_parent is not None:
                rank = self.rank_parent.calc_rank()
                self.set_rank(rank)
            else:
                raise RuntimeError(
                    f'calc_rank found uninitialized rank with no parent')
        return self.rank()

    def add_gen_expr(self, gen_expr):
        self.gen_expr = gen_expr

    def _find_rank_root(self):
        tup = self
        while tup.rank_parent is not None:
            tup = tup.rank_parent
        return tup
    
    def equate_rank(self, tup):
        a = self._find_rank_root()
        b = tup._find_rank_root()
        if a.name < b.name:
            b.rank_parent = a
        elif b.name < a.name:
            a.rank_parent = b
        else:
            # already have the same root
            pass 

    def set_rank_range(self, rng):
        if self.rank_range is not None:
            raise RuntimeError(
                f'EinTup {repr(self)} already has a rank range set as '
                f'{self.rank_range}.  Attempting to set it to {rng}')
        self.rank_range = rng

    def lift_rank_range(self):
        rng = self.rank_range
        if rng is None:
            return
        self.rank_range = None
        tup = self._find_rank_root()
        if tup != self and tup.rank_range is not None:
            raise RuntimeError(
                f'set_rank_range: rank range for {repr(self)} already set '
                f'on its root {repr(tup)} as {tup.rank_range}')
        tup.set_rank_range(rng)

    def get_rank_constraint_root(self):
        return self

    def gen_dims(self):
        if not self.has_rank():
            raise RuntimeError(f'Cannot call gen_dims before rank is set')
        if not self.has_dims():
            dims = self.gen_expr.calc_value()
            if isinstance(dims, int):
                dims = [dims] * self.rank()
            self.set_dims(dims)
        return self.dims()

    def has_dims(self):
        return self._dims is not None

    def has_rank(self):
        return self._rank is not None

    def dims(self):
        if not self.has_dims(): 
            raise RuntimeError('Cannot call dims() on uninitialized EinTup')
        return self._dims

    def rank(self):
        if not self.has_rank():
            raise RuntimeError(f'Cannot call rank() before rank is set')
        return self._rank

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

    def get_tups(self):
        tuplists = [ ch.get_tups() for ch in self.children ]
        merged = tuplists[0] if len(tuplists) else []
        for l in tuplists[1:]:
            merged = util.merge_tup_lists(merged, l)
        return merged

class StaticExpr(object):
    # An interface supporting the value() call, for use in StaticBinOpBase
    def value(self):
        raise NotImplementedError

    # Call during constraint resolution phase.
    def calc_value(self):
        return self.value()

    def get_rank_constraint_root(self):
        return None

class ElemShape(ShapeExpr):
    def __init__(self, shape_list):
        self.shape_list = shape_list

    def dims(self):
        return [sum(len(ex.dims()) for ex in self.shape_list)]

class GroupShape(ShapeExpr):
    def __init__(self, shape_list):
        self.shape_list = shape_list

    def dims(self):
        return [ dim for sh in self.shape_list for dim in sh.dims() ]

class FlatShape(ShapeExpr):
    def __init__(self, shape_list):
        self.shape_list = shape_list

    def dims(self):
        return [ np.prod([sh.nelem() for sh in self.shape_list]) ]

class SliceExpr(AST, ShapeExpr):
    # Each entry in an Array index_list is either a naked EinTup or a SliceExpr
    def __init__(self, basis, *children):
        super().__init__(*children)
        self.basis = basis
        self.elem_shape = ElemShape([self])

    def __repr__(self):
        return f'{self.__class__.__name__}({self.basis})'

    # The full signature is basis + [self.elem_shape] 
    def get_basis(self):
        return self.basis

    def get_rank_constraint_root(self):
        return None

    def evaluate(self, trg_basis):
        # return a tensor whose shape is broadcastable to 
        # trg_basis + [self.rank_sig()]
        raise NotImplementedError

class IntSlice(SliceExpr, StaticExpr):
    def __init__(self, runtime, val):
        super().__init__(basis=list())
        self.val = val

    def dims(self):
        return [self.value()+1]

    def value(self):
        return self.val

    def evaluate(self, trg_basis): 
        ten = tf.constant(self.val, dtype=util.tf_int)
        ten = util.to_sig(ten, [], trg_basis + [self.elem_shape])
        return ten

class RankSlice(SliceExpr, StaticExpr):
    def __init__(self, runtime, rank):
        super().__init__(basis=list())
        self.rank = rank

    # Using RankSlice as a shape
    def dims(self):
        return [self.value()+1]

    def value(self):
        return self.rank.value()

    def evaluate(self, trg_basis):
        ten = tf.constant(self.rank.value(), dtype=util.tf_int)
        ten = util.to_sig(ten, [], trg_basis + [self.elem_shape])
        return ten

class DimsSlice(SliceExpr, StaticExpr):
    def __init__(self, tup_exprs):
        super().__init__(basis=list())
        self.shapes = tup_exprs

    def __repr__(self):
        return f'DimsSlice({self.shapes})'

    def dims(self):
        return [ v+1 for v in self.value() ]

    def get_rank_constraint_root(self):
        if len(self.shapes) == 1 and isinstance(self.shapes[0], EinTup):
            return self.shapes[0]
        return None

    def value(self):
        return util.single_dims(self.shapes)

    def evaluate(self, trg_basis):
        src_basis = self.get_basis()
        ten = tf.constant(self.value(), dtype=util.tf_int)
        ten = util.to_sig(
                ten, src_basis + [self.elem_shape], 
                trg_basis + [self.elem_shape])
        return ten

class EinTupSlice(SliceExpr):
    def __init__(self, eintup):
        self.tup = eintup 
        super().__init__([self.tup])

    def __repr__(self):
        return f'EinTupSlice({self.basis})'

    def dims(self):
        return self.tup.dims()

    def get_rank_constraint_root(self):
        return self.tup

    def evaluate(self, trg_basis):
        src_basis = self.get_basis()
        ten = util.ndrange(self.tup.dims()) 
        ten = util.to_sig(ten, src_basis + [self.elem_shape], 
                trg_basis + [self.elem_shape])
        return ten

class FlattenSlice(SliceExpr):
    def __init__(self, slice_list):
        super().__init__(basis=list())
        self.slice_list = slice_list

    def __repr__(self):
        return f'FlattenSlice({self.slice_list})'
        
    def get_basis(self):
        return merge_bases(self.slice_list)

    def dims(self):
        ub = np.prod([sh.nelem() for sh in self.slice_list], dtype=np.int32)
        return [ub]

    def evaluate(self, trg_basis):
        ten, merged_basis, merged_rank = combine_slices(self.slice_list)
        ten = util.flatten_with_bounds(ten, self.slice_list)
        ten = util.to_sig(ten, merged_basis + [self.elem_shape], 
                trg_basis + [self.elem_shape])
        return ten

class SliceBinOp(SliceExpr):
    def __init__(self, runtime, lhs, rhs, op_string):
        if not isinstance(lhs, SliceExpr) or not isinstance(rhs, SliceExpr):
            raise RuntimeError(
                f'SliceBinOp can only operate on two SliceExpr instances. '
                f'Got {type(lhs)} and {type(rhs)}')
        super().__init__(basis=list())
        self.lhs = lhs
        self.rhs = rhs
        allowed_ops = ('+', '-', '*', '//', '//^', '%')
        if op_string not in allowed_ops:
            raise RuntimeError(
                f'SliceBinOp received op {op_string}.  Only allowed ops are '
                f'{allowed_ops}')

        self.op = util.ops[op_string] 
        self.scalar_op = util.scalar_ops[op_string]
        self.op_string = op_string

        if op_string in ('//', '//^', '%') and not isinstance(rhs, StaticExpr):
            raise RuntimeError(
                f'Can only divide by StaticExpr.  Got {type(rhs)}')

        self.add_rank_constraint()

    def __repr__(self):
        return f'SliceBinOp({self.lhs} {self.op_string} {self.rhs})' 

    """
    Returns the root of the Rank equality constraint subtree established by this
    SliceBinOp.  Every EinTupSlice and DimsSlice used in the SliceBinOp tree
    must have the same rank.  So, a rank equality constraint tree is built
    during SliceBinOp construction.
    """
    def get_rank_constraint_root(self):
        ltup = self.lhs.get_rank_constraint_root()
        rtup = self.rhs.get_rank_constraint_root()
        if ltup is None and rtup is None:
            return None
        elif ltup is None or ltup.name < rtup.name:
            return rtup
        else:
            return ltup

    def add_rank_constraint(self):
        ltup = self.lhs.get_rank_constraint_root()
        rtup = self.rhs.get_rank_constraint_root()
        if ltup is None or rtup is None:
            return
        ltup.equate_rank(rtup)

    def dims(self):
        # Broadcast as necessary
        scalar_types = (IntSlice, RankSlice)
        ldims = self.lhs.dims()
        rdims = self.rhs.dims()
        lvals = None
        rvals = None
        if isinstance(self.lhs, StaticExpr):
            lvals = self.lhs.value()
        if isinstance(self.rhs, StaticExpr):
            rvals = self.rhs.value()

        if isinstance(self.lhs, scalar_types):
            ldims = ldims[0:1] * len(rdims)
            if lvals is not None:
                lvals = [lvals] * len(rdims)

        if isinstance(self.rhs, scalar_types):
            rdims = rdims[0:1] * len(ldims)
            if rvals is not None:
                rvals = [rvals] * len(ldims)
        
        l_static = isinstance(self.lhs, StaticExpr)
        r_static = isinstance(self.rhs, StaticExpr)

        def err():
            raise RuntimeError(
                    f'SliceBinOp does not support this combination: '
                    f'{self.lhs} {self.op_string} {self.rhs}')

        if not l_static and r_static:
            if self.op_string in ('+','-'):
                return [ self.scalar_op(l,r) for l,r in zip(ldims, rvals) ]
            elif self.op_string in ('*','//','//^'):
                return [ self.scalar_op(l-1,r) + 1 for l,r in zip(ldims, rvals) ]
            elif self.op_string == '%':
                return [ min(l,r) for l,r in zip(ldims, rvals) ]
            else:
                err()
        elif not l_static and not r_static:
            if self.op_string == '+':
                return [ self.scalar_op(l,r)-1 for l,r in zip(ldims, rdims) ]
            elif self.op_string == '-':
                return ldims
            else:
                err()
        elif l_static and not r_static:
            if self.op_string == '+':
                return [ self.scalar_op(l,r) for l,r in zip(lvals, rdims) ]
            elif self.op_string == '-':
                return lvals
            elif self.op_string == '*':
                return [ self.scalar_op(l,r-1)+1 for l,r in zip(lvals, rdims) ]
            else:
                err()
        else: # l_static and r_static 
            if self.op_string == '-':
                return [ self.scalar_op(l,r) for l,r in zip(lvals, rvals) ]
            else:
                err()

    def get_basis(self):
        lbasis = self.lhs.get_basis()
        rbasis = self.rhs.get_basis()
        return util.merge_tup_lists(lbasis, rbasis)

    def evaluate(self, trg_basis):
        src_basis = self.get_basis()
        lten = self.lhs.evaluate(src_basis)
        rten = self.rhs.evaluate(src_basis)
        src_sig = src_basis + [self.elem_shape]
        trg_sig = trg_basis + [self.elem_shape]
        lten = tf.broadcast_to(lten, util.single_dims(src_sig))
        rten = tf.broadcast_to(rten, util.single_dims(src_sig))
        ten = self.op(lten, rten)
        ten = util.to_sig(ten, src_sig, trg_sig)
        return ten

class Array(AST):
    def __init__(self, runtime, array_name, index_list, **kwds):
        super().__init__(**kwds)
        self.runtime = runtime
        self.name = array_name
        self.index_list = index_list

    def __repr__(self):
        cls_name = self.__class__.__name__
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'{cls_name}({self.name})[{ind_list}]'

    def has_slices(self):
        return any(isinstance(sl, SliceExpr) for sl in self.index_list)

    def get_slices(self):
        return [ sl for sl in self.index_list if isinstance(sl, SliceExpr) ]

    def get_slice_subsig(self):
        sig = self.runtime.array_sig[self.name]
        z = zip(sig, self.index_list)
        return [ tup for tup, use in z if isinstance(use, SliceExpr) ]

    def get_sig(self):
        if self.name not in self.runtime.array_sig:
            raise RuntimeError(
                f'Array {self.name} does not have a registered signature')
        return self.runtime.array_sig[self.name]

    def nonslice_tups(self):
        return [ use for use in self.index_list if not isinstance(use,
            SliceExpr) ]

    def get_slice_tup(self):
        sig = self.get_sig()
        pos = self._get_slice_pos()
        return sig[pos]

    def check_index_usage(self):
        sig_list = self.runtime.array_sig[self.name]
        use_list = self.index_list
        if len(sig_list) != len(use_list):
            raise RuntimeError(
                f'Array {array_name} called with incorrect number of indices.\n'
                f'Expected {len(sig_list)} but called with {len(use_list)}')

        def match(sig, use):
            return use == STAR or sig.rank() == use.rank()

        if not all(match(sig, use) for sig, use in zip(sig_list, use_list)):
            raise RuntimeError(
                f'Array {array_name} called with incorrect ranks.\n'
                f'Signature is: {[sig.rank() for sig in sig_list]}\n'
                f'Usage is    : {[use.rank() for use in use_list]}\n')

    def get_array(self, trg_sig):
        if self.name not in self.runtime.arrays:
            raise RuntimeError(
                f'Array {self.name} used as RValue before initialization')
        self.check_index_usage()
        ten = self.runtime.arrays[self.name]
        sig = self.runtime.array_sig[self.name]
        z = zip(sig, self.index_list)
        def subst_sig(use):
            return use == STAR or isinstance(use, SliceExpr)
        use_sig = [ sig if subst_sig(use) else use for sig, use in z]
        ten = util.to_sig(ten, use_sig, trg_sig)
        return ten

    def add_rank_constraints(self, usage_list):
        for sig, use in zip(self.get_sig(), usage_list):
            sig_root = sig.get_rank_constraint_root()
            use_root = use.get_rank_constraint_root()
            if sig_root is not None and use_root is not None:
                sig_root.equate_rank(use_root)

class LValueArray(Array):
    def __init__(self, runtime, array_name, index_list):
        if array_name not in runtime.array_sig:
            runtime.array_sig[array_name] = index_list
        super().__init__(runtime, array_name, index_list)
        self.add_rank_constraints(index_list)

    def __repr__(self):
        ind_list = ','.join(ind if isinstance(ind, str) else repr(ind) 
                for ind in self.index_list)
        return f'LValueArray({self.name})[{ind_list}]'

    """
    If do_add, perform a straightforward scatter_add on the existing tensor
    (pre_ten).  Otherwise, the result of this operation is one of:
    1. the existing element from pre_ten, if absent from idx_ten
    2. the sum of all rhs elements mapped into that element from idx_ten

    To calculate this second form, it is a tf.scatter_nd 
   
    """
    def _evaluate_sliced(self, rhs, do_add):
        # see ops/scatter_nd.et
        out_sig = self.nonslice_tups()
        dest = self.get_slice_subsig()
        idx_ten, idx_sig, idx_rank = combine_slices(self.get_slices())
        idx_ten = util.flatten_with_bounds(idx_ten, dest)
        idx_rank = ElemShape([idx_rank])
        slice_, batch, elem = util.union_ixn(idx_sig, out_sig)
        if len(batch) > 0:
            raise RuntimeError(f'batched scatter unsupported')

        upd_ten = rhs.evaluate(slice_ + elem)
        upd_ten = util.pack_nested(upd_ten, [slice_, elem])
        idx_ten = util.pack_nested(idx_ten, [slice_, [idx_rank]])
        pre_ten = self.get_tensor(upd_ten.dtype)
        out = dest + elem
        out_dims = util.packed_dims_nested([dest, elem])
        pre_ten = util.to_sig(pre_ten, self.get_sig(), out, in_packed=False,
                out_packed=False)
        pre_ten = tf.reshape(pre_ten, out_dims)
        # shape_ten = tf.constant(out_dims, util.tf_int)
        with tf.device('/GPU:0'):
            # see https://github.com/tensorflow/tensorflow/issues/56567
            # will execute on CPU (which borks on out of bounds)
            # if upd_ten is int32
            upd_dtype = upd_ten.dtype
            if upd_dtype == tf.int32:
                upd_ten = tf.cast(upd_ten, tf.int64)
            if do_add:
                out_ten = tf.tensor_scatter_nd_add(pre_ten, idx_ten, upd_ten)
            else:
                out_ten = tf.scatter_nd(idx_ten, upd_ten, pre_ten.shape)
                out_ten = tf.cast(out_ten, upd_dtype)
                orig_mask = tf.constant(0, shape=pre_ten.shape, dtype=tf.int64)
                upd_mask = tf.constant(1, shape=upd_ten.shape, dtype=tf.int64)
                mask = tf.tensor_scatter_nd_max(orig_mask, idx_ten, upd_mask)
                out_ten = tf.where(tf.cast(mask, tf.bool), out_ten, pre_ten)

        out_ten = tf.cast(out_ten, upd_dtype)
        out_ten = tf.reshape(out_ten, util.packed_dims(out))
        out_ten = util.to_sig(out_ten, out, self.get_sig(), in_packed=True, 
            out_packed=False)
        return out_ten

    # this ensures the first access of an LValueArray sets the shape of the
    # tensor
    def get_tensor(self, dtype):
        if self.name not in self.runtime.arrays:
            shape = util.single_dims(self.index_list)
            self.runtime.arrays[self.name] = tf.zeros(shape, dtype)
        return self.runtime.arrays[self.name]

    def assign_or_add(self, rhs, do_add):
        self.check_index_usage()
        if self.has_slices():
            new_val = self._evaluate_sliced(rhs, do_add)
            self.runtime.arrays[self.name] = new_val
        else:
            rhs_val = rhs.evaluate(self.index_list)
            full_dims = util.single_dims(self.index_list)
            rhs_val = tf.broadcast_to(rhs_val, full_dims)
            prev_val = self.get_tensor(rhs_val.dtype)
            new_val = util.fit_to_size(rhs_val, prev_val, do_add)
            self.runtime.arrays[self.name] = new_val 
        
class RValueArray(Array):
    def __init__(self, runtime, array_name, index_list, **kwds):
        if array_name not in runtime.array_sig:
            raise RuntimeError(
                f'All arrays must first appear on the left hand side.'
                f'Array {array_name} first appears on the right.')
        super().__init__(runtime, array_name, index_list, **kwds)
        self.add_rank_constraints(index_list)

    def __repr__(self):
        return f'RValueArray({self.name})[{self.index_list}]'

    def get_tups(self):
        tups = []
        for item in self.index_list:
            if isinstance(item, SliceExpr):
                slice_tups = item.get_basis()
                tups.extend(slice_tups)
            else:
                tups.append(item)
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
        # TODO: perform the rank check here
        par_sig = self.nonslice_tups()
        slices = self.get_slices()
        idx_ten, idx_sig, idx_rank = combine_slices(slices)
        idx_ten = util.pack(idx_ten, idx_sig + [idx_rank]) 
        slice_sig = self.get_slice_subsig()

        # See ops/gather_nd.et
        ixn_union_triplet = util.union_ixn(idx_sig, par_sig)
        fetch_sig, batch_sig, elem_sig = ixn_union_triplet 

        # target shapes
        target_idx = batch_sig + fetch_sig
        target_par = batch_sig + slice_sig + elem_sig
        # TODO: eliminate this call the GroupShape
        target_par_grouped = batch_sig + [GroupShape(slice_sig)] + elem_sig
        target_res = batch_sig + fetch_sig + elem_sig

        par_ten = self.get_array(target_par)
        par_ten = util.pack(par_ten, target_par_grouped)

        idx_ten = util.to_sig(idx_ten, idx_sig + [idx_rank], target_idx +
                [idx_rank], in_packed=True, out_packed=True)

        idx_ten = util.flatten_with_bounds(idx_ten, slice_sig)

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
        self.check_index_usage()
        if self.has_slices():
            return self._evaluate_sliced(trg_sig)
        else:
            return self.get_array(trg_sig)

class ArraySlice(RValueArray, SliceExpr):
    # Represents an expression ary[a,b,c,:,e,...] with exactly one ':' and
    # the rest of the indices simple eintup names.  
    def __init__(self, runtime, array_name, index_list, **kwds):
        basis = [ t for t in index_list if t != STAR ]
        kwds['basis'] = basis
        super().__init__(runtime, array_name, index_list, **kwds)
        try:
            star_ind = index_list.index(STAR)
        except ValueError:
            raise RuntimeError(f'ArraySlice did not contain a Star index')
        sig = runtime.array_sig[array_name]
        self.ind_tup = sig[star_ind]
        self.name = array_name

    def __repr__(self):
        return f'ArraySlice({self.name})[{self.index_list}]'

    def dims(self):
        def mask(use):
            if use == STAR:
                return [False] * self.ind_tup.rank()
            else:
                return [True] * use.rank()

        mask_items = [ m for use in self.index_list for m in mask(use) ]
        marg_dims = [ i for i, m in enumerate(mask_items) if m ]
        ten = self.runtime.arrays[self.name]
        maxs = tf.reduce_max(ten, axis=marg_dims)
        return maxs.numpy().tolist()

    def evaluate(self, trg_basis):
        ten = super().evaluate(trg_basis + [self.ind_tup])
        return ten

class RandomCall(AST):
    # apply a function pointwise to the materialized arguments
    # args can be: constant or array-like
    def __init__(self, runtime, min_expr, max_expr, dtype_string):
        super().__init__(min_expr, max_expr)
        self.runtime = runtime
        self.dtype_string = dtype_string
        if dtype_string == 'INT':
            self.dtype = util.tf_int 
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
        trg_dims = util.single_dims(trg_sig) 
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
        ten = util.ndrange(self.key_tup.dims())
        ten = util.to_sig(ten, src_sig, trg_sig)
        return ten

class ArrayBinOp(AST):
    def __init__(self, runtime, lhs, rhs, op_string):
        super().__init__(lhs, rhs)
        # TODO: expand to include RankExpr, IntExpr, and add evaluate()
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
    def __init__(self, lhs, rhs, do_add=False):
        super().__init__(lhs, rhs)
        self.lhs = lhs
        self.rhs = rhs
        self.do_add = do_add

    def __repr__(self):
        op_str = '+=' if self.do_add else '='
        return f'Assign: {repr(self.lhs)} {op_str} {repr(self.rhs)}'

    def evaluate(self):
        self.lhs.assign_or_add(self.rhs, self.do_add)

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
        dtype = util.tf_int if isinstance(val, int) else tf.float64
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

class RankExpr(ScalarExpr, ShapeExpr, StaticExpr):
    def __init__(self, runtime, index_expr_list):
        super().__init__()
        self.runtime = runtime
        self.index_exprs = index_expr_list

    def __repr__(self):
        return f'RankExpr({repr(self.index_exprs)})'

    # needed to support ShapeExpr
    def dims(self):
        return [self.value()]

    def value(self):
        return sum(index_expr.rank() for index_expr in self.index_exprs)

    # to be used during the constraint resolution phase.
    # the tups's ranks may not be ready net, so this call
    # will trigger the calculation of their dependent expressions
    def calc_value(self):
        return sum(index_expr.calc_rank() for index_expr in self.index_exprs)

    def get_tups(self):
        return self.index_exprs

class DimKind(enum.Enum):
    Star = 'Star'
    Index = 'Index'

class Dims(AST, StaticExpr):
    """
    Dims can be indexed or not.  Indexed Dims 
    """
    def __init__(self, runtime, kind, shape_list, ind_tup=None):
        super().__init__()
        self.runtime = runtime
        self.shapes = shape_list
        self.kind = kind
        self.ind_tup = ind_tup 

    def __repr__(self):
        if self.kind == DimKind.Star:
            return f'Dims({self.shapes})'
        else:
            return f'Dims({self.shapes})[{self.ind_tup}]'

    def evaluate(self, trg_sig):
        if self.kind == DimKind.Star:
            if len(self.shapes) != 1:
                raise RuntimeError(
                    f'Only single-tup Star Dims can call evaluate()')
            src_sig = self.get_tups()
            ten = tf.constant(self.value(), dtype=util.tf_int)
            ten = util.to_sig(ten, src_sig, trg_sig)
            return ten

        if self.ind_tup.rank() != 1:
            raise RuntimeError(
                f'Dims Index index \'{self.ind_tup}\' must be '
                f'rank 1, got \'{self.ind_tup.rank()}\'')
        if self.ind_tup.dims()[0] != self.rank():
            shape_list = [sh.name for sh in self.shapes]
            raise RuntimeError(
                f'Index Dims index {self.ind_tup} first value '
                f'({self.ind_tup.dims()[0]}) must be equal to rank of '
                f'shapes list {shape_list} ({self.rank()})')

        src_sig = self.get_tups()
        ten = tf.constant(util.single_dims(self.shapes), dtype=util.tf_int)
        ten = util.to_sig(ten, src_sig, trg_sig)
        return ten

    def rank(self):
        return sum(sh.rank() for sh in self.shapes)

    def value(self):
        if self.kind == DimKind.Index:
            raise RuntimeError(
                f'Cannot call value() on a {DimKind.Index.value} Dims')
        dims = util.single_dims(self.shapes)
        if self.kind == DimKind.Star:
            return dims

    def get_tups(self):
        return [self.ind_tup]

class DimsConstraint(AST, StaticExpr):
    def __init__(self, eintup):
        super().__init__()
        self.tup = eintup 

    def __repr__(self):
        return f'DimsConstraint({self.tup})'

    def value(self):
        return self.tup.dims()

    def get_rank_constraint_root(self):
        return self.tup

    def calc_value(self):
        if not self.tup.has_dims():
            self.tup.gen_dims()
        return self.value()

class RankConstraint(AST, StaticExpr):
    def __init__(self, eintup):
        super().__init__()
        self.tup = eintup 

    def value(self):
        return self.tup.rank()

    def calc_value(self):
        if not self.tup.has_rank():
            self.tup.calc_rank()
        return self.value()

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

class StaticBinOpBase(AST, StaticExpr):
    """
    A Binary operator for use only in constraints.
    Accepts IntExpr, RankExpr, Dims Star types.
    If both arguments are scalar, returns a scalar.  Otherwise, returns
    a list, broadcasting one argument if necessary
    """
    # Dims types   
    def __init__(self, arg1, arg2):
        cls_name = super().__class__.__name__
        if not (isinstance(arg1, StaticExpr) and
                isinstance(arg2, StaticExpr)):
            raise RuntimeError(
                f'{cls_name} only IntExpr, RankExpr, Dims and StaticBinOpBase '
                'accepted')
        if ((isinstance(arg1, Dims) and arg1.kind == DimKind.Index) or
                (isinstance(arg2, Dims) and arg2.kind == DimKind.Index)):
            raise RuntimeError(
                f'{cls_name} does not support Index Dims')

        super().__init__(arg1, arg2)
        self.arg1 = arg1
        self.arg2 = arg2

        self.add_rank_constraint()

    # map the op to broadcasted values
    def _map_op(self, vals1, vals2):
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

    def get_rank_constraint_root(self):
        ltup = self.arg1.get_rank_constraint_root()
        rtup = self.arg2.get_rank_constraint_root()
        if ltup is None and rtup is None:
            return None
        elif ltup is None or ltup.name < rtup.name:
            return rtup
        else:
            return ltup

    def add_rank_constraint(self):
        ltup = self.arg1.get_rank_constraint_root()
        rtup = self.arg2.get_rank_constraint_root()
        if ltup is None or rtup is None:
            return
        ltup.equate_rank(rtup)

    def value(self):
        vals1 = self.arg1.value()
        vals2 = self.arg2.value()
        return self.reduce(self._map_op(vals1, vals2))

    def calc_value(self):
        vals1 = self.arg1.calc_value()
        vals2 = self.arg2.calc_value()
        return self.reduce(self._map_op(vals1, vals2))

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
        full_dims = util.single_dims(sig)
        ten = self.runtime.arrays[self.name]
        ten = tf.broadcast_to(ten, full_dims)
        return ten

class TensorWrap(AST):
    """
    Wraps a static value and produces a constant tensor
    """
    def __init__(self, runtime, node):
        if not isinstance(node, StaticExpr):
            raise RuntimeError(
                f'TensorWrap can only wrap a Dims or RankExpr instance')
        super().__init__(node)
        self.node = node

    def value(self):
        return tf.constant(self.node.value(), dtype=util.tf_int)

class TFCall(AST):
    """
    Represents a python function call of a TensorFlow function.
    Arguments can be TensorArg, Dims, RankExpr, or python literals.
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

