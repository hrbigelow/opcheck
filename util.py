import tensorflow as tf
import numpy as np
from itertools import accumulate
import operator
import math

# Assume inds[...,i] = c[i], compute flat[...] = WRAP(c, digits)
def flatten(inds, digit_sizes):
    if inds.shape[-1] != len(digit_sizes):
        raise RuntimeError(
            f'flatten: last dimension size must equal number of digit_sizes. '
            f'Got inds.shape[-1] = {inds.shape[-1]} and '
            f'digit sizes {digit_sizes}')
    accu = accumulate(digit_sizes, operator.mul)
    prod = np.prod(digit_sizes)
    mult = tf.constant([prod // r for r in accu], dtype=tf.int32)
    inds = tf.multiply(inds, mult)
    inds = tf.reduce_sum(inds, -1, keepdims=True)
    return inds

# return True if 0 <= inds[...,i] < last_dim_bounds[i] 
def range_check(inds, last_dim_bounds):
    lim = tf.constant(last_dim_bounds, dtype=tf.int32)
    below = tf.less(inds, lim)
    above = tf.greater_equal(inds, 0)
    below = tf.reduce_all(below, axis=-1, keepdims=True)
    above = tf.reduce_all(above, axis=-1, keepdims=True)
    in_bounds = tf.logical_and(above, below)
    return in_bounds

def union_ixn(a, b):
    a_extra = [ el for el in a if el not in b ]
    b_extra = [ el for el in b if el not in a ]
    ab_ixn =  [ el for el in a if el in b ]
    return a_extra, ab_ixn, b_extra

def broadcastable(array_dims, sig_dims):
    if len(array_dims) != len(sig_dims):
        return False
    return all(ad in (1, sd) for ad, sd in zip(array_dims, sig_dims))

def ndrange(dims):
    ten = [tf.range(e) for e in dims]
    ten = tf.meshgrid(*ten, indexing='ij')
    ten = tf.stack(ten, axis=len(dims))
    return ten

def flat_dims(tups):
    # tup.dims() may be empty, but this still works correctly
    return [ dim for tup in tups for dim in tup.dims()]

def packed_dims(tups):
    # tup.nelem() returns 1 for a zero-rank tup.  this
    # seems to work correctly.
    return [ tup.nelem() for tup in tups ]

def pack(ten, sig):
    check_shape(ten, sig, is_packed=False)
    return tf.reshape(ten, packed_dims(sig))

# used to construct a slightly order-preserving signature for
# the result of a binary op
def merge_tup_lists(a, b):
    ait, bit = iter(a), iter(b)
    ae = next(ait, None)
    be = next(bit, None)
    out = []
    while ae is not None or be is not None:
        if ae is None:
            if be not in out:
                out.append(be)
            be = next(bit, None)
        else:
            out.append(ae)
            ae = next(ait, None)
    return out


# check tensor shape against sig
def check_shape(ten, sig, is_packed):
    expect_dims = packed_dims(sig) if is_packed else flat_dims(sig)
    if ten.shape.as_list() != expect_dims:
        desc = 'packed' if is_packed else 'flat'
        raise RuntimeError(
            f'Tensor shape {ten.shape.as_list()} not consistent with '
            f'signature {desc} shape {expect_dims}')
    

# reshape / transpose ten, with starting shape src_sig, to be broadcastable to
# trg_sig.  if in_packed, expect ten shape to be in the packed form of src_sig.
# produce a tensor with either packed or flat (broadcastable) form of trg_sig
def to_sig(ten, src_sig, trg_sig, in_packed=False, out_packed=False):
    check_shape(ten, src_sig, in_packed)

    if not in_packed:
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

    if not out_packed:
        ten = tf.reshape(ten, trg_dims)

    return ten

def equal_tens(a, b, eps):
    if not a.dtype.is_floating:
        eps = 0
    if a.shape != b.shape:
        print(f'equal_tens: {a.shape} != {b.shape}')
    return (
            a.shape == b.shape and
            tf.reduce_all(tf.less_equal(tf.abs(a - b), eps)).numpy()
            )

def maybe_broadcast(a, length):
    if isinstance(a, (list, tuple)):
        if len(a) != length:
            raise RuntimeError(
                f'Cannot broadcast {a} to length {length}')
        else:
            return a
    else:
        return [a] * length

def ceildiv(a, b):
    return math.ceil(a / b)


def ceildiv_tensor(a, b):
    return tf.math.ceil(a / b)

ops = { 
        '+': tf.add,
        '-': tf.subtract,
        '*': tf.multiply,
        '//': tf.math.floordiv,
        '//^': ceildiv_tensor,
        '%': tf.math.floormod
        }

scalar_ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '//': operator.floordiv,
        '//^': ceildiv,
        '%': operator.mod
        }




