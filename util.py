import tensorflow as tf
import numpy as np
from itertools import accumulate
import operator

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

def order_union_ixn(a, b):
    a_extra = [ el for el in a if el not in b ]
    b_extra = [ el for el in b if el not in a ]
    ab_ixn =  [ el for el in a if el in b ]
    return a_extra, ab_ixn, b_extra


