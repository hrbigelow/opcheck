import tensorflow as tf
import numpy as np
from itertools import accumulate
import operator

# Assume inds[...,coord], where coord has shape [len(last_dims)]
# Flatten (and remove) the last dimension
def flatten(inds, last_dims):
    accu = accumulate(last_dims, operator.mul)
    prod = np.prod(last_dims)
    mult = tf.constant([prod // r for r in accu], dtype=tf.int32)
    inds = tf.multiply(inds, mult)
    inds = tf.reduce_add(inds, [len(last_dims)], keepdims=True)
    return inds

def range_check(inds, last_dims):
    lim = tf.constant(last_dims, dtype=tf.int32)
    above = tf.greater_equal(inds, lim)
    below = tf.less(inds, 0)
    above = tf.reduce_any(above, axis=inds.shape[-1])
    below = tf.reduce_any(below, axis=inds.shape[-1])
    out_of_bounds = tf.logical_or(above, below)
    return out_of_bounds



