import tensorflow as tf
import numpy as np
import math
from .error import *
from numpy.random import randint
from random import choice

"""
Library of custom generator functions and status functions to be used in
Schema API calls:

    add_index_predicate
    add_index_generator
"""
def diff_ceil(x):
    if isinstance(x, (tf.Tensor, tf.Variable)):
        return tf.grad_pass_through(tf.math.ceil)(x)
    elif isinstance(x, np.ndarray):
        return np.ceil(x)
    else:
        return math.ceil(x)

def diff_floor(x):
    if isinstance(x, (tf.Tensor, tf.Variable)):
        return tf.grad_pass_through(tf.math.floor)(x)
    elif isinstance(x, np.ndarray):
        return np.floor(x)
    else:
        return math.floor(x)

def ceildiv(a, b):
    return diff_ceil(a / b)

def floordiv(a, b):
    return diff_floor(a / b)

def mod(a, b):
    if isinstance(a, (tf.Tensor, tf.Variable)):
        return tf.math.floormod(a, b)
    elif isinstance(a, np.ndarray):
        return np.mod(a, b)
    else:
        return math.fmod(a, b)

def reduce_prod(a):
    if isinstance(a, (tf.Tensor, tf.Variable)):
        return tf.reduce_prod(a, keepdims=True)
    else:
        return np.array([np.prod(a)])

def to_dims(a):
    """
    """
def not_both_over_one(shape1, shape2):
    """
    Return a status confirming that no more than one of the shapes has
    components greater than one.
    """
    o1 = any(s > 1 for s in shape1)
    o2 = any(s > 1 for s in shape2)
    if o1 and o2:
        return CustomError(
                f'Both shapes had components greater than one: '
                f'Got \'{shape1}\' and \'{shape2}\'')
    else:
        return Success()

def gen_not_both_over_one(ranks_list, lo, hi):
    """
    Generate a list of shape tuples in range [lo, hi], in which no more than
    one shape has dimensions over 1. 
    """
    if len(ranks_list) != 2:
        raise SchemError(
            f'gen_not_both_over_one: only works with two indices')

    rank1, rank2 = ranks_list
    tup1 = ([1] * rank1, randint(lo, hi+1, rank2).tolist())
    tup2 = (randint(lo, hi+1, rank1).tolist(), [1] * rank2)
    combos = [tup1, tup2]
    return combos

def gen_range(ranks_list, lo, hi):
    """
    Generate a list of one shape tuple, with shapes in range [lo, hi]
    """
    tup = tuple(randint(lo, hi+1, r).tolist() for r in ranks_list)
    return [tup]

def gen_blocked_sizes(ranks_list, block_lo, block_hi, input_lo, input_hi):
    """
    Generate a block size in the range [block_lo, block_hi] and an input
    in the range [input_lo, input_hi].  Expect ranks_list[0] to contain the
    rank of input
    """
    block_size = randint(block_lo, block_hi+1)
    input_rank = ranks_list[0]
    vals = [i for i in range(input_lo, input_hi+1) if i % block_size == 0]
    idims = [choice(vals) for _ in range(input_rank)]
    return [(idims, [block_size])]

def gen_range_block(ranks_list, lo, hi, block_size):
    """
    Generate a list of one shape tuple, with values divisible by block_size
    """
