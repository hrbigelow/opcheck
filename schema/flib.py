import tensorflow as tf
import numpy as np
import math
from .error import *
from numpy.random import randint
from random import choice
from collections import namedtuple

"""
Library of custom generator functions and status functions to be used in
Schema API calls:

    add_index_predicate
    add_index_generator
"""

# Used by predicate functions
Index = namedtuple('Index', ['code', 'desc', 'dims'])

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
    o1 = any(s > 1 for s in shape1.dims)
    o2 = any(s > 1 for s in shape2.dims)
    if o1 and o2:
        return CustomError(
                f'Both shapes had components greater than one: '
                f'Got \'{shape1.dims}\' and \'{shape2.dims}\'')
    else:
        return Success()

def divis_by(numer, denom):
    """
    Return a status confirming that {numer} is evenly divisible by {denom}
    """
    # this may broadcast
    rem = numer.dims % denom.dims
    if np.all(rem == 0):
        return Success()
    else:
        text = []
        sn = numer.code
        sd = denom.code
        for i, mod in enumerate(rem):
            if len(numer.dims) == 1:
                ncode, nval = sn, numer.dims[0]
            else:
                ncode, nval = f'{sn}{i+1}', numer.dims[i]
            if len(denom.dims) == 1:
                dcode, dval = sd, denom.dims[0]
            else:
                dcode, dval = f'{sd}{i+1}', denom.dims[i]

            if mod != 0:
                line = (f'{ncode} is not divisible by {dcode}  '
                        f'({nval} % {dval} = {mod})')
                text.append(line)

        error_mask = (rem != 0).tolist()
        main = (f'\'{numer.desc}\' dimensions must be divisible by '
                f'\'{denom.desc}\'')
        all_text = main + '\n' + '\n'.join(text)
        return ComponentConstraintError(all_text, error_mask)

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
    pass

def pad_input_blocked(input_shape, pad_start, pad_end, block_size):
    pad_size = input_shape.dims + pad_start.dims + pad_end.dims
    if np.all(pad_size % block_size.dims == 0):
        return Success()
    else:
        return CustomError(
            f'Padded input shape must be divisible by block size. '
            f'Got padded shape {pad_size} but block size {block_size}')

def gen_pad_input_blocked(ranks_list, min_input_size, max_input_size):
    irank = ranks_list[0]
    block_size = randint(1, 10, irank)
    in_dims = randint(min_input_size, max_input_size, irank)
    start_dims = randint(1, 10, irank)
    end_dims = block_size - (in_dims + start_dims) % block_size
    return [(in_dims.tolist(), start_dims.tolist(), end_dims.tolist(),
        block_size.tolist())]


