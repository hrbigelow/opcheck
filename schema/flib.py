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
def ceildiv(a, b):
    return np.ceil(a / b).astype(a.dtype)

def floordiv(a, b):
    return a // b

def mod(a, b):
    return np.mod(a, b)

def reduce_prod(a):
    return np.array([np.prod(a)])

def not_both_over_one(shape1, shape2):
    """
    Return true if, for all i, not(shape1[i] > 1 and shape2[i] > 1) is true
    """
    o1 = np.any(shape1 > 1)
    o2 = np.any(shape2 > 1)
    return not (o1 and o2)

def not_both_over_one_templ(shape1, shape2):
    msg =  f'"{shape1}" and "{shape2}" dimensions cannot '
    msg += f'both contain an element over 1'
    return msg

def divis_by(numer, denom):
    """
    Return a status confirming that {numer} is evenly divisible by {denom}
    """
    # this may broadcast
    rem = numer % denom
    return np.all(rem == 0)

def divis_by_templ(numer, denom):
    return f'"{numer}" dimensions must be divisible by "{denom}" dimensions'

def gen_divis_by(dummy, lo, hi):
    """
    Generate a list of shape tuples.  Each tuple has two members.  Each member
    is rank 1.  The first member is divisible by the second.  Both are in range
    [lo, hi]
    """
    q = randint(lo, hi+1)
    mul = randint(1, hi // q + 1)
    p = q * mul
    return [([p], [q])]

class PredAbove(object):
    """
    Test that components are >= min_val
    """
    def __init__(self, min_val):
        self.min_val = min_val

    def __call__(self, shape):
        return all(s >= self.min_val for s in shape)

class PredAboveTempl(object):
    def __init__(self, min_val):
        self.min_val = min_val

    def __call__(self, shape):
        return f'"{shape}" dimensions must be >= {self.min_val}'
        
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


