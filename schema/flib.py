import tensorflow as tf
from .error import *
from numpy.random import randint

"""
Library of custom generator functions and status functions to be used in
Schema API calls:

    add_index_predicate
    add_index_generator
"""
def diff_ceil(x):
    return tf.grad_pass_through(tf.math.ceil)(x)

def ceildiv(a, b):
    return diff_ceil(a / b)

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




