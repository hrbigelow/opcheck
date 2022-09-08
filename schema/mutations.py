import tensorflow as tf
from .base import Kind
from numpy.random import randint
from copy import deepcopy

"""
Functions for mutating arguments to generate error test cases
"""
def insert_to_shape(shape):
    shape = deepcopy(shape)
    flat = all(isinstance(el, int) for el in shape)
    if flat:
        shape = [ [d] for d in shape ]
    rank = len(shape)
    inner = len(shape[0])
    new_pos = randint(0, rank+1)
    new_dim = randint(1, 11, inner).tolist()
    shape.insert(new_pos, new_dim)
    if flat:
        shape = [ d[0] for d in shape ]
    return shape 

def delete_from_shape(shape):
    shape = deepcopy(shape)
    rank = len(shape)
    del_pos = randint(0, rank)
    shape.pop(del_pos)
    return shape

def alter_rank(arg_val, mutate_shape_func, kind):
    if kind == Kind.DATA_TENSOR:
        shape, dtype = arg_val
        shape = mutate_shape_func(shape)
        return shape, dtype
    elif kind == Kind.SHAPE_LIST:
        return mutate_shape_func(arg_val)
    elif kind == Kind.SHAPE_TENSOR:
        ten = arg_val
        shape = ten.numpy().tolist()
        shape = mutate_shape_func(shape)
        return tf.constant(shape, dtype=ten.dtype)
    elif kind == Kind.SHAPE_TENSOR2D:
        ten = arg_val
        shape = ten.shape.as_list()
        shape = mutate_shape_func(shape)
        return tf.constant(shape, dtype=ten.dtype)
    elif kind == Kind.SHAPE_INT:
        pass # how to deal with this?


def increase_rank(arg_val, kind):
    return alter_rank(arg_val, insert_to_shape, kind)

def decrease_rank(arg_val, kind):
    return alter_rank(arg_val, delete_from_shape, kind)


def get_rank(arg_val, kind):
    """
    Gets the rank of the argument value, applying the interpretation of 'kind'
    """
    if kind == Kind.DATA_TENSOR:
        shape, dtype = arg_val
        return len(shape)
    elif kind == Kind.SHAPE_LIST:
        return len(arg_val)
    elif kind == Kind.SHAPE_TENSOR:
        return arg_val.shape[0]
    elif kind == Kind.SHAPE_TENSOR2D:
        return arg_val.shape[0]
    elif kind == Kind.SHAPE_INT:
        return arg_val









