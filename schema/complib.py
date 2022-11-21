"""
A library of functions for use with the comp_dims API call.  These functions
can be one of two types.

1. a function accepting single dimensions and returning a single computed
dimension (and possibly trailing custom arguments)

2. a function accepting whole shapes of indexes as integer lists, and returning
an integer list.
"""
import numpy as np

def filter_pad(filt, dilation):
    return (filt - 1) * dilation + 1

def filter_pad_t(filt, dilation):
    return f'({filt} - 1) * {dilation} + 1'

def ceildiv(a, b):
    return np.ceil(a / b).astype(int)

def floordiv(a, b):
    return a // b

def mod(a, b):
    return np.mod(a, b)

def reduce_prod(a):
    return np.array([np.prod(a)])

