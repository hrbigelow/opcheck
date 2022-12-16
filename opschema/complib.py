"""
A library of functions for use with the comp_dims API call.  These functions
can be one of two types.

1. a function accepting single dimensions and returning a single computed
dimension (and possibly trailing custom arguments)

2. a function accepting whole shapes of indexes as integer lists, and returning
an integer list.
"""
import numpy as np
import math

def dilate(s, d):
    return (s - 1) * d + 1

def dilate_t(s,  d):
    return f'({s} - 1) * {d} + 1'

def conv(i, f, padding):
    if padding == 'VALID':
        return i - f + 1
    else:
        return i

def conv_t(i, f, padding):
    if padding == 'VALID':
        return f'{i} - {f} + 1'
    else:
        return i

def tconv(i, f, padding):
    if padding == 'VALID':
        return i + f - 1
    else:
        return i

def tconv_t(i, f, padding):
    if padding == 'VALID':
        return f'{i} + {f} - 1'
    else:
        return i

def ceildiv(a, b):
    return math.ceil(a / b)

def mod(a, b):
    return a % b

def reduce_prod(a):
    return int(np.prod(a))

