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
    return s + max(0, s-1) * d

def dilate_t(s,  d):
    return f'{s} + max(0, {s}-1) * {d}'

def conv(i, f, padding):
    # return size of convolution without stride
    if padding == 'VALID':
        return i - f + 1
    else:
        return i

def conv_t(i, f, padding):
    if padding == 'VALID':
        return f'{i} - {f} + 1'
    else:
        return i

def strided_conv(i, f, s, padding):
    if padding == 'VALID':
        return ceildiv(i - f + 1, s)
    else:
        return ceildiv(i, s)

def strided_conv_t(i, f, s, padding):
    if padding == 'VALID':
        return f'ceil(({i} + {f} - 1) / {s})'
    else:
        return f'ceil({i} / {s})' 

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

