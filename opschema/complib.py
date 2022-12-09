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

def filter_pad(filt, dilation):
    return (filt - 1) * dilation + 1

def filter_pad_t(filt, dilation):
    return f'({filt} - 1) * {dilation} + 1'

def ceildiv(a, b):
    return math.ceil(a / b)

def mod(a, b):
    return a % b

def reduce_prod(a):
    return np.prod(a)

def conv(input, pad_filter, stride, padding):
    if padding == 'VALID':
        out = ceildiv(input - pad_filter + 1, stride)
    else:
        out = ceildiv(input, stride)
    return out

def conv_t(input, pad_filter, stride, padding):
    if padding == 'VALID':
        tem = f'ceil(({input} - {pad_filter} + 1) / {stride})'
    else:
        tem = f'ceil({input} / {stride})' 
    return tem

# transpose conv output calculation
def tconv(input, pad_filter, stride, padding):
    if padding == 'VALID':
        out = input - (pad_filter - 1) * stride
    else:
        out = input
    return out

def tconv_t(input, pad_filter, stride, padding):
    if padding == 'VALID':
        txt = f'{input} - ({pad_filter} - 1) * {stride}'
    else:
        txt = input
    return txt

