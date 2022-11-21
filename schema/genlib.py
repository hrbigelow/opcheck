"""
Functions for use in gen_dims_func API call.  These are generator functions
which generate lo, hi integer pairs defining the range for dims in test case
generation.  If lo and/or hi are None, this indicates an open range.
"""
import random

def stride_dil():
    yield 1, 1
    yield 2, None 

def below_above(mid):
    if mid > 0:
        yield 1, mid
    yield mid + 1, None 

def interval(lo, hi):
    yield lo, hi

def mod_padding(input, block, max_total_pad):
    """
    yield ranges for s and e such that (s + input + e) % block == 0
    also yield random ranges as well.
    max_total_pad must be >= largest possible block size
    """
    rem = (block - (input % block)) % block
    max_mul = max_total_pad - rem
    max_t = max_mul // block
    t = random.randint(0, max_t)
    tot = t * block + rem
    beg = random.randint(0, tot)
    end = tot - beg
    yield ((beg, beg), (end, end))
    yield ((0, block), (0, block))

def divis_by(denom, max_val):
    """
    Generate a list of shape tuples.  Each tuple has two members.  Each member
    is rank 1.  The first member is divisible by the second.  Both are in range
    [lo, hi]
    """
    max_mul = max_val // denom
    mul = random.randint(1, max_mul)
    val = denom * mul

    # generate one fake value not equal to val
    fake1, fake2 = random.sample(range(1, max_val+1), 2)
    fake = fake1 if fake1 != val else fake2
    yield val, val
    yield fake, fake

