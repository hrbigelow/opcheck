"""
Functions for use in gen_dims_func API call.  These are generator functions
which generate lo, hi integer pairs defining the range for dims in test case
generation.  If lo and/or hi are None, this indicates an open range.
"""
class WrapParams(object):
    def __init__(self, gen_func, *pars):
        self.gen_func = gen_func
        self.trailing_pars = pars

    def __call__(self, *args):
        yield from self.gen_func(*args, *self.trailing_pars)

def stride_dil(rng):
    yield 1, 1
    yield 2, None 

def below_above(rng, mid):
    if mid > 0:
        yield 1, mid
    yield mid + 1, None 

def interval(rng, lo, hi):
    yield lo, hi

def mod_padding(rng, input, block, max_total_pad):
    """
    yield ranges for s and e such that (s + input + e) % block == 0
    also yield random ranges as well.
    max_total_pad must be >= largest possible block size
    """
    rem = (block - (input % block)) % block
    max_mul = max_total_pad - rem
    max_t = max_mul // block
    t = rng.randint(0, max_t)
    tot = t * block + rem
    beg = rng.randint(0, tot)
    end = tot - beg
    yield ((beg, beg), (end, end))
    yield ((0, block), (0, block))

def divis_by(rng, denom, max_val):
    """
    Generate a list of shape tuples.  Each tuple has two members.  Each member
    is rank 1.  The first member is divisible by the second.  Both are in range
    [lo, hi]
    """
    max_mul = max_val // denom
    mul = rng.randint(1, max_mul)
    val = denom * mul

    # generate one fake value not equal to val
    fake1, fake2 = rng.sample(range(1, max_val+1), 2)
    fake = fake1 if fake1 != val else fake2
    yield val, val
    yield fake, fake

