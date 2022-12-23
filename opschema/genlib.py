import math

"""
Functions for use in gen_dims_func API call.  These are generator functions
which generate lo, hi integer pairs defining the range for dims in test case
generation.  If lo and/or hi are None, this indicates an open range.
"""
def get_factors(n):
    nsq = math.floor(math.sqrt(n))
    fac = [n]
    for i in range(2, nsq+1):
        if i % n == 0:
            fac.append(i)
    return fac

class WrapParams(object):
    def __init__(self, gen_func, *pars):
        self.gen_func = gen_func
        self.trailing_pars = pars

    def __call__(self, *args):
        yield from self.gen_func(*args, *self.trailing_pars)

class GenRange(object):
    def __init__(self, lo, hi, gen_zero=False):
        self.lo = lo
        self.hi = hi
        self.gen_zero = gen_zero

    def __call__(self, _):
        yield self.lo, self.hi
        if self.gen_zero:
            yield 0, 0

class GenFromFunc(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, _, *args):
        yield from self.func(*args)

def stride_dil(rng):
    yield 1, 1
    yield 2, None 

def below_above(rng, mid):
    if mid > 0:
        yield 1, mid
    yield max(0, mid + 1), 1000
    yield 0, 0

def interval(rng, lo, hi):
    yield lo, hi

def mod_padding(rng, input, block, max_total_pad):
    """
    yield ranges for s and e such that (s + input + e) % block == 0
    also yield random ranges as well.
    max_total_pad must be >= largest possible block size
    """
    assert block != 0, 'mod_padding assert failed'
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

def group_channels(rng, max_input_val):
    """
    Generate a pair of input, filter channel dims, in which input is divisible
    by filter
    """
    inp = rng.choice(range(1, max_input_val+1))
    factors = get_factors(inp)
    inp_tup = (inp, inp)
    filt = rng.choice(factors)
    filt_tup = filt, filt

    fake = next(filter(lambda i: inp % i != 0, range(1, inp+1)), inp)
    
    yield inp_tup, inp_tup
    yield inp_tup, filt_tup
    yield inp_tup, (0, 0)
    yield inp_tup, (fake, fake)
    yield (0, 0), filt_tup
    yield (0, 0), (0, 0)


