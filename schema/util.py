import random
import math
from .error import SchemaError, Success

def feasible_region(k, min_map, max_map):
    """
    Enumerate all k-integer tuples with non-negative integers.
    Constraints:
    {max_map}: (i1,i2,...) => max_val
    {min_map}: (i1,i2,...) => min_val

    These are processed to constrain the sum digits[i1] + digits[i2] + ...
    either above, below, or equal.
    """
    def ub_valid(d):
        return all(sum(d[i] for i in ixs) <= ub for ixs, ub in max_map.items())

    def lb_valid(d):
        return all(sum(d[i] for i in ixs) >= lb for ixs, lb in min_map.items())

    t = 0
    digits = [0] * k 
    while True:
        if any(d > 100 for d in digits):
            raise SchemaError(
                f'feasible_region: region exceeds 100.  Possible missing '
                f'constraint')
        if ub_valid(digits):
            t = 0
            if lb_valid(digits):
                yield list(digits)
        else:
            digits[t] = 0
            t += 1
            if t == k:
                break
        digits[t] += 1
        # print('digits: ', digits)

def is_iterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True

def dtype_combos(k, dtypes, tests, max_errors, ranks, layout):
    """
    Yield all k-wise dtype combos of {dtypes} which fail at most {max_errors}
    tests.  Each test in {tests} is a function object with the following
    members:
    
    t.status:  a SchemaStatus object describing the error
    t.__call__(self, config):  returns True or False
    t.left_ind(self):  returns the index of the left-most input digit
    """
    def increment(digits, s, e, D):
        # increment the sub-space digits[s:e], assuming alphabet size D
        t = s
        while t != e and digits[t] == D-1:
            digits[t] = 0
            t += 1
        if t == e:
            return False
        digits[t] += 1
        return True

    digits = [0] * k
    D = len(dtypes)

    while True:
        dtype_tup = tuple(dtypes[d] for d in digits)
        fail = tuple(t for t in tests if not t(dtype_tup, ranks, layout))
        if len(fail) <= max_errors:
            l = 0
            if len(fail) == 0:
                yield Success, dtype_tup
            else:
                yield fail[0].status_type, dtype_tup
        else:
            l = min(t.left_ind() for t in fail)
            digits[:l] = [0] * l

        if not increment(digits, l, k, D):
            break

