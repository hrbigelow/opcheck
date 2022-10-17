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

