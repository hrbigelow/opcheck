import random
import math

def feasible_region(k, sum_min_map, sum_max_map, sum_equiv_map, sum_const_map):
    """
    Enumerate all k-integer tuples with non-negative integers.
    {sum_max_map} and {sum_min_map} are maps with integer tuple keys.  Each key
    denotes a set of digits to be summed.  The value denotes the maximum
    (respectively minimum) value that the sum of digits is allowed to have.
    {sum_equiv_list} is a list of pairs of indices, whose sums must be equal
    {sum_const_map} is a map of integer tuple => value
    """
    def sum_digits(d, inds):
        return sum(digits[i] for i in inds)

    def upper_bound_valid(d):
        return all(sum_digits(d, inds) <= ub for inds, ub in
                sum_max_map.items())

    def lb_valid(d):
        return all(sum_digits(d, inds) >= lb for inds, lb in
                sum_min_map.items())

    def eq_valid(d):
        return all(sum_digits(d, inds1) == sum_digits(d, inds2) for inds1,
                inds2 in sum_equiv_map.items()) 

    def const_valid(d):
        return all(sum_digits(d, inds) == val for inds, val in
                sum_const_map.items())
    t = 0
    digits = [0] * k 
    while True:
        if upper_bound_valid(digits):
            t = 0
            if lb_valid(digits) and eq_valid(digits) and const_valid(digits):
                yield list(digits)
        else:
            digits[t] = 0
            t += 1
            if t == k:
                break
        digits[t] += 1

def bsearch_integers(k, min_val, max_val, val_func):
    """
    Conduct binary search over a space of {k} real numbers until {val_func}(i1,
    ..., ik) is between min_val and max_val.  val_func must be strictly
    increasing in all arguments.
    """
    niter = 0
    space = [random.randint(3, 10) for _ in range(k)]
    space = [1.0] * k
    steps = [1.0] * k
    less = [True] * k
    factor = 1.2
    ind = 0
    while True:
        ind = (ind + 1) % k
        niter += 1
        ints = [math.floor(s) for s in space]
        val = val_func(ints)
        if val < min_val:
            # ind = random.randint(0, k-1)
            if less[ind]:
                steps[ind] *= factor
            else:
                steps[ind] = 1.0
            space[ind] += steps[ind]
            less[ind] = True

        elif val > max_val:
            # ind = random.randint(0, k-1)
            if less[ind]:
                steps[ind] = 1.0
            else:
                steps[ind] *= factor
            space[ind] -= steps[ind]
            less[ind] = False
            space[ind] = max(1.0, space[ind])
        else:
            break
        if niter % 10 == 0:
            print(f'ind: {ind}, niter: {niter}, val={val}, space: {space}')
    return ints, niter

def is_iterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True

