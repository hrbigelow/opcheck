def feasible_region(k, sum_min_map, sum_max_map, sum_equiv_list, sum_const_map):
    """
    Enumerate all k-integer tuples with non-negative integers.
    {sum_max_map} and {sum_min_map} are maps with integer tuple keys.  Each key
    denotes a set of digits to be summed.  The value denotes the maximum
    (respectively minimum) value that the sum of digits is allowed to have.
    {sum_equiv_list} is a list of pairs of indices, whose sums must be equal
    {sum_const_map} is a map of integer tuple => value
    """
    def sum_digits(inds):
        return sum(digits[i] for i in inds)

    def upper_bound_valid():
        return all(sum_digits(inds) <= ub for inds, ub in sum_max_map.items())

    def other_valid():
        return (
                all(sum_digits(inds) >= lb 
                    for inds, lb in sum_min_map.items()) and
                all(sum_digits(inds1) == sum_digits(inds2) 
                    for inds1, inds2 in sum_equiv_list) and
                all(sum_digits(inds) == val for inds, val in
                    sum_const_map.items())
                )
    t = 0
    digits = [0] * k 
    while True:
        if upper_bound_valid():
            t = 0
            if other_valid():
                yield list(digits)
        else:
            digits[t] = 0
            t += 1
            if t == k:
                break
        digits[t] += 1

def bsearch_integers(k, min_val, max_val, val_func):
    """
    Conduct binary search over a space of {k} integers until {val_func}(i1, ...,
    ik) is between min_val and max_val.  val_func is strictly increasing in all
    arguments.
    """
    ints = [random.randint(3, 10) for _ in range(k)]
    while True:
        val = val_func(*ints)
        if val < min_val:
            min_ind = ints.index(min(ints))
            ints[min_ind] *= 2
        elif val > max_val:
            max_ind = ints.index(max(ints))
            ints[max_ind] //= 2
        else:
            break
    return ints
