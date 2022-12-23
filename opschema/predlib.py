"""
A library of predicate functions (and their template counterparts) for use with
the dims_pred API call, which are used in the base.IndexPredicate class.  There
are two types of functions in this library:

1. Predicates that will be called with individual components of index shapes
2. Predicates that are called with the entire shapes.

Note that the entire shape of an index can either be an integer list, or a
single integer (representing a rank-agnostic shape)
"""
def not_both_over_one(shape1, shape2):
    """
    Return true if, for all i, not(shape1[i] > 1 and shape2[i] > 1) is true
    """
    if isinstance(shape1, int):
        shape1 = (shape1,)
    if isinstance(shape2, int):
        shape2 = (shape2,)
    o1 = not all(s <= 1 for s in shape1)
    o2 = not all(s <= 1 for s in shape2)
    return not (o1 and o2)

def not_both_over_one_templ(shape1, shape2):
    msg =  f'{shape1} and {shape2} dimensions cannot '
    msg += f'both contain an element over 1'
    return msg

def divis_by(numer, denom):
    return denom > 0 and numer % denom == 0

def divis_by_t(numer, denom):
    return f'{numer} must be divisible by {denom}'

