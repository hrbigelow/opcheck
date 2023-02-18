import numpy as np

def softdrink_revenue(small_price, medium_price, large_price):
    """
    Compute expected soft drink revenue as a function of pricing.

    Input constraints:
       small_price must be in [1, 100]
       medium_price must be greater than small_price
       large_price must be greater than medium_price
    """
    if small_price < 1 or small_price > 100:
        raise ValueError(
                f'`small_price` must be in [1, 100].  got {small_price}')
    if not (medium_price > small_price):
        raise ValueError(
                f'`medium_price` must be greater than `small_price`.'
                f'Got small_price = {small_price}, medium_price = {medium_price}')
    if not (large_price > medium_price):
        raise ValueError(
                f'`large_price` must be greater than `medium_price`.'
                f'Got medium_price = {medium_price}, large_price = {large_price}')
    # compute and return revenue ...

def get_vals(lo, hi, reps):
    """
    Sample `reps` different values in [lo, hi]
    """
    return list(np.random.choice(np.arange(lo, hi+1), reps, replace=False))

def gen_softdrink_revenue_tests():
    """
    Generate throw/no-throw unit tests 
    """
    for sp in get_vals(1, 100, 3) + [105]:
        for mp in get_vals(sp+1, sp+100, 3) + [sp-1]:
            for lp in get_vals(mp+1, mp+100, 3) + [mp-1]:
                yield { 'small_price': sp, 'medium_price': mp, 'large_price': lp }

if __name__ == '__main__':
    for kwargs in gen_softdrink_revenue_tests():
        try:
            expected_softdrink_revenue(**kwargs)
            print(kwargs, 'PASS')
        except ValueError as ex:
            print(kwargs, ex)




