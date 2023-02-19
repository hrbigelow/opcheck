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

def gen_small_price():
    """Generate three valid small prices, and one invalid one"""
    yield from get_vals(1, 100, 3)
    yield 105

def gen_medium_price(small_price):
    """Generate three valid medium prices, and one invalid one"""
    yield from get_vals(small_price+1, small_price+100, 3)
    yield small_price - 1

def gen_large_price(medium_price):
    """Generate three valid large prices, and one invalid one"""
    yield from get_vals(medium_price+1, medium_price+100, 3)
    yield medium_price - 1

def gen_softdrink_revenue_tests():
    """
    Generate throw/no-throw unit tests 
    """
    for sp in gen_small_price():
        for mp in gen_medium_price(sp):
            for lp in gen_large_price(mp):
                yield { 'small_price': sp, 'medium_price': mp, 'large_price': lp }

if __name__ == '__main__':
    for kwargs in gen_softdrink_revenue_tests():
        try:
            expected_softdrink_revenue(**kwargs)
            print(kwargs, 'PASS')
        except ValueError as ex:
            print(kwargs, ex)




