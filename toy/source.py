import numpy as np
from numpy.random import choice

# Helper functions
def gen_range():
    """Generate three values in range [1, 100], and one invalid one"""
    yield from choice(np.arange(1, 101), 3, replace=False)
    yield 105

def gen_greater(val):
    """Generate three valid values greater than val, and one invalid"""
    yield from choice(np.arange(val+1, val+100), 3, replace=False)
    yield val - 1


def softdrink_revenue(small_price, medium_price, large_price):
    """Input constraints:
       small_price must be in [1, 100]
       medium_price must be greater than small_price
       large_price must be greater than medium_price
       logic ...
    """
    if not (1 <= small_price <= 100):
        raise ValueError(f'`small_price` must be in [1, 100].  got {small_price}')
    if not (medium_price > small_price):
        raise ValueError(f'`medium_price` must be greater than `small_price`.'
                f'Got small_price = {small_price}, medium_price = {medium_price}')
    if not (large_price > medium_price):
        raise ValueError(f'`large_price` must be greater than `medium_price`.'
                f'Got medium_price = {medium_price}, large_price = {large_price}')
    # compute and return revenue ...

def softdrink_revenue_tests():
    """Generate throw/no-throw unit tests"""
    for sp in gen_range():
        for mp in gen_greater(sp):
            for lp in gen_greater(mp):
                yield { 'small_price': sp, 'medium_price': mp, 'large_price': lp }


def test_range(val):
    return 1 <= val <= 100

def test_greater(val, prev_val):
    return val > prev_val

def error_range(arg, val):
    return f'`{arg}` must be in [1, 100].  Got {val}'

def error_greater(arg, val, prev_arg, prev_val):
    return (f'`{arg}` must be greater than `{prev_arg}`. '
            f'Got {prev_arg} = {prev_val}, {arg} = {val}')

class PredicateNode:
    """
    A computation graph node which holds a test + error function pair
    """
    def __init__(self, name, test_func, err_func, *parents):
        self.name = name
        self.test_func = test_func
        self.err_func = err_func
        self.parents = list(parents)

    def test(self, test_setting):
        """call test_func with self and parent values from `test_setting`"""
        names = (self.name,) + tuple(p.name for p in self.parents)
        values = tuple(test_setting[name] for name in names)
        return self.test_func(*values)

    def error(self, test_setting):
        """call err_func with self and parent (name, value)'s from `test_setting`"""
        names = (self.name,) + tuple(p.name for p in self.parents)
        pairs = tuple((n, test_setting[n]) for n in names)
        return self.err_func(*itertools.chain(*pairs))

def evaluate_predicate_graph(nodes, **test_vals):
    """
    Evaluate `test_vals` against the predicate graph defined by `nodes`
    Returns: None for success, or an error message string
    """
    nodes = topo_sort(nodes)

    def _rec(remain):
        if len(remain) == 0:
            return None
        node, *remain = remain
        if node.test(test_vals):
            return _rec(remain)
        else:
            return node.error(test_vals)

    return _rec(nodes) 

def predicate_test():
    node1 = PredicateNode('small', test_range, error_range)
    node2 = PredicateNode('medium', test_greater, error_greater, node1)
    node3 = PredicateNode('large', test_greater, error_greater, node2)
    nodes = [node1, node2, node3]
    print(evaluate_predicate_graph(nodes, small=10, medium=20, large=30))
    print(evaluate_predicate_graph(nodes, small=10, medium=5, large=30))

class GeneratorNode:
    # A computation graph node which holds a generator as its function
    def __init__(self, name, gen_func, *parents):
        self.name = name
        self.gen_func = gen_func
        self.parents = list(parents)

    def values(self, current_setting):
        parent_values = tuple(current_setting[p.name] for p in self.parents)
        yield from self.gen_func(*parent_values)

def generate_all(nodes):
    """
    Generate all combinations from nodes
    nodes: topological order of GeneratorNodes
    """
    current_vals = { n.name: None for n in nodes }

    def _genrec(remain):
        if len(remain) == 0:
            yield dict(current_vals)
            return

        node, *remain = remain
        for val in node.values(current_vals):
            current_vals[node.name] = val
            yield from _genrec(remain)

    yield from _genrec(nodes)


class OpSchema:
    """
    Provides an API for expressing constraints on a function's inputs.
    The `
    """
    def __init__(self, op_name):
        self.name = op_name
        self.pred_graph = {} 
        self.gen_graph = {}

    def in_range100(self, arg_name):
        """Declare that `arg_name` must be in the range [1, 100]"""
        pnode = PredicateNode(arg_name, test_range, error_range)
        self.pred_graph[arg_name] = pnode

        gnode = GeneratorNode(arg_name, gen_range)
        self.gen_graph[arg_name] = gnode

    def greater_than(self, arg_name, prev_name):
        """Declare that `arg_name` must be greater than `prev_name`"""
        prev_pnode = self.pred_graph[prev_name] 
        pnode = PredicateNode(arg_name, test_greater, error_greater, prev_pnode)
        self.pred_graph[arg_name] = pnode

        prev_gnode = self.gen_graph[prev_name]
        gnode = GeneratorNode(arg_name, gen_greater, prev_gnode)
        self.gen_graph[arg_name] = gnode

    def validate(self, **kwargs):
        """Validate the given set of function inputs against the declared constraints"""
        msg = predicate_graph(self.pred_graph.values(), kwargs)
        if msg is not None:
            raise ValueError(msg)

    def generate(self):
        """Generate input test values"""
        yield from generator_graph(self.gen_graph.values())

    def documentation(self):
        """Generate top-level documentation for the allowed inputs for this function"""
        pass


def softdrink_schema():
    op = OpSchema('softdrink_revenue')
    op.in_range100('small_price')
    op.greater_than('medium_price', 'small_price')
    op.greater_than('large_price', 'medium_price')
    return op


def softdrink_revenue_tests_new():
    op = softdrink_schema()
    yield from op.generate()


def generate_nearest(nodes, observed_vals, max_edits):
    """
    Generate all values from nodes within max_edits of observed_vals.
    nodes: a set of GeneratorNodes in topological order
    observed_vals: a map of node.name => value
    max_edits: maximum edit distance of a generated set to report
    """
    current_vals = { n.name: None for n in nodes }
    avail_edits = [max_edits]

    def _genrec(remain):
        if len(remain) == 0:
            yield dict(current_vals)
            return

        node, *remain = remain
        for val in node.values(current_vals):
            current_vals[node.name] = val
            if val == observed_vals[node.name]:
                yield from _genrec(remain)
            elif avail_edits[0] > 0:
                avail_edits[0] -= 1
                yield from _genrec(remain)
                avail_edits[0] += 1

    yield from _genrec(nodes)

def softdrink_revenue_new(small_price, medium_price, large_price):
    op = softdrink_schema()
    op.validate(small_price, medium_price, large_price)
    # logic ...


if __name__ == '__main__':
    for kwargs in softdrink_revenue_tests():
        try:
            expected_softdrink_revenue(**kwargs)
            print(kwargs, 'PASS')
        except ValueError as ex:
            print(kwargs, ex)

