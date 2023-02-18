def test_range(val):
return 1 <= val <= 100

def error_range(arg, val):
    return f'`{arg}` must be in [1, 100].  Got {val}')

def gen_range():
    yield from get_vals(1, 100, 3)
    yield 105 # for testing the error condition

def test_greater(val, prev_val):
    return val > prev_val

def error_greater(arg, val, prev_arg, prev_val):
    return (f'`{arg}` must be greater than `{prev_arg}`. '
            f'Got {prev_arg} = {prev_val}, {arg} = {val}')

def gen_greater(prev_val):
    yield from get_vals(prev_val+1, prev_val+100, 3)
    yield prev_val - 1 # invalid value for testing the error condition


class InputConstraints:
    """
    Provides an API for expressing constraints on a function's inputs.
    The `
    """
    def __init__(self, op_name):
        self.name = op_name
        self.pred_graph = {} 
        self.gen_graph = {}

    def allowed_range_1_100(self, arg_name):
        """
        Declare that `arg_name` must be in the range [1, 100] 
        """
        pnode = PredicateNode(arg_name, test_range, error_range)
        self.pred_graph[arg_name] = pnode

        gnode = GeneratorNode(arg_name, gen_range)
        self.gen_graph[arg_name] = gnode

    def greater_than(self, arg_name, prev_name):
        """
        Declare that `arg_name` must be greater than `prev_name`
        """
        prev_pnode = self.pred_graph[prev_name] 
        pnode = PredicateNode(arg_name, test_greater, error_greater, prev_pnode)
        self.pred_graph[arg_name] = pnode

        prev_gnode = self.gen_graph[prev_name]
        gnode = GeneratorNode(arg_name, gen_greater, prev_gnode)
        self.gen_graph[arg_name] = gnode

    def validate(self, **kwargs):
        """
        Validate the given set of function inputs against the declared constraints.
        """
        msg = predicate_graph(self.pred_graph.values(), kwargs)
        if msg is not None:
            raise ValueError(msg)

    def generate(self):
        """
        Generate input test values
        """
        yield from generator_graph(self.gen_graph.values())

    def documentation(self):
        """
        Generate top-level documentation for the allowed inputs for this function.
        """
        # ...

def gen_softdrink_revenue_tests():
    yield from softdrink_revenue_op.generate()

