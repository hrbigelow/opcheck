import itertools

class PredicateNode:
    # A Computation graph node which holds a predicate (and associated message) function
    # as its function.
    def __init__(self, name, pred_func, msg_func, *parents):
        self.name = name
        self.pred_func = pred_func
        self.msg_func = msg_func
        self.parents = list(parents)

    def test(self, test_setting):
        # executes the predicate with the current graph setting
        self_value = test_setting[self.name]
        parent_values = tuple(test_setting[p.name] for p in self.parents)
        return self.pred_func(self_value, *parent_values)

    def error(self, test_setting):
        # generates the user-facing error message when the predicate fails
        self_value = test_setting[self.name]
        parent_names = tuple(p.name for p in self.parents)
        parent_values = tuple(test_setting[name] for name in parent_names)
        parent_names_vals = tuple(itertools.chain(*zip(parent_names, parent_values)))
        self_value = test_setting[self.name]
        return self.msg_func(self.name, self_value, *parent_names_vals)


def predicate_graph(nodes, **test_vals):
    """
    nodes: predicate graph nodes 
    test_vals: map of node name => value 
    Returns: None for success, or an error message string
    """
    nodes = topo_sort(nodes)

    def _itergraph(remain):
        if len(remain) == 0:
            return None
        node, *remain = remain
        self_val = test_vals[node.name]
        if node.test(self_val, test_vals):
            return _itergraph(remain)
        else:
            return node.error(node.name, test_vals)

    return _itergraph(nodes) 


def test_range(val):
    return 1 <= val <= 100

def error_range(arg, val):
    return f'`{arg}` must be in [1, 100].  Got {val}')

def test_greater(val, prev_val):
    return val > prev_val

def error_greater(arg, val, prev_arg, prev_val):
    return (f'`{arg}` must be greater than `{prev_arg}`. '
            f'Got {prev_arg} = {prev_val}, {arg} = {val}')

class InputConstraints:
    """
    Provides an API for expressing constraints on a function's inputs.
    The `
    """
    def __init__(self, op_name):
        self.name = op_name
        self.pred_graph = {} 

    def allowed_range_1_100(self, arg_name):
        """
        Declare that `arg_name` must be in the range [1, 100] 
        """
        pnode = PredicateNode(arg_name, test_range, error_range)
        self.pred_graph[arg_name] = pnode

    def greater_than(self, arg_name, prev_name):
        """
        Declare that `arg_name` must be greater than `prev_name`
        """
        prev_pnode = self.pred_graph[prev_name] 
        pnode = PredicateNode(arg_name, test_greater, error_greater, prev_pnode)
        self.pred_graph[arg_name] = pnode

    def validate(**kwargs):
        """
        Validate the given set of function inputs against the declared constraints.
        """
        msg = predicate_graph(self.pred_graph.values(), kwargs)
        if msg is not None:
            raise ValueError(msg)

softdrink_revenue_op = InputConstraints('softdrink_revenue')
softdrink_revenue_op.allowed_range_1_100('small_price')
softdrink_revenue_op.greater_than('medium_price', 'small_price')
softdrink_revenue_op.greater_than('large_price', 'medium_price')

def softdrink_revenue(small_price, medium_price, large_price):
    softdrink_revenue_op.validate(small_price, medium_price, large_price)
    # compute and return revenue ...



