import itertools
import inspect

"""
FuncNode represents a computation graph.  Each node has a {name} and {func}
member.  Parent nodes are stored in the order added.  When a node is evaluated,
its enclosed function receives the result of the parent node evaluations in
that order.  The function must take the same number of arguments as the node
has parents.
"""

class FuncNode(object):
    # stores all created nodes
    registry = {}

    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.parents = []
        self.children = []
        self.cached_val = None

    def __repr__(self):
        return (f'{type(self).__name__}({self.name})'
                f'[pa: ({",".join(p.name for p in self.parents)})]')

    @classmethod
    def add_node(cls, name, func, *arg_names):
        """
        Creates a new node with {name} and {func}.  {func} must take
        len(arg_names) arguments, which will be provided by parent nodes of
        those names.  {func} may have *args or **kwargs in its signature, but
        not both.  The outputs of the parents in order are passed to the
        positional arguments of {func}.  Any remaining parents are either
        passed to *args as a list of values, or passed to **kwargs as a
        dictionary, using the parent node names as the keys of the dictionary.
        """
        if name in cls.registry:
            raise RuntimeError(
                f'{type(cls).__qualname__}: node name \'{name}\' already '
                f'exists in the registry.  Node names must be unique')
        
        pars = inspect.signature(func).parameters.values()
        args_par = next((p for p in pars if p.kind == p.VAR_POSITIONAL), None)
        kwds_par = next((p for p in pars if p.kind == p.VAR_KEYWORD), None)
        
        if args_par is not None and kwds_par is not None:
            raise RuntimeError(
                f'{type(cls).__name__}: Function cannot have both **args and '
                f'**kwargs in its signature')
        wildcard = args_par or kwds_par
        pos_pars = [p for p in pars if p != wildcard]

        if wildcard is None: 
            if len(arg_names) != len(pos_pars):
                raise RuntimeError(
                    f'{type(cls).__qualname__}: function takes {len(pos_pars)} '
                    f'arguments, but {len(arg_names)} arg_names provided ')
        else:
            if len(arg_names) < len(pos_pars):
                raise RuntimeError(
                    f'{type(cls).__qualname__}: function takes {len(pos_pars)} '
                    f'positional arguments but only {len(arg_names)} parents '
                    f'provided.')

        first_missing = next((n for n in arg_names if n not in cls.registry), None)
        if first_missing is not None:
            raise RuntimeError(
                f'{type(cls).__qualname__}: arg_names contained '
                f'\'{first_missing}\' but no node by that name exists in '
                f'the registry')
        
        node = cls(name, func)
        for arg_name in arg_names:
            pa = cls.registry[arg_name]
            node.append_parent(pa)
        cls.registry[name] = node
        return node

    @classmethod
    def get_roots(cls):
        return [ n for n in cls.registry.values() if len(n.parents) == 0 ]

    @classmethod
    def clear_registry(cls):
        cls.registry.clear()

    @staticmethod
    def progeny(nodes):
        """Return all nodes whose parents are a subset of nodes"""
        all_ch = { ch for n in nodes for ch in n.children }
        progeny = [ n for n in all_ch if all(p in nodes for p in n.parents) ]
        return progeny 

    def add_child(self, node):
        self.children.append(node)
        node.parents.append(self)

    def append_parent(self, node):
        self.parents.append(node)
        node.children.append(self)

    def get_cached_value(self):
        """Retrieve the cached function evaluation value"""
        return self.cached_val
"""
Generation Graph API - a list-valued computation graph

Like an ordinary computation graph, each node represents a function, and the
node's parents represent inputs to the function.  For a Generation graph, the
functions associated with each node return a list of values rather than a
single value.  A node is then evaluated once for every possible combination of
values received from its parents.  For example, if a node has two parents and
they produce 2 and 3 values, the node is evaluated 6 times.  If the graph is
acyclic, it can be fully enumerated for all possible settings (a setting = a
set of values, one per node).  Further details:

1. Each node has a distinct name
2. A node's function is invoked with keyword arguments, using the parent node
   names + values

"""
class GenNode(FuncNode):
    def __init__(self, name, func):
        super().__init__(name, func)

    def values(self):
        kwargs = { p.name: p.get_cached_value() for p in self.parents }
        vals = self.func(**kwargs)
        try:
            iter(vals)
        except TypeError:
            raise RuntimeError(f'{self}: function does not return an iterable') 
        return vals
"""
Predicate Graph API - a computation graph for predicates

The predicate graph is a type of computation graph with a predicate function
associated with each node.

In addition to the semantics of FuncNode, a PredNode can also have 'Predicate
Parents'.  These parents do not provide values to the node's function during
evaluation.  However, they must successfully evaluate before the node can
evaluate.  This way, they enforce an evaluation order.

Expect func to return a pair (success, value)
"""
class PredNode(FuncNode):
    def __init__(self, name, func):
        super().__init__(name, func)
        self.pred_parents = []
        self.pred_children = []

    @staticmethod
    def progeny(nodes):
        all_ch = { ch for n in nodes for ch in n.pred_children + n.children}
        return [ n for n in all_ch if all(p in nodes for p in n.pred_parents
            + n.parents) ]

    def add_predicate_parent(self, node):
        """
        Add a parent node which does not provide input to the function.
        Parent node evaluation must succeed before this node is evaluated.
        """
        self.pred_parents.append(node)
        node.pred_children.append(self)

    def evaluate(self):
        if not all(pp.evaluate() for pp in self.pred_parents):
            return False
        if not all(p.evaluate() for p in self.parents):
            return False
        kwargs = { p.name: p.get_cached_value() for p in self.parents }
        success, value = self.func(**kwargs)
        if success:
            self.cached_val = value
        return success


def get_roots(nodes):
    """Find the subset of nodes having no parents"""
    return [ n for n in nodes if len(n.parents) == 0 ]

def gen_graph_iterate(nodes, val_map={}):
    """Produce all possible settings of the graph"""
    if len(nodes) == 0:
        yield dict(val_map)
        return

    next_nodes = FuncNode.progeny(nodes)
    node_value_lists = tuple(n.values() for n in nodes)
    for vals in itertools.product(*node_value_lists):
        for node, val in zip(nodes, vals):
            node.set_current_value(val)
            val_map[node.name] = val
        yield from iterate(next_nodes, val_map)

def pred_graph_evaluate(nodes):
    """Evaluate PredNodes in dependency order until a predicate fails"""
    if len(nodes) == 0:
        return True
    elif not all(n.evaluate() for n in nodes):
        return False
    else:
        next_nodes = PredNode.progeny(nodes)
        return pred_graph_evaluate(next_nodes)


if __name__ == '__main__':
    def unit():
        return [1,2]
    def mul(a, b):
        return [a * b]
        # return a * b
    a = GenNode('a', unit)
    b = GenNode('b', unit)
    c = GenNode('c', mul)
    c.add_parent(a)
    c.add_parent(b)
    print(list(iterate([a, b])))

