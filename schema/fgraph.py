import itertools
import inspect
import enum
from .error import SchemaError

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

    def __init__(self, name, func, num_positional, vararg_type):
        """
        num_positional is the number of positional arguments that func takes.
        vararg_type is the type of variable arg it has (*args, **kwargs, or
        neither)
        """
        self.name = name
        self.func = func
        self.parents = []
        self.children = []
        self.cached_val = None
        self.num_positional = num_positional
        self.vararg_type = vararg_type 

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
            raise SchemaError(
                f'{type(cls).__qualname__}: node name \'{name}\' already '
                f'exists in the registry.  Node names must be unique')
        
        pars = inspect.signature(func).parameters.values()
        args_par = next((p for p in pars if p.kind == p.VAR_POSITIONAL), None)
        kwds_par = next((p for p in pars if p.kind == p.VAR_KEYWORD), None)
        
        if args_par is not None and kwds_par is not None:
            raise SchemaError(
                f'{type(cls).__name__}: Function cannot have both **args and '
                f'**kwargs in its signature')
        wildcard = args_par or kwds_par
        pos_pars = [p for p in pars if p != wildcard]

        if wildcard is None: 
            if len(arg_names) != len(pos_pars):
                raise SchemaError(
                    f'{type(cls).__qualname__}: function takes {len(pos_pars)} '
                    f'arguments, but {len(arg_names)} arg_names provided ')
        else:
            if len(arg_names) < len(pos_pars):
                raise SchemaError(
                    f'{type(cls).__qualname__}: function takes {len(pos_pars)} '
                    f'positional arguments but only {len(arg_names)} parents '
                    f'provided.')
        num_pos_pars = len(pos_pars)
        if wildcard is None:
            vararg_type = VarArgs.Empty
        elif wildcard == args_par:
            vararg_type = VarArgs.Positional
        else:
            vararg_type = VarArgs.Keyword

        node = cls(name, func, num_pos_pars, vararg_type)
        for arg_name in arg_names:
            pa = cls.maybe_get_node(arg_name)
            if pa is None:
                raise SchemaError(
                    f'{type(cls).__qualname__}: arg_names contained '
                    f'\'{arg_name}\' but no node by that name exists in '
                    f'the registry')
            node.append_parent(pa)
        cls.registry[name] = node
        return node

    @classmethod
    def get_ordered_nodes(cls):
        return _topo_sort(cls.registry.values())

    @classmethod
    def clear_registry(cls):
        cls.registry.clear()

    @classmethod
    def maybe_get_node(cls, name):
        return cls.registry.get(name, None)

    @classmethod
    def get_node(cls, name):
        node = cls.registry.get(name, None)
        if node is None:
            raise SchemaError(
                f'{type(cls).__qualname__}: Node \'{name}\' does not exist '
                f'in the registry.')
        return node

    def add_child(self, node):
        self.children.append(node)
        node.parents.append(self)

    def append_parent(self, node):
        self.parents.append(node)
        node.children.append(self)

    def maybe_append_parent(self, name):
        """
        Append {name} as a parent of this node if the node by that name doesn't
        already exist.  It is assumed that the node called {name} already
        exists.
        """
        pa = next((n for n in self.parents if n.name == name), None)
        if pa is not None:
            return
        pa = self.maybe_get_node(name)
        if pa is None:
            raise SchemaError(
                f'{type(self).__qualname__}: Attempting to append a '
                f'non-existent node named \'{name} as a parent of node '
                f'\'{self.name}\'')
        self.append_parent(pa)

    def all_children(self):
        return self.children

    def value(self):
        """Evaluate the current node based on cached values of the parents"""
        all_args = [(n.name, n.get_cached_value()) for n in self.parents]
        pos_args = [v for n,v in all_args[:self.num_positional]]
        if self.vararg_type == VarArgs.Positional:
            args = tuple(v for n,v in all_args[self.num_positional:])
            return self.func(*pos_args, *args)
        elif self.vararg_type == VarArgs.Keyword:
            kwargs = {n:v for n,v in all_args[self.num_positional:]}
            return self.func(*pos_args, **kwargs)
        else:
            return self.func(*pos_args)

    def get_cached_value(self):
        """Retrieve the cached function evaluation value"""
        return self.cached_val

    def set_cached_value(self, val):
        """Set the cached function evaluation value"""
        self.cached_val = val

class VarArgs(enum.Enum):
    Positional = 0 # *args
    Keyword = 1    # **kwargs
    Empty = 2         # neither

def _topo_sort(nodes):
    """Sort nodes with ancestors first"""
    order = []
    todo = set(n.name for n in nodes)
    # done = set()
    def dfs(node):
        if node.name not in todo:
            return
        todo.remove(node.name)
        for ch in node.children:
            dfs(ch)
        order.append(node)
    for n in nodes:
        dfs(n)
    return order[::-1]

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
    registry = {}

    def __init__(self, *args):
        super().__init__(*args)

    def values(self):
        vals = super().value()
        try:
            iter(vals)
        except TypeError:
            raise SchemaError(f'{self}: function does not return an iterable') 
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
    registry = {}

    def __init__(self, *args):
        super().__init__(*args)
        self.pred_parents = []
        self.pred_children = []

    def add_predicate_parent(self, node):
        """
        Add a parent node which does not provide input to the function.
        Parent node evaluation must succeed before this node is evaluated.
        """
        self.pred_parents.append(node)
        node.pred_children.append(self)

    def evaluate(self):
        """
        Return whether this predicate (and all of its pre-requisites) passed.
        """
        if not all(pp.evaluate() for pp in self.pred_parents):
            return False
        if not all(p.evaluate() for p in self.parents):
            return False
        success, value = self.value()
        self.set_cached_value(value)
        return success

    def all_children(self):
        return self.pred_children + self.children

def gen_graph_iterate(topo_nodes):
    """Produce all possible settings of the graph"""
    # print('gen_graph_iterate: ', ','.join(n.name for n in visited_nodes))
    val_map = {}
    def gen_rec(i):
        if i == len(topo_nodes):
            yield dict(val_map)
            return
        node = topo_nodes[i]
        values = node.values()
        for val in values:
            node.set_cached_value(val)
            val_map[node.name] = val
            yield from gen_rec(i+1)
    yield from gen_rec(0)

def pred_graph_evaluate(topo_nodes):
    """Evaluate PredNodes in dependency order until a predicate fails"""
    for n in topo_nodes:
        if not n.evaluate():
            return n.get_cached_value()
    return None

if __name__ == '__main__':
    def unit():
        return [1,2]
    def mul(a, b):
        return [a * b]
        # return a * b
    def pr(*args):
        return '\,'.join(args)

    PredNode.clear_registry()
    PredNode.add_node('undershorts', pr)
    PredNode.add_node('pants', pr, 'undershorts')
    PredNode.add_node('socks', pr)
    PredNode.add_node('shoes', pr, 'pants', 'undershorts', 'socks')
    PredNode.add_node('watch', pr)
    PredNode.add_node('shirt', pr)
    PredNode.add_node('belt', pr, 'shirt', 'pants')
    PredNode.add_node('tie', pr, 'shirt')
    PredNode.add_node('jacket', pr, 'belt', 'tie')

    topo_order = PredNode.get_ordered_nodes()
    for n in topo_order:
        print(n.name)

    a = GenNode.add_node('a', unit)
    b = GenNode.add_node('b', unit)
    c = GenNode.add_node('c', mul, 'a', 'b')
    topo_order = GenNode.get_ordered_nodes()
    print(list(gen_graph_iterate(topo_order)))

