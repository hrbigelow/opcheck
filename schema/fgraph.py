import itertools
import inspect
import enum
from .error import SchemaError

def node_name(func_node_class, name=None):
    if name is None:
        return func_node_class.__name__
    else:
        return f'{func_node_class.__name__}({name})'

class NodeFunc(object):
    def __init__(self, name=None):
        self.sub_name = name

    @property
    def name(self):
        return node_name(self.__class__, self.sub_name)

    def __call__(self):
        raise NotImplementedError

class FuncNode(object):
    """
    Represents a computation graph.  each instance wraps a NodeFunc.  Parent
    nodes are stored in the order added.  When a node is evaluated, its
    enclosed NodeFunc receives the result of the parent node evaluations in
    that order.  The function must take the same number of arguments as the
    node has parents.
    """
        # stores all created nodes
    registry = None

    def __init__(self, func, num_named_pars, vararg_type):
        """
        num_named_pars is the number of named parameters that func takes. (any
        arguments that are not *args or **kwargs).  vararg_type is the type of
        variable arg it has (*args, **kwargs, or neither)
        """
        self.name = func.name 
        self.sub_name = func.sub_name
        self.func = func
        self.parents = []
        self.children = []
        self.cached_val = None
        self.num_named_pars = num_named_pars
        self.vararg_type = vararg_type 

    def __repr__(self):
        return (f'{type(self).__name__}({self.name})'
                f'[pa: {",".join(p.name for p in self.parents)}]')

    def clone_node_only(self):
        return type(self)(self.func, self.num_named_pars, self.vararg_type)

    @classmethod
    def add_node(cls, func, *parents):
        """
        Creates a new node enclosing {func} as its function.  The name of the
        node is defined by {func}.name.  {func} must take len(parents)
        arguments, which will be provided by parent nodes of those names.
        {func} may have *args or **kwargs in its signature, but not both.  The
        outputs of the parents in order are passed to the positional arguments
        of {func}.  Any remaining parents are either passed to *args as a list
        of values, or passed to **kwargs as a dictionary, using the parent node
        names as the keys of the dictionary.
        """
        if cls.registry is None:
            raise RuntimeError(
                f'{type(cls).__qualname__}: registry is not set.  Call '
                f'set_registry(reg) with a map object first')

        if func.name in cls.registry:
            raise SchemaError(
                f'{type(cls).__qualname__}: node name \'{func.name}\' already '
                f'exists in the registry.  Node names must be unique')
        
        pars = inspect.signature(func).parameters.values()
        args_par = next((p for p in pars if p.kind == p.VAR_POSITIONAL), None)
        kwds_par = next((p for p in pars if p.kind == p.VAR_KEYWORD), None)
        
        if args_par is not None and kwds_par is not None:
            raise SchemaError(
                f'{type(cls).__name__}: Function cannot have both **args and '
                f'**kwargs in its signature')
        wildcard = args_par or kwds_par
        named_pars = [p for p in pars if p != wildcard]

        if wildcard is None: 
            if len(parents) != len(named_pars):
                raise SchemaError(
                    f'{type(cls).__qualname__}: function takes {len(named_pars)} '
                    f'arguments, but {len(parents)} parents provided ')
        else:
            if len(parents) < len(named_pars):
                raise SchemaError(
                    f'{type(cls).__qualname__}: function takes {len(named_pars)} '
                    f'positional arguments but only {len(parents)} parents '
                    f'provided.')
        num_named_pars = len(named_pars)
        if wildcard is None:
            vararg_type = VarArgs.Empty
        elif wildcard == args_par:
            vararg_type = VarArgs.Positional
        else:
            vararg_type = VarArgs.Keyword

        node = cls(func, num_named_pars, vararg_type)
        for pa in parents:
            node.append_parent(pa)
        cls.registry[func.name] = node
        return node 

    @classmethod
    def get_ordered_nodes(cls):
        if cls.registry is None:
            raise RuntimeError(
                f'{type(cls).__qualname__}: Registry not set.  call '
                f'set_registry() first')
        return _topo_sort(cls.registry.values())

    @classmethod
    def set_registry(cls, reg):
        old_reg = cls.registry
        cls.registry = reg
        return old_reg

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

    @classmethod
    def find_unique_name(cls, prefix):
        names = { n for n in cls.registry.keys() if n.startswith(prefix) }
        if len(names) == 1:
            return names.pop()
        else:
            return None

    @classmethod
    def find_unique(cls, prefix):
        found = []
        for name, node in cls.registry.items():
            if name.startswith(prefix):
                found.append(node)
        if len(found) == 1:
            return found[0]
        else:
            return None

    def add_child(self, node):
        self.children.append(node)
        node.parents.append(self)

    def append_parent(self, node):
        self.parents.append(node)
        node.children.append(self)

    def maybe_append_parent(self, node):
        """
        Append {node} as a parent of this node if not already a parent 
        """
        pa = next((n for n in self.parents if n.name == node.name), None)
        if pa is not None:
            return
        self.append_parent(node)

    def all_children(self):
        return self.children

    def value(self):
        """
        Evaluate the current node based on cached values of the parents
        """
        all_args = [(n.func.sub_name, n.get_cached_value()) for n in
                self.parents]
        pos_args = [v for n,v in all_args[:self.num_named_pars]]
        if self.vararg_type == VarArgs.Positional:
            args = tuple(v for n,v in all_args[self.num_named_pars:])
            return self.func(*pos_args, *args)
        elif self.vararg_type == VarArgs.Keyword:
            kwargs = {}
            for pos in range(self.num_named_pars, len(all_args)):
                name, val = all_args[pos]
                if name is None:
                    raise SchemaError(
                        f'All **kwargs argument parents must have sub_names. '
                        f'{self.__class__.__name__} \'{self.name}\' has '
                        f'function accepting **kwargs '
                        f'arguments but parent {pos+1} '
                        f'({self.parents[pos].name}) has no sub_name')
                kwargs[name] = val
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
set of values, one per node).

This combinatorial generation is produced using gen_graph_iterate

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

def get_ancestors(*nodes):
    found = set()
    def dfs(n):
        if n.name in found:
            return
        found.add(n)
        for pa in n.parents:
            dfs(pa)
    for node in nodes:
        dfs(node)
    return found

def all_values(*nodes):
    """
    Collects the slice of value combinations for {nodes} induced by the
    subgraph of {nodes} and all of their ancestors
    """
    ancestors = get_ancestors(*nodes)
    topo_nodes = _topo_sort(ancestors)
    config = gen_graph_iterate(topo_nodes)
    results = [ tuple(c[n.name] for n in nodes) for c in config ]
    return results

def gen_graph_iterate(nodes):
    """
    Produce all possible settings of the graph nodes as a generator of map
    items.  Each map item is node.name => val
    """
    # print('gen_graph_iterate: ', ','.join(n.name for n in visited_nodes))
    topo_nodes = _topo_sort(nodes)
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

def pred_graph_evaluate(nodes):
    """Evaluate PredNodes in dependency order until a predicate fails"""
    topo_nodes = _topo_sort(nodes)
    for n in topo_nodes:
        if not n.evaluate():
            return n.get_cached_value()
    return None

def func_graph_evaluate(nodes, use_subname=False):
    topo_nodes = _topo_sort(nodes)
    for node in topo_nodes:
        val = node.value()
        node.set_cached_value(val)
    if use_subname:
        return { n.sub_name: n.get_cached_value() for n in topo_nodes }
    else:
        return { n.name: n.get_cached_value() for n in topo_nodes }

