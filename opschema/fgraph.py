import inspect
import enum
from .error import SchemaError


"""
Usage:

node_reg = {}
F.set_registry(node_reg)
name = 'a'
nf = NodeFunc(name)
node = F.add_node(nf)

name2 = 'b
nf2 = NodeFunc(name2)

# add node2, this time
node2 = F.add_node_sn(nf2, node)

print(node_reg)
# { 'NodeFunc(a)': FuncNode(a), 'b': FuncNode(b)[pa: NodeFunc(a)] }

"""

def node_name(func_node_class, name=None):
    if name is None:
        return func_node_class.__name__
    else:
        return f'{func_node_class.__name__}({name})'

class NodeFunc(object):
    def __init__(self, name=None):
        self.sub_name = name

    def wrapped_name(self, name):
        return f'{self.__class__.__name__}({name})'

    @property
    def name(self):
        return node_name(self.__class__, self.sub_name)

    @property
    def graphviz_name(self):
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

    def __init__(self, func, use_subname, num_named_pars, vararg_type):
        """
        num_named_pars is the number of named parameters that func takes. (any
        arguments that are not *args or **kwargs).  vararg_type is the type of
        variable arg it has (*args, **kwargs, or neither)

        """
        self.name = func.name 
        self.sub_name = func.sub_name
        self.use_subname = use_subname
        self.func = func
        self.parents = []
        self.use_parent_subname = []
        self.children = []
        self.cached_val = None
        self.num_named_pars = num_named_pars
        self.vararg_type = vararg_type 

    def __repr__(self):
        return (f'{type(self).__name__}({self.used_name()})'
                f'[pa: {",".join(p.used_name() for p in self.parents)}]')

    def used_name(self):
        return self.sub_name if self.use_subname else self.name

    @property
    def graphviz_name(self):
        return self.func.graphviz_name

    def clone_node_only(self):
        return type(self)(self.func, self.use_subname, self.num_named_pars,
                self.vararg_type)

    @classmethod
    def _add_node(cls, func, pass_subname, *parents):
        """
        Creates a new node enclosing {func} as its function.  The name of the
        node is defined by {func}.name.  {func} must take len(parents)
        arguments, which will be provided by parent nodes of those names.
        {func} may have *args or **kwargs in its signature, but not both.  The
        outputs of the parents in order are passed to the positional arguments
        of {func}.  Any remaining parents are either passed to *args as a list
        of values, or passed to **kwargs as a dictionary, using the parent node
        names as the keys of the dictionary.

        if {pass_subname} is True, use func.sub_name as the registry retrieval
        key and argument name for children.  Otherwise, use func.name.
        """
        if cls.registry is None:
            raise RuntimeError(
                f'{type(cls).__qualname__}: registry is not set.  Call '
                f'set_registry(reg) with a map object first')

        used_name = func.sub_name if pass_subname else func.name

        if used_name in cls.registry:
            raise SchemaError(
                f'{type(cls).__qualname__}: node name \'{used_name}\' already '
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
                    f'{cls.__qualname__}: function takes {len(named_pars)} '
                    f'arguments, but {len(parents)} parents provided ')
        else:
            if len(parents) < len(named_pars):
                raise SchemaError(
                    f'{cls.__qualname__}: function takes {len(named_pars)} '
                    f'positional arguments but only {len(parents)} parents '
                    f'provided.')
        num_named_pars = len(named_pars)
        if wildcard is None:
            vararg_type = VarArgs.Empty
        elif wildcard == args_par:
            vararg_type = VarArgs.Positional
        else:
            vararg_type = VarArgs.Keyword

        node = cls(func, pass_subname, num_named_pars, vararg_type)
        for pa in parents:
            node._append_parent(pa, pa.use_subname)
        cls.registry[node.used_name()] = node
        return node 

    @classmethod
    def add_node(cls, func, *parents):
        """
        Add a node, using func.name as retrieval key and passed argument name 
        """
        return cls._add_node(func, False, *parents)

    @classmethod
    def add_node_sn(cls, func, *parents):
        """
        Add a node, using func.sub_name as retrieval key and passed argument
        name
        """
        return cls._add_node(func, True, *parents)

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

    def _add_child(self, node, pass_subname):
        self.children.append(node)
        node.parents.append(self)
        node.use_parent_subname.append(pass_subname)

    def add_child(self, node):
        self._add_child(node, False)

    def add_child_sn(self, node):
        self._add_child(node, True)

    def _append_parent(self, node, pass_subname):
        self.parents.append(node)
        self.use_parent_subname.append(pass_subname)
        node.children.append(self)

    def append_parent(self, node):
        """
        Append {node} as a parent of this node.
        Pass node.name as the argument name to this node
        """
        self._append_parent(node, False)

    def append_parent_sn(self, node):
        """
        Append {node} as a parent of this node.
        Pass node.sub_name as the argument name to this node.
        """
        self._append_parent(node, True)

    def _maybe_append_parent(self, node, pass_subname):
        pa = next((n for n in self.parents if n.name == node.name), None)
        if pa is not None:
            return
        self._append_parent(node, pass_subname)

    def maybe_append_parent(self, node):
        """
        Append {node} as a parent of this node if not already a parent.
        Pass node.name as the argument to this node.
        """
        self._maybe_append_parent(node, False)

    def maybe_append_parent_sn(self, node):
        """
        Append {node} as a parent of this node if not already a parent
        Pass node.sub_name as the argument name to this node
        """
        self._maybe_append_parent(node, True)

    def all_children(self):
        return self.children

    def value(self):
        """
        Evaluate the current node based on cached values of the parents
        """
        z = zip(self.parents, self.use_parent_subname)
        all_args = [(n.sub_name if s else n.name, n.get_cached()) for n,s in z]
        pos_args = [v for n,v in all_args[:self.num_named_pars]]
        if self.vararg_type == VarArgs.Positional:
            args = tuple(v for n,v in all_args[self.num_named_pars:])
            return self.func(*pos_args, *args)
        elif self.vararg_type == VarArgs.Keyword:
            kwargs = {}
            for pos in range(self.num_named_pars, len(all_args)):
                name, val = all_args[pos]
                if name is None:
                    pa = self.parents[pos]
                    raise SchemaError(
                        f'{self.__class__.__name__} \'{self.name}\' has '
                        f'arguments but parent {pos+1} '
                        f'({pa.name}) has no usable name')
                kwargs[name] = val
            return self.func(*pos_args, **kwargs)
        else:
            return self.func(*pos_args)

    def get_cached(self):
        """Retrieve the cached function evaluation value"""
        return self.cached_val

    def set_cached(self, val):
        """Set the cached function evaluation value"""
        self.cached_val = val

class VarArgs(enum.Enum):
    Positional = 0 # *args
    Keyword = 1    # **kwargs
    Empty = 2         # neither

def _topo_sort(nodes):
    """
    Sort nodes with ancestors first
    """
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
    for n in sorted(nodes, key=lambda n: n.name):
        dfs(n)
    topo_list = order[::-1]
    return topo_list

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
        yield from vals
        return
        try:
            iter(vals)
        except TypeError:
            raise SchemaError(f'{self}: function does not return an iterable') 
        yield from vals
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
        self.set_cached(value)
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
    config = gen_graph_iterate(ancestors)
    results = [ tuple(c[n.name] for n in nodes) for c in config ]
    return results

def gen_graph_iterate(nodes, full_name=True):
    """
    Produce all possible settings of the graph nodes as a generator of map
    items.  Each map item is name => val.  If `full_name`, use node.name as the
    name, otherwise use node.sub_name
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
            node.set_cached(val)
            name = node.name if full_name else node.sub_name
            val_map[name] = val
            yield from gen_rec(i+1)
    yield from gen_rec(0)

def _gen_graph(live_nodes, result_nodes, yield_map, full_name, op):
    """
    Iterate over all settings of live_nodes.  For each setting, collect the
    current values of result_nodes (which must be a subset of live_nodes)
    and yield as a tuple
    """
    # map from li => ri
    for rn in result_nodes:
        if rn not in live_nodes:
            raise RuntimeError(
                f'All nodes in result_nodes must be in live_nodes. Got '
                f'result node \'{rn.name}\'.  Available live_nodes are: '
                f'{", ".join(l.name for l in live_nodes)}')

    live_nodes = _topo_sort(live_nodes)
    imap = [-1] * len(live_nodes)

    for ri, r in enumerate(result_nodes):
        li = live_nodes.index(r)
        imap[li] = ri

    result = [None] * len(result_nodes)
    res_names = [r.name if full_name else r.sub_name for r in result_nodes]

    def gen_rec(i):
        if i == len(live_nodes):
            if yield_map:
                yield dict(zip(res_names, result))
            else:
                yield tuple(result) 
            return
        node = live_nodes[i]
        values = node.values()

        if op and op.show_graph_calls:
            initial_edits = op.avail_edits 
            if i > 0:
                pre_node = live_nodes[i-1]
                indented_name = ' ' * i + pre_node.name
                msg = (
                        f'initial_edits: {initial_edits} '
                        f'avail_edits: {op.avail_edits} '
                        f'node_val: {pre_node.get_cached()} '
                        )
                print(f'{indented_name:50s}{msg}')

        for val in values:
            node.set_cached(val)
            ri = imap[i]
            if ri >= 0:
                result[ri] = val
            yield from gen_rec(i+1)
        else:
            if op and op.show_graph_calls:
                print(' ' * (i+1) + f'{node.name}  (no values)')

    yield from gen_rec(0)

def gen_graph_values(live_nodes, result_nodes, op=None):
    """
    Iterate over all configurations of live_nodes, reporting the tuple of
    values of result_nodes, which must be a subset of live_nodes
    """
    return _gen_graph(live_nodes, result_nodes, False, False, op)

def gen_graph_map(live_nodes, result_nodes, full_name=True, op=None):
    """
    Iterate over all configurations of live_nodes, reporting the map values of
    result_nodes (node name => value), which must be a subset of live_nodes.
    if `full_name`, use node.name.  Otherwise use node.sub_name
    """
    return _gen_graph(live_nodes, result_nodes, True, full_name, op)

def pred_graph_evaluate(*nodes):
    """
    Evaluate PredNodes in dependency order until a predicate fails.
    If any predicate fails, return its value.  Otherwise, return None
    """
    topo_nodes = _topo_sort(nodes)
    for n in topo_nodes:
        if not n.evaluate():
            return n.get_cached()
    return None

