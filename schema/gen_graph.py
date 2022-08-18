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
import itertools

class GenNode(object):
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.parents = []
        self.children = []
        self.cur_val = None

    def __repr__(self):
        return (f'{type(self).__name__}({self.name})'
                f'[pa: ({",".join(p.name for p in self.parents)})]')

    def add_child(self, node):
        self.children.append(node)
        node.parents.append(self)

    def add_parent(self, node):
        self.parents.append(node)
        node.children.append(self)

    def values(self):
        kwargs = { p.name: p.get_current_value() for p in self.parents }
        vals = self.func(**kwargs)
        try:
            iter(vals)
        except TypeError:
            raise RuntimeError(f'{self}: function does not return an iterable') 
        return vals

    def get_current_value(self):
        return self.cur_val

    def set_current_value(self, val):
        self.cur_val = val

def get_roots(nodes):
    """Find the subset of nodes having no parents"""
    return [ n for n in nodes if len(n.parents) == 0 ]

def iterate(nodes, val_map={}):
    """Produce all possible settings of the graph"""
    if len(nodes) == 0:
        yield dict(val_map)
        return

    all_ch = { ch for n in nodes for ch in n.children }
    next_nodes = [ n for n in all_ch if all(p in nodes for p in n.parents) ]
    node_value_lists = tuple(n.values() for n in nodes)
    for vals in itertools.product(*node_value_lists):
        for node, val in zip(nodes, vals):
            node.set_current_value(val)
            val_map[node.name] = val
        yield from iterate(next_nodes, val_map)

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

