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

if __name__ == '__main__':
    def gen_first():
        yield from 'ABCD'

    def gen_second(letter):
        rule = dict(zip('ABCD', [1,2,3,4]))
        # rule = { 'A': (1,2), 'B': (2,3), 'C': (3,4), 'D': (4,5) }
        yield rule[letter]

    def gen_third(number):
        rule = dict(zip([1,2,3,4], 'WXYZ'))
        # rule = { 1: 'WX', 2: 'XY', 3: 'YZ', 4: 'Z', 5: 'ZW' }
        yield rule[number]

    node1 = GeneratorNode('first', gen_first)
    node2 = GeneratorNode('second', gen_second, node1)
    node3 = GeneratorNode('third', gen_third, node2)
    nodes = [node1, node2, node3]

    print('Generating all combinations')
    for vals in generate_all(nodes):
        print(vals)

    
    observed = { 'first': 'C', 'second': 2, 'third': 'Y' }
    print(f'\nGenerating combinations within 1 edit from {observed}') 
    for vals in generate_nearest(nodes, observed, 1):
        print(vals)

    print(f'\nGenerating combinations within 2 edits from {observed}') 
    for vals in generate_nearest(nodes, observed, 2):
        print(vals)

