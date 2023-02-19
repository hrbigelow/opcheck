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

