import sys
import json
import tensorflow as tf
from ein_lexer import EinLexer
from ein_parser import EinParser
from arg_parser import ArgParser 
from numpy.random import randint

def gen_shapes(size_map):
    return {
            ind: [randint(2, 5) for _ in range(sz)] for ind, sz in
            size_map.items()
            }

def equal_tensors(a, b):
    return a.shape == b.shape and tf.reduce_all(tf.math.equal(a, b))

def validate(parser, json_entry):
    dat = json_entry['tfcall']
    argparser = ArgParser(parser.arrays, parser.slice_tuple.shape_map)
    args = [ argparser.parse(st) for st in dat['args'] ]
    func = dat['func']
    eintup_result = argparser.parse(dat['return-value'])
    tf_result = tf.__dict__[func](*args)
    equal = equal_tensors(tf_result, eintup_result)
    return equal



if __name__ == '__main__':
    program_file = sys.argv[1]
    op = sys.argv[2]

    lexer = EinLexer()
    parser = EinParser()
    with open(program_file, 'r') as fp:
        programs = json.load(fp)

    program = programs[op]
    size_map = { 'b': 3, 'e': 4, 's': 2, 'c': 1 }

    shape_map = gen_shapes(size_map)
    shape_map['c'][0] = 4

    progs = [ parser.parse(lexer.tokenize(st)) for st in program['program'] ]
    parser.update_shapes(shape_map)

    for ast in progs:
        parser.init_program(ast)
        while parser.slice_tuple.advance():
            ast.evaluate()
    valid = validate(parser, program)
    print(f'{op}: {valid}')




