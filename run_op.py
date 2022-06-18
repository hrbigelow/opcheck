import sys
import json
import tensorflow as tf
import numpy as np
from ein_parser import EinParser
from arg_parser import ArgParser 
from ein_array import ShapeConfig
from numpy.random import randint

def equal_tensors(a, b):
    return a.shape == b.shape and tf.reduce_all(tf.math.equal(a, b))


def validate(cfg, json_entry):
    dat = json_entry['tfcall']
    argparser = ArgParser(cfg)
    kwargs = { k: argparser.parse(v) for k, v in dat['args'].items() }
    func = eval(dat['func'])
    eintup_result = argparser.parse(dat['return-value'])
    tf_result = func(**kwargs)
    equal = equal_tensors(tf_result, eintup_result)
    return equal


def run_programs(cfg, json_entry):
    cons = json_entry['rank-constraints']
    argparser = ArgParser(cfg)
    einparser = EinParser(cfg)
    constraints = [ argparser.parse(c) for c in cons ]
    indices = set.union(*(a.get_indices() for a in constraints))

    statements = [ einparser.parse(st) for st in program['program'] ]

    for r in np.ndindex((10,) * len(indices)):
        rank_map = dict(zip(indices, r))
        cfg.set_ranks(rank_map)
        if not all(c.value() for c in constraints):
            print(f'skipping {rank_map}')
            continue

        print(f'processing {rank_map}')
        cfg.init_arrays()

        for ast in statements:
            cfg.prepare_for_statement(ast)
            while cfg.tup.advance():
                ast.evaluate()
        valid = validate(cfg, program)
        print(f'{rank_map}: {valid}')



if __name__ == '__main__':
    program_file = sys.argv[1]
    op = sys.argv[2]

    with open(program_file, 'r') as fp:
        programs = json.load(fp)

    program = programs[op]
    cfg = ShapeConfig()
    run_programs(cfg, program)






