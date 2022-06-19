import sys
import json
import tensorflow as tf
import numpy as np
from ein_parser import EinParser
from arg_parser import ArgParser
from constraint_parser import ConsParser
from ein_array import ShapeConfig
from numpy.random import randint

def equal_tensors(a, b, eps):
    return (
            a.shape == b.shape and
            tf.reduce_all(tf.less_equal(tf.abs(a - b), eps))
            )


def validate(cfg, json_entry):
    dat = json_entry['tfcall']
    argparser = ArgParser(cfg)
    kwargs = { k: argparser.parse(v).value() for k, v in dat['args'].items() }
    eintup_result = argparser.parse(dat['return-value']).value()
    func = eval(dat['func'])
    tf_result = func(**kwargs)
    equal = equal_tensors(tf_result, eintup_result, 1e-6)
    return equal


def run_programs(cfg, json_entry):
    cons = json_entry['rank-constraints']
    consparser = ConsParser(cfg)
    constraints = [ consparser.parse(c) for c in cons ]

    program = json_entry['program']
    einparser = EinParser(cfg)
    statements = [ einparser.parse(st) for st in program ]

    indices = set.union(*(a.get_indices() for a in constraints))

    for r in np.ndindex((10,) * len(indices)):
        rank_map = dict(zip(indices, r))
        cfg.set_ranks(rank_map)
        if not all(c.value() for c in constraints):
            # print(f'skipping {rank_map}')
            continue

        # print(f'processing {rank_map}, dims: {cfg.tup.dims_map}')
        cfg.init_arrays()

        for ast in statements:
            cfg.prepare_for_statement(ast)
            while cfg.tup.advance():
                ast.evaluate()
            # cfg.print_array('indices')
        valid = validate(cfg, json_entry)
        print(f'{rank_map}: {valid}')



if __name__ == '__main__':
    program_file = sys.argv[1]
    op = sys.argv[2]

    with open(program_file, 'r') as fp:
        programs = json.load(fp)

    program = programs[op]
    cfg = ShapeConfig()
    run_programs(cfg, program)






