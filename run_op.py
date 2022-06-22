import sys
import json
import tensorflow as tf
import numpy as np
from ein_parser import EinParser
from arg_parser import ArgParser
from constraint_parser import ConsParser
from ein_array import ShapeConfig
from ein_ast import RangeConstraint, LogicalOp

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
    constraints = json_entry['constraints']
    consparser = ConsParser(cfg)
    tests = [ consparser.parse(c) for c in constraints ]
    range_defs = [ t for t in tests if isinstance(t, RangeConstraint) ]
    rank_tests = [ t for t in tests if t.is_static() and isinstance(t,
        LogicalOp) ]

    program = json_entry['program']
    einparser = EinParser(cfg)
    statements = [ einparser.parse(st) for st in program ]

    for st in statements:
        st.add_range_constraints(range_defs)

    indices = {ind for s in statements for ind in s.get_indices()}

    for r in np.ndindex((10,) * len(indices)):
        rank_map = dict(zip(indices, r))
        cfg.set_ranks(rank_map)
        if not all(c.value() for c in rank_tests):
            # print(f'skipping {rank_map}')
            continue

        print(f'processing {rank_map}, dims: {cfg.tup.dims_map}')
        cfg.init_arrays()

        for ast in statements:
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






