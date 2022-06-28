import sys
import json
import tensorflow as tf
import numpy as np
from parse import BCParser
from config import Config
from ast_nodes import RangeConstraint, LogicalOp

def equal_tensors(a, b, eps):
    return (
            a.shape == b.shape and
            tf.reduce_all(tf.less_equal(tf.abs(a - b), eps))
            )

def validate(cfg, json_entry):
    dat = json_entry['tfcall']
    parser = BCParser(cfg)
    parser.set_argument_mode()
    kwargs = { k: parser.parse(v).value() for k, v in dat['args'].items() }
    if 'const-args' in dat:
        kwargs.update(dat['const-args'])

    eintup_result = parser.parse(dat['return-value']).value()
    func = eval(dat['func'])
    tf_result = func(**kwargs)
    equal = equal_tensors(tf_result, eintup_result, 1e-6)
    return equal


def run_programs(cfg, json_entry):
    parser = BCParser(cfg)

    # parse statements first.  All EinTups are instantiated here
    program = json_entry['program']
    parser.set_statement_mode()
    statements = [ parser.parse(st) for st in program ]

    # parse constraints.  It is an error if new EinTups are mentioned here
    constraints = json_entry['constraints']
    parser.set_constraint_mode()
    tests = [ parser.parse(c) for c in constraints ]

    range_defs = [ t for t in tests if isinstance(t, RangeConstraint) ]
    rank_tests = [ t for t in tests if isinstance(t, LogicalOp) ]

    primary_tup_names = [ tup.name for tup in cfg.get_primary_tups() ]

    # The total space of all rank combinations being considered.  Hack
    rank_space = (10,) * len(primary_tup_names)
    for rank_combo in np.ndindex(rank_space):
        rank_map = dict(zip(primary_tup_names, rank_combo))
        cfg.set_dims(rank_map)
        if not all(c.value() for c in rank_tests):
            # print(f'skipping {rank_map}')
            continue

        for ast in statements:
            ast.evaluate()
        valid = validate(cfg, json_entry)
        print(f'{rank_map}: {valid}')

if __name__ == '__main__':
    program_file = sys.argv[1]

    with open(program_file, 'r') as fp:
        program = json.load(fp)

    cfg = Config(5, 10)
    run_programs(cfg, program)

