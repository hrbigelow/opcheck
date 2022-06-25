import sys
import json
import tensorflow as tf
import numpy as np
from bcast_parse import BCParser
from config import Config
from bcast_ast import RangeConstraint, LogicalOp

def equal_tensors(a, b, eps):
    return (
            a.shape == b.shape and
            tf.reduce_all(tf.less_equal(tf.abs(a - b), eps))
            )

def validate(cfg, json_entry):
    dat = json_entry['tfcall']
    parser = BCParser(arg)
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
    constraints = json_entry['constraints']
    parser = BCParser(cfg)
    parser.set_constraint_mode()
    tests = [ parser.parse(c) for c in constraints ]
    range_defs = [ t for t in tests if isinstance(t, RangeConstraint) ]
    rank_tests = [ t for t in tests if t.is_static() and isinstance(t,
        LogicalOp) ]

    program = json_entry['program']
    parser.set_statement_mode()
    statements = [ parser.parse(st) for st in program ]

    for st in statements:
        st.add_range_constraints(range_defs)

    indices = {ind for s in statements for ind in s.live_indices()}
    indices = list(sorted(indices))

    eintup_names = ','.join(indices)
    print(f'{eintup_names} ranks\tValid?')

    for r in np.ndindex((10,) * len(indices)):
        rank_map = dict(zip(indices, r))
        cfg.set_ranks(rank_map)
        for rt in rank_tests:
            rt.prepare()
        if not all(c.value() for c in rank_tests):
            # print(f'skipping {rank_map}')
            continue

        # print(f'processing {rank_map}, dims: {cfg.tup.dims_map}')

        for ast in statements:
            ast.evaluate()
            # cfg.print_array('indices')
        valid = validate(cfg, json_entry)
        rank_string = ','.join(str(rank_map[k]) for k in indices)
        print(f'{rank_string}\t{valid}')
        # print(f'{rank_map}: {valid}')



if __name__ == '__main__':
    program_file = sys.argv[1]

    with open(program_file, 'r') as fp:
        program = json.load(fp)

    cfg = Config()
    run_programs(cfg, program)

