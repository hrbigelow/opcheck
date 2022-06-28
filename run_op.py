import sys
import json
import tensorflow as tf
import numpy as np
from parse import BCParser
from config import Config
from ast_nodes import RangeConstraint, LogicalOp

def equal_tensors(a, b, eps):
    if not a.dtype.is_floating:
        eps = 0
    return (
            a.shape == b.shape and
            tf.reduce_all(tf.less_equal(tf.abs(a - b), eps)).numpy()
            )

def validate(cfg, json_entry):
    dat = json_entry['tfcall']
    parser = BCParser(cfg)
    parser.set_argument_mode()

    if isinstance(dat['args'], list):
        args = [ parser.parse(arg).value() for arg in dat['args'] ]
        kwargs = { }
    elif isinstance(dat['args'], dict):
        args = [ ]
        kwargs = { k: parser.parse(v).value() for k, v in dat['args'].items() }
    else:
        dat_args_type = type(dat['args'])
        raise RuntimeError(
            f'expected list or object JSON for tfcall "args" field. '
            f'Got {dat_args_type}')

    if 'const-args' in dat:
        if isinstance(dat['const-args'], list):
            args.extend(dat['const-args'])
        elif isinstance(dat['const-args'], dict): 
            kwargs.update(dat['const-args'])
        else:
            const_args_type = type(dat['const-args'])
            raise RuntimeError(
                f'expected list or object JSON for tfcall "const-args" field. '
                f'Got {const_args_type}')

    func = eval(dat['func'])
    tf_results = func(*args, **kwargs)

    return_tensors = dat['return-value']
    if isinstance(return_tensors, str):
        return_tensors = [return_tensors]
    if not isinstance(tf_results, (list, tuple)):
        tf_results = [tf_results]

    et_results = [ parser.parse(name).value() for name in return_tensors ]

    if len(tf_results) != len(et_results):
        raise RuntimeError(
            f'Got {len(tf_results)} results from TF native call '
            f'but {len(et_results)} from eintup program')

    z = zip(tf_results, et_results)
    equal = [equal_tensors(tf_res, et_res, 1e-6) for tf_res, et_res in z]
    return equal

    # print(cfg.array_sig[return_tensor])
    # print(tf.reduce_sum(tf.subtract(tf_result, eintup_result)))
    # print(tf_result.device)


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

    # parsing finished.  finalize
    for node in statements + tests:
        node.post_parse_init()

    range_defs = [ t for t in tests if isinstance(t, RangeConstraint) ]
    rank_tests = [ t for t in tests if isinstance(t, LogicalOp) ]

    primary_tup_names = [ tup.name for tup in cfg.get_primary_tups() ]

    # The total space of all rank combinations being considered.  Hack
    rank_space = (3,) * len(primary_tup_names)
    for rank_combo in np.ndindex(rank_space):
        z = zip(primary_tup_names, rank_combo)
        rank_map = dict(((n, r+1) for n, r in z))
        # rank_map = dict(zip(primary_tup_names, rank_combo))
        cfg.set_dims(rank_map)
        # cfg.set_one_dim('coord', 0, cfg.tup('elem').rank())
        if not all(c.value() for c in rank_tests):
            # print(f'skipping {rank_map}')
            continue

        for ast in statements:
            ast.evaluate()
        valid = validate(cfg, json_entry)
        print(f'{rank_map}: {valid}')

if __name__ == '__main__':
    program_file = sys.argv[1]
    min_dim = int(sys.argv[2])
    max_dim = int(sys.argv[3])

    with open(program_file, 'r') as fp:
        program = json.load(fp)

    cfg = Config(min_dim, max_dim)
    run_programs(cfg, program)

