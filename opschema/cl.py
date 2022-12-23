import fire
import pickle
import sys
import os
import opschema
import random
import numpy as np
import signal
from multiprocessing import Process


def list_schemas():
    for op_path in opschema.list_schemas():
        print(op_path)

def gen_input(op_path, out_dir):
    op = opschema.init_op(op_path)
    inputs = list(op.generate_args())
    file_name = os.path.join(out_dir, f'{op_path}.inputs.pkl')
    with open(file_name, 'wb') as fh:
        pickle.dump(inputs, fh) 

def test_op(op_path, out_dir, test_id):
    # file_name = os.path.join(out_dir, f'{op_path}.inputs.pkl')
    # with open(file_name, 'rb') as fh:
        # inputs = pickle.load(fh)
    opschema.register(op_path)
    op = opschema.get(op_path)
    
    gen = op.generate_args()
    for test_num, op_args in enumerate(gen, 1):
        if test_num == test_id:
            args = { k: v.value() for k, v in op_args.items() }
            try:
                print('OpSchema message:')
                op.wrapped_op(**args)
            except:
                print('TensorFlow traceback:')
                sys.excepthook(*sys.exc_info())
            break

def validate(op_path, out_dir, test_ids=None, skip_ids=None, max_dtype_err=0,
        test_edits=0, rand_seed=0, show_traceback=False):
    opschema.register(op_path)
    op = opschema.get(op_path)

    if isinstance(test_ids, int):
        test_ids = {test_ids}
    elif isinstance(test_ids, tuple):
        test_ids = set(test_ids)

    if isinstance(skip_ids, int):
        skip_ids = {skip_ids}
    elif isinstance(skip_ids, tuple):
        skip_ids = set(skip_ids)

    return op.validate(out_dir, test_ids, skip_ids, max_dtype_err, test_edits,
            rand_seed, show_traceback)

def explain(op_path, include_inventory=False):
    return opschema.explain(op_path, include_inventory)

def graph(op_path, out_dir):
    opschema.print_graphs(op_path, out_dir)

def main():
    func_map = { 
            'list': list_schemas,
            'explain': explain,
            # 'gen_input': gen_input,
            # 'test_op': test_op,
            'validate': validate,
            'graph': graph
            }
    fire.Fire(func_map)

if __name__ == '__main__':
    main()

