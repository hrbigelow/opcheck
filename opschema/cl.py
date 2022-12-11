import fire
import pickle
import sys
import os
import opschema

def list_schemas():
    print('Available Schemas in ops directory')
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

def validate(op_path, out_dir, test_ids=None):
    opschema.register(op_path)
    op = opschema.get(op_path)

    if test_ids is None:
        pass
    elif isinstance(test_ids, int):
        test_ids = {test_ids}
    else:
        test_ids = set(test_ids)
    op.validate(out_dir, test_ids)

def explain(op_path, include_inventory=False):
    return opschema.explain(op_path, include_inventory)

def main():
    cmd = sys.argv.pop(1)
    func_map = { 
            'list': list_schemas,
            'explain': explain,
            'gen_input': gen_input,
            'test_op': test_op,
            'validate': validate
            }
    func = func_map.get(cmd, None)
    if func is None:
        avail = ', '.join(func_map.keys())
        print(f'Couldn\'t understand subcommand \'{cmd}\'.  Use one of: {avail}')
        return 1
    fire.Fire(func)

if __name__ == '__main__':
    main()

