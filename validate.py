import profile 
import opgrind
import sys
import tensorflow as tf
import random
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print()
        print('Usage: validate.py <out_dir> <op_path> [test_ids] [skip_ids]')
        print()
        print('Produces TP,FP,TN,FN and stats reports in <out_dir> for <op_path>')
        print('Run python explain.py to see valid <op_path> values.')
        print(f'test_ids and skip_ids are (optional) comma-separated lists '
                'of integers to test (or skip, respectively)')
        print()
        sys.exit()

    random.seed(192384938948348)
    np.random.seed(982348)
    out_dir = sys.argv[1]
    op_path = sys.argv[2]
    opgrind.register(op_path)
    args = dict(enumerate(sys.argv))
    test_ids = args.get(3, '').split(',')
    skip_ids = args.get(4, '').split(',')

    test_ids = {int(f) for f in test_ids if f != ''}
    skip_ids = {int(f) for f in skip_ids if f != ''}

    if len(test_ids) == 0:
        test_ids = None

    opgrind.validate(op_path, out_dir, test_ids, skip_ids)
    # profile.run('opgrind.validate(op_path, out_dir, test_ids, skip_ids)')

