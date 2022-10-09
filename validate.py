import profile 
import opgrind
import sys
import tensorflow as tf
import random
import numpy as np

if __name__ == '__main__':
    random.seed(192384938948348)
    np.random.seed(982348)
    out_dir = sys.argv[1]
    op_path = sys.argv[2]
    opgrind.register(op_path)
    if len(sys.argv) > 3:
        test_ids = [int(id) for id in sys.argv[3].split(',')]
    else:
        test_ids = None
    opgrind.validate(op_path, out_dir, test_ids)
    # profile.run('opgrind.validate(op_path, out_dir, test_ids)')

