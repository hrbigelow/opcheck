import profile 
import opcheck
import sys
import tensorflow as tf
import random
import numpy as np

if __name__ == '__main__':
    random.seed(192384938948348)
    np.random.seed(982348)
    opcheck.init()
    out_dir = sys.argv[1]
    op_path = sys.argv[2]
    if len(sys.argv) > 3:
        test_ids = [int(id) for id in sys.argv[3].split(',')]
    else:
        test_ids = None
    opcheck.validate(op_path, out_dir, test_ids)
    # profile.run('opcheck.validate(op_path, out_dir, test_ids)')


