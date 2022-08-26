import opcheck
import sys
import tensorflow as tf

if __name__ == '__main__':
    opcheck.init()
    op_path = sys.argv[1]
    opcheck.validate(op_path)

