import opcheck
from . import available_ops, _init_op
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('\nUsage: python explain.py <op> [-i]')
        print('\nAvailable Checked Operations:\n')
        ops = available_ops()
        print('\n'.join(ops))
        print()
    else:
        op_path = sys.argv[1]
        include_inventory = len(sys.argv) == 3 and sys.argv[2] == '-i'
        op = _init_op(op_path)
        print(op.schema_report(include_inventory))

