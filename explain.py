import opcheck
import sys

if __name__ == '__main__':
    opcheck.init()
    if len(sys.argv) < 2:
        print('\nUsage: python explain.py <op>')
        print('\nAvailable Checked Operations:\n')
        opcheck.inventory()
        print()
    else:
        op_path = sys.argv[1]
        opcheck.explain(op_path)

