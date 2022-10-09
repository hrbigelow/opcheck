import opgrind
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('\nUsage: python explain.py <op>')
        print('\nAvailable Checked Operations:\n')
        ops = opgrind.available_ops()
        print('\n'.join(ops))
        print()
    else:
        op_path = sys.argv[1]
        opgrind.register(op_path)
        opgrind.explain(op_path)

