import opcheck
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('\nUsage: python explain.py <op>')
        print('\nAvailable Checked Operations:\n')
        ops = opcheck.available_ops()
        print('\n'.join(ops))
        print()
    else:
        op_path = sys.argv[1]
        opcheck.register(op_path)
        # opcheck.explain(op_path)
        opcheck.schema_report(op_path)

