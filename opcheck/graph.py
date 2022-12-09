import sys
import os
from . import fgraph
from . import register, get

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('\nUsage: python graph.py <out_dir> <op>')
        print('\nAvailable Checked Operations:\n')
        ops = available_ops()
        print('\n'.join(ops))
        print()
    else:
        out_dir = sys.argv[1]
        if not os.path.exists(out_dir):
            raise RuntimeError(
                f'Output directory \'{out_dir}\' does not exist')
        op_path = sys.argv[2]
        register(op_path)
        op = get(op_path)
        op.print_graphs(out_dir)

