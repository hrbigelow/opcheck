import sys
import os
import opcheck
from schema import fgraph

if __name__ == '__main__':
    opcheck.init()
    if len(sys.argv) < 3:
        print('\nUsage: python graph.py <out_dir> <op>')
        print('\nAvailable Checked Operations:\n')
        opcheck.list_ops()
        print()
    else:
        out_dir = sys.argv[1]
        if not os.path.exists(out_dir):
            raise RuntimeError(
                f'Output directory \'{out_dir}\' does not exist')
        op_path = sys.argv[2]
        opcheck.print_gen_graph(op_path, out_dir)
        opcheck.print_pred_graph(op_path, out_dir)
        opcheck.print_inventory_graph(op_path, out_dir)


