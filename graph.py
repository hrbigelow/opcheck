import sys
import os
import opcheck
from schema import fgraph

if __name__ == '__main__':
    opcheck.init()
    if len(sys.argv) < 3:
        print('\nUsage: python graph.py <out_dir> <op>')
        print('\nAvailable Checked Operations:\n')
        opcheck.inventory()
        print()
    else:
        out_dir = sys.argv[1]
        if not os.path.exists(out_dir):
            raise RuntimeError(
                f'Output directory \'{out_dir}\' does not exist')
        op_path = sys.argv[2]
        opcheck.gen_graph_viz(op_path, out_dir)
        opcheck.pred_graph_viz(op_path, out_dir)
        opcheck.inv_graph_viz(op_path, out_dir)


