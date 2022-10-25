import sys
import os
import opgrind
from schema import fgraph

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('\nUsage: python graph.py <out_dir> <op>')
        print('\nAvailable Checked Operations:\n')
        ops = opgrind.available_ops()
        print('\n'.join(ops))
        print()
    else:
        out_dir = sys.argv[1]
        if not os.path.exists(out_dir):
            raise RuntimeError(
                f'Output directory \'{out_dir}\' does not exist')
        op_path = sys.argv[2]
        opgrind.register(op_path)
        opgrind.print_gen_graph(op_path, out_dir)
        opgrind.print_pred_graph(op_path, out_dir)
        opgrind.print_comp_dims_graph(op_path, out_dir)



