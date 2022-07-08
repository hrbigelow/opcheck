import sys
from runtime import Runtime

def main(et_file, min_dim=None, max_dim=None):
    if min_dim is None and max_dim is None:
        rt = Runtime()
    else:
        rt = Runtime(min_dim, max_dim)
    rt.parse_et_file(et_file)
    rt.validate_all()

if __name__ == '__main__':
    et_file = sys.argv[1]
    min_dim = max_dim = None
    if len(sys.argv) > 2:
        min_dim = int(sys.argv[2])
        max_dim = int(sys.argv[3])

    main(et_file, min_dim, max_dim)
