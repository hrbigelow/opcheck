import sys
from runtime import Runtime

if __name__ == '__main__':
    et_file = sys.argv[1]
    min_dim = int(sys.argv[2])
    max_dim = int(sys.argv[3])

    rt = Runtime(min_dim, max_dim)
    rt.parse_et_file(et_file)
    rt.validate_all()

