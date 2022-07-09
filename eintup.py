import sys
from runtime import Runtime

def main(et_file, reps=10, min_dim=None, max_dim=None):
    if min_dim is None and max_dim is None:
        rt = Runtime()
    else:
        rt = Runtime(min_dim, max_dim)
    rt.parse_et_file(et_file)
    rt.validate_all(reps)

if __name__ == '__main__':
    et_file = sys.argv[1]
    reps = 1
    min_dim = 1
    max_dim = 100
    if len(sys.argv) > 2:
        reps = int(sys.argv[2])
    if len(sys.argv) > 3:
        min_dim = int(sys.argv[3])
    if len(sys.argv) > 4:
        max_dim = int(sys.argv[4])

    rt = Runtime(reps, min_dim, max_dim)
    rt.parse_et_file(et_file)
    rt.validate_all()

