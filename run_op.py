import sys
from config import Config

if __name__ == '__main__':
    et_file = sys.argv[1]
    min_dim = int(sys.argv[2])
    max_dim = int(sys.argv[3])

    cfg = Config(min_dim, max_dim)
    cfg.parse_et_file(et_file)
    cfg.validate_all()

