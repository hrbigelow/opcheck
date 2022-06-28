import sys
import json
import tensorflow as tf
import numpy as np
from config import Config
import misc

if __name__ == '__main__':
    program_file = sys.argv[1]
    min_dim = int(sys.argv[2])
    max_dim = int(sys.argv[3])

    with open(program_file, 'r') as fp:
        inst = json.load(fp)
    program_text = inst['program']
    constraint_text = inst['constraints']

    cfg = Config(min_dim, max_dim)
    cfg.compile(program_text, constraint_text)

    # The total space of all rank combinations being considered.  Hack
    primary_tup_names = [ tup.name for tup in cfg.get_primary_tups() ]
    rank_space = (10,) * len(primary_tup_names)
    for rank_combo in np.ndindex(rank_space):
        rank_map = dict(zip(primary_tup_names, rank_combo))
        cfg.set_ranks(rank_map)
        outs = cfg.run()
        valid = misc.validate(cfg, json_entry)
        print(f'{rank_map}: {valid}')

