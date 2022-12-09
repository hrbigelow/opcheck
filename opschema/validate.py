import fire
import opschema
import random
import numpy as np

def main(out_dir, op_path, test_ids=None, skip_ids=None, dtype_err_quota=2):
    random.seed(192384938948348)
    np.random.seed(982348)
    opschema.register(op_path)
    if isinstance(test_ids, int):
        test_ids = {test_ids}
    elif isinstance(test_ids, tuple):
        test_ids = set(test_ids)

    opschema.validate(op_path, out_dir, test_ids, skip_ids, dtype_err_quota)

if __name__ == '__main__':
    fire.Fire(main)

