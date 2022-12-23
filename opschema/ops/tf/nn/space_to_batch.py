import numpy as np
from opschema.predlib import divis_by, divis_by_t
from opschema import genlib

def init_schema(op):
    op.add_index('b', 'batch', 1)
    op.add_index('i', 'input spatial', (1,3))
    op.add_index('j', 'padded input spatial', 'i')
    op.add_index('k', 'block shape', 'i')
    op.add_index('r', 'remaining', (0,8))
    op.add_index('s', 'padding start', 'i')
    op.add_index('e', 'padding end', 'i')
    op.add_index('o', 'output spatial', 'i')
    op.add_index('p', 'output batch', 1)

    op.arg_tensor('input', 'bir')
    op.arg_shape_tensor('block_shape', 1, int(1e10), 'k')
    op.arg_shape_tensor2d('paddings', 's', 'e')
    op.arg_unchecked('name')

    op.gen_dims('b', 1, 50, 50, True)
    op.gen_dims('i', 1, 50, 50, True)
    op.gen_dims('k', 1, 50, 50, False)
    op.gen_dims('r', 1, 100, 100, True)

    mod_padding100 = genlib.WrapParams(genlib.mod_padding, 50)
    op.gen_dims_func('se', mod_padding100, 'ik', 1e10, False)

    # ensure that padded input is divisible by block size
    op.dims_pred_cw('pad_input_block', divis_by, divis_by_t, 'jk')
    op.valid_dtypes('input', ('int', 'uint', 'float', 'bfloat'))

    def jdims(s, e, i):
        return s + e + i

    def jdims_t(s, e, i):
        return f'{s} + {i} + {e}'

    def odims(padded, block_shape):
        return padded // block_shape

    def odims_t(padded, block_shape):
        return f'{padded} // {block_shape}'

    def pdims(k, b):
        elems = int(np.prod(k))
        return [elems * b]

    def pdims_t(k, b):
        return f'product({k}) * {b}'

    op.comp_dims_cw('j', jdims, jdims_t, 'sei')
    op.comp_dims_cw('o', odims, odims_t, 'jk')
    op.comp_dims('p', pdims, pdims_t, 'kb')

    op.return_tensor('por')

