import numpy as np
from opcheck.predlib import divis_by, divis_by_t
from opcheck.genlib import mod_padding

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
    op.arg_shape_tensor('block_shape', 'k')
    op.arg_shape_tensor2d('paddings', 's', 'e')
    op.arg_unchecked('name')

    op.gen_dims('b', 50)
    op.gen_dims('i', 400)
    op.gen_dims('k', 50)
    op.gen_dims('r', 100)

    op.gen_dims_func('se', mod_padding, 'ik', 1e10, False, 100)

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

    def pdims(block_shape, batch):
        elems = np.prod(block_shape)
        return [elems * b for b in batch]

    def pdims_t(block_shape, batch):
        return f'product({block_shape}) * {batch}'

    op.comp_dims_cw('j', jdims, jdims_t, 'sei')
    op.comp_dims_cw('o', odims, odims_t, 'jk')
    op.comp_dims('p', pdims, pdims_t, 'kb')

    op.return_tensor('por')

