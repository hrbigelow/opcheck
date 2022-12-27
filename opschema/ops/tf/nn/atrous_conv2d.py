from opschema import genlib
from opschema.complib import conv, conv_t, dilate, dilate_t

def init_schema(op):
    op.add_index('b', 'batch', (1,10))
    op.add_index('i', 'input spatial', 2)
    op.add_index('k', 'input channel', 1)
    op.add_index('f', 'filter spatial', 'i')
    op.add_index('g', 'dilated filter spatial', 'i')
    op.add_index('l', 'output channel', 1)
    op.add_index('o', 'output spatial', 'i')
    op.add_index('r', 'rate', 1) 

    op.arg_tensor('value', 'bik')
    op.arg_tensor('filters', 'fkl')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_shape_int('rate', 'r')
    op.arg_unchecked('name')

    op.gen_dims('b', 1, 50, 50, True)
    op.gen_dims('k', 1, 30, 30, True)
    op.gen_dims('f', 1, 100, 100, True)
    op.gen_dims_func('i', genlib.below_above, 'f', 300, False)  
    op.gen_dims('l', 1, 30, 30, True)
    op.gen_dims('r', 1, 30, 30, True)

    op.valid_dtypes('value', ('int32', 'float',))
    op.equate_dtypes('filters', 'value')

    op.comp_dims_cw('g', dilate, dilate_t, 'fr')
    op.comp_dims_cw('o', conv, conv_t, 'ig', 'padding')

    op.dims_pred_rng('f', 1, None)
    op.dims_pred_rng('k', 1, None)
    op.dims_pred_rng('r', 1, None)
    op.dims_pred_rng('l', 1, None)

    op.return_tensor('bol')


