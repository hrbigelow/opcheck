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

    op.gen_dims('b', 50)
    op.gen_dims('k', 30)
    op.gen_dims('f', 100)
    op.gen_dims_func('i', genlib.below_above, 'f', 300, False)  
    op.gen_dims('l', 30)
    op.gen_dims('r', 30)

    op.valid_dtypes('value', ('int32', 'float',))
    op.equate_dtypes('filters', 'value')

    op.comp_dims_cw('g', dilate, dilate_t, 'fr')
    op.comp_dims_cw('o', conv, conv_t, 'ig', 'padding')
    op.return_tensor('bol')


