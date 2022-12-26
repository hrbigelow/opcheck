from opschema import genlib
from opschema.complib import dilate, dilate_t, tconv, tconv_t

def init_schema(op):
    op.add_index('b', 'batch', (1, 10))
    op.add_index('i', 'input spatial', 2)
    op.add_index('k', 'input channel', 1)
    op.add_index('f', 'filter spatial', 'i')
    op.add_index('g', 'dilated filter spatial', 'i')
    op.add_index('l', 'output channel', 1)
    op.add_index('o', 'output spatial declared', 'i')
    op.add_index('q', 'output spatial', 'i')
    op.add_index('r', 'rate', 1) 

    op.arg_tensor('value', 'bik')
    op.arg_tensor('filters', 'flk')
    op.arg_shape_tensor('output_shape', 1, None, 'bol')
    op.arg_shape_int('rate', 'r')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_unchecked('name')
    op.return_tensor('bql')

    op.gen_dims('b', 1, 100, 100, True)
    op.gen_dims('f', 1, 100, 100, True)
    op.gen_dims_func('i', genlib.below_above, 'f', 1000, False)  
    op.gen_dims('k', 1, 30, 30, True)
    op.gen_dims('l', 1, 30, 30, True)
    op.gen_dims('r', 1, 30, 30, True)

    op.comp_dims_cw('g', dilate, dilate_t, 'fr') 

    def odims_gen(i, g, padding):
        if padding == 'VALID':
            v = i + g - 1
            yield v, v
        else:
            yield i, i

    op.gen_dims_calc('o', odims_gen, 'ig', 'padding')
    op.comp_dims_cw('q', tconv, tconv_t, 'ig', 'padding')

    op.valid_dtypes('value', ('float',))
    op.equate_dtypes('filters', 'value')

