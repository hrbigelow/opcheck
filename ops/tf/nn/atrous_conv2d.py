from opschema import genlib

def init_schema(op):
    op.add_index('b', 'batch', (1,10))
    op.add_index('i', 'input spatial', 2)
    op.add_index('k', 'input channel', 1)
    op.add_index('f', 'filter spatial', 'i')
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

    def odims(i, f, r, padding):
        if padding == 'VALID':
            out = i - (f - 1) * r
        else:
            out = i
        return out

    def odims_t(i, f, r, padding):
        if padding == 'VALID':
            txt = f'{i} - ({f} - 1) * {r}'
        else:
            txt = f'{i}'
        return txt

    op.comp_dims_cw('o', odims, odims_t, 'ifr', 'padding')
    op.return_tensor('bol')


