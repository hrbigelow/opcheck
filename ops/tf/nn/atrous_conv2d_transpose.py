from schema import flib

def init_schema(op):
    op.add_index('b', 'batch', (1, 10))
    op.add_index('i', 'input spatial', (2, 2))
    op.add_index('k', 'input channel', (1, 1))
    op.add_index('f', 'filter spatial', 'i')
    op.add_index('l', 'output channel', (1, 1))
    op.add_index('o', 'output spatial', 'i')
    op.add_index('r', 'rate', (1, 1)) 

    op.arg_tensor('value', 'bik')
    op.arg_tensor('filters', 'flk')
    op.arg_shape_tensor('output_shape', 'bol')
    op.arg_shape_int('rate', 'r')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_unchecked('name')
    op.return_tensor('bol')

    op.add_index_generator('f', flib.gen_range, 'f', 1, 8)
    op.add_index_generator('r', flib.gen_range, 'r', 1, 8)

    def odims(i, f, r, padding):
        if padding == 'VALID':
            out = i - (f - 1) * r
        else:
            out = i
        return out

    def odims_templ(i, f, r, padding):
        if padding == 'VALID':
            txt = f'{i} - ({f} - 1) * 2'
        else:
            txt = f'{i}'
        return txt

    op.computed_index('o', odims, odims_templ, 'ifr', 1, 'padding')

    op.valid_dtypes('value', ('float',))
    op.equate_dtypes('filters', 'value')

