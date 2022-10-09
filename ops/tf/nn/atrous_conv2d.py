from schema import flib

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 2, 2)
    op.add_index('k', 'input channel', 1, 1)
    op.add_index('f', 'filter spatial')
    op.add_index('l', 'output channel', 1, 1)
    op.add_index('o', 'output spatial')
    op.add_index('r', 'rate', 1, 1) 

    op.equate_ranks('f', 'i')
    op.equate_ranks('o', 'i')

    op.arg_tensor('value', 'bik')
    op.arg_tensor('filters', 'fkl')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_shape_int('rate', 'r')
    op.arg_unchecked('name')

    op.valid_dtypes('value', ('int', 'float',))
    op.equate_dtypes('filters', 'value')

    op.add_index_generator('f', flib.gen_range, 'f', 3, 10)
    op.add_index_generator('r', flib.gen_range, 'r', 1, 10)

    def odims(i, f, r, padding):
        if padding == 'VALID':
            out = i - (f - 1) * r
        else:
            out = i
        return out

    def odims_txt(i, f, r, padding):
        if padding == 'VALID':
            txt = f'{i} - ({f} - 1) * 2'
        else:
            txt = f'{i}'
        return txt

    op.computed_index('o', odims, odims_txt, 'ifr', 1, 'padding')
    op.return_tensor('bol')


