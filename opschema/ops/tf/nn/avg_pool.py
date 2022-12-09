from opschema import genlib, complib

def init_schema(op):
    op.add_index('b', 'batch', 1)
    op.add_index('i', 'input spatial', (1, 3))
    op.add_index('o', 'output spatial', 'i')
    op.add_index('c', 'channel', 1)
    op.add_index('k', 'ksize', 1)
    op.add_index('s', 'strides', 1)

    formats = {
            'NCW': (0, 1),
            'NCHW': (0, 2),
            'NCDHW': (0, 3),
            'NWC': (1, 1),
            'NHWC': (1, 2),
            'NDHWC': (1, 3),
            None: (1, None),
            }

    op.arg_layout('data_format', formats, 'i')
    op.arg_tensor('input', 'bci', 'bic')
    op.arg_shape_bcast_list('ksize', 'k')
    op.arg_shape_bcast_list('strides', 's')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_unchecked('name')

    op.gen_dims('b', 100)
    op.gen_dims('c', 50)
    op.gen_dims('k', 10)
    op.gen_dims('s', 100)
    op.gen_dims_func('i', genlib.below_above, 'k', 1000, False)

    op.valid_dtypes('input', ('bfloat16', 'float',))
    op.exclude_combos('input', ('float64', 'bfloat16'), 'i', 3)

    def odims(i, k, s, padding):
        if padding == 'VALID':
            tmp = i - k + 1
            out = complib.ceildiv(tmp, s)
        else:
            out = complib.ceildiv(i, s)
        return out

    def odims_t(i, k, s, padding):
        if padding == 'VALID':
            tem = f'ceil(({i} - {k} + 1) / {s})'
        else:
            tem = f'ceil({i} / {s})'
        return tem

    op.comp_dims_cw('o', odims, odims_t, 'iks', 'padding')

    op.return_tensor('bco', 'boc')

    
