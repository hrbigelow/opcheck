from schema import flib, LAYOUT
from schema.flib import filter_pad, filter_pad_t, gen_divis_by, gen_split
from schema.flib import gen_stride_dil

def init_schema(op):
    op.add_index('b', 'batch', (1,5))
    op.add_index('i', 'input spatial', (1,3))
    op.add_index('f', 'filter spatial', 'i')
    op.add_index('p', 'padded filter spatial', 'i')
    op.add_index('s', 'strides', 'i')
    op.add_index('d', 'dilations', 'i')
    op.add_index('k', 'input channel', (1, 1))
    op.add_index('j', 'output filter', 'k')
    op.add_index('l', 'output channel', 'k')
    op.add_index('o', 'output spatial', 'i')

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
    op.arg_tensor('input', 'bki', 'bik')
    op.arg_tensor('filters', 'fjl')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_shape_bcast_list('strides', 's')
    op.arg_shape_bcast_list('dilations', 'd')
    op.arg_unchecked('name')

    op.gen_dims('b', 500)
    op.gen_dims('l', 100)
    op.gen_dims('f', 500)
    op.gen_dims_func('s', gen_stride_dil, '', 10, True) 
    op.gen_dims_func('d', gen_stride_dil, '', 10, True) 
    op.comp_dims('p', filter_pad, filter_pad_t, 'fd') 
    op.gen_dims_func('i', gen_split, 'p', 1000, False)  
    op.gen_dims_func('jk', gen_divis_by, '', 100, False, 1, 10)

    def odims(i, p, s, padding):
        if padding == 'VALID':
            out = flib.ceildiv(i - p + 1, s)
        else:
            out = flib.ceildiv(i, s)
        return out

    def odims_t(i, p, s, padding):
        if padding == 'VALID':
            tem = f'ceil(({i} - {p} + 1) / {s})'
        else:
            tem = f'ceil({i} / {s})' 
        return tem
    op.comp_dims('o', odims, odims_t, 'ips', 'padding')



    op.valid_dtypes('input', ('int32', 'float', 'bfloat16'))
    op.equate_dtypes('filters', 'input')

    op.exclude_combos('input', 'int32', 'i', (1,2), LAYOUT, 0)
    op.exclude_combos('input', 'int32', 'i', 3)
    op.exclude_combos('input', 'bfloat16', 'i', (1,2))
    op.exclude_combos('input', 'bfloat16', 'i', 3, LAYOUT, 0)

    op.add_index_predicate('s-d exclusion', flib.not_both_over_one,
            flib.not_both_over_one_templ, 'sd')

    op.add_index_predicate('k % j == 0', flib.divis_by, flib.divis_by_templ, 'kj')
    
    op.return_tensor('blo', 'bol')

