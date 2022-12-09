from opcheck.base import LAYOUT
from opcheck.complib import filter_pad, filter_pad_t, ceildiv
from opcheck.genlib import stride_dil, divis_by, below_above 
from opcheck import predlib, complib

def init_schema(op):
    op.add_index('b', 'batch', (1,5))
    op.add_index('i', 'input spatial', (1,3))
    op.add_index('f', 'filter spatial', 'i')
    op.add_index('p', 'padded filter spatial', 'i')
    op.add_index('s', 'strides', 'i')
    op.add_index('d', 'dilations', 'i')
    op.add_index('k', 'input channel', 1)
    op.add_index('j', 'output filter', 1)
    op.add_index('l', 'output channel', 1)
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

    op.gen_dims('b', 100)
    op.gen_dims('l', 30)
    op.gen_dims('f', 100)
    op.gen_dims('j', 30)
    op.gen_dims_func('s', stride_dil, '', 10, True) 
    op.gen_dims_func('d', stride_dil, '', 10, True) 
    op.comp_dims_cw('p', filter_pad, filter_pad_t, 'fd') 
    op.gen_dims_func('i', below_above, 'p', 1000, False)  
    op.gen_dims_func('k', divis_by, 'j', 300, False, 300)

    def odims(i, p, s, padding):
        if padding == 'VALID':
            out = ceildiv(i - p + 1, s)
        else:
            out = ceildiv(i, s)
        return out

    def odims_t(i, p, s, padding):
        if padding == 'VALID':
            tem = f'ceil(({i} - {p} + 1) / {s})'
        else:
            tem = f'ceil({i} / {s})' 
        return tem

    op.comp_dims_cw('o', complib.conv, complib.conv_t, 'ips', 'padding')

    op.valid_dtypes('input', ('int32', 'float', 'bfloat16'))
    op.equate_dtypes('filters', 'input')
    op.exclude_combos('input', 'int32', 'i', (1,2), LAYOUT, 0)
    op.exclude_combos('input', 'int32', 'i', 3)
    op.exclude_combos('input', 'bfloat16', 'i', (1,2))
    op.exclude_combos('input', 'bfloat16', 'i', 3, LAYOUT, 0)

    op.dims_pred('s-d exclusion', 
            predlib.not_both_over_one,
            predlib.not_both_over_one_templ, 'sd')

    op.dims_pred_cw('k % j == 0', predlib.divis_by, predlib.divis_by_t, 'kj')
    
    op.return_tensor('blo', 'bol')

