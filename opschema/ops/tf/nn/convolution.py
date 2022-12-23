from opschema.base import LAYOUT
from opschema.complib import dilate, dilate_t, ceildiv, strided_conv, strided_conv_t
from opschema.genlib import WrapParams, stride_dil, group_channels, below_above 
from opschema import predlib, complib

def init_schema(op):
    op.add_index('b', 'batch', (1,5))
    op.add_index('i', 'input spatial', (1,3))
    op.add_index('f', 'filter spatial', 'i')
    op.add_index('g', 'dilated filter spatial', 'i')
    op.add_index('s', 'strides', 'i')
    op.add_index('d', 'dilations', 'i')
    op.add_index('k', 'input channel', 1)
    op.add_index('j', 'filter input channel', 1)
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

    op.gen_dims('b', 1, 100, 100, True)
    op.gen_dims('l', 1, 30, 30, True)
    op.gen_dims('f', 1, 30, 30, True)
    op.gen_dims_func('s', stride_dil, '', 10, True) 
    op.gen_dims_func('d', stride_dil, '', 5, True) 

    group_wrap = WrapParams(group_channels, 30)
    op.gen_dims_func('kj', group_wrap, '', 1000, False) 
    op.comp_dims_cw('g', dilate, dilate_t, 'fd') 
    op.gen_dims_func('i', below_above, 'g', 1000, False)  

    op.comp_dims_cw('o', strided_conv, strided_conv_t, 'igs', 'padding')

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

    def ratio_limit(k, j):
        return k < j * 10

    def ratio_limit_t(k, j):
        return f'{k} < {j} * 10'

    # op.dims_pred_cw('k < j * 10', ratio_limit, ratio_limit_t, 'kj')
    
    op.return_tensor('blo', 'bol')

