import opcheck
from schema import flib

def init_schema(op):
    op.add_index('b', 'batch', 1, 5)
    op.add_index('i', 'input spatial', 1, 3)
    op.add_index('f', 'filter spatial', None, None)
    op.add_index('o', 'output spatial')
    op.add_index('k', 'input channel', 1, 1)
    op.add_index('l', 'output channel', 1, 1)
    op.add_index('s', 'strides', None, None)
    op.add_index('d', 'dilations', None, None)

    op.equate_ranks('f', 'i')
    op.equate_ranks('o', 'i')
    op.equate_ranks('s', 'i')
    op.equate_ranks('d', 'i')

    data_formats = [ 
            { 1: 'NCW', 2: 'NCHW', 3: 'NCDHW' },
            { 1: 'NWC', 2: 'NHWC', 3: 'NDHWC' }
            ]

    op.arg_layout('data_format', data_formats, 'i')
    op.arg_tensor('input', 'bki', 'bik')
    op.arg_tensor('filters', 'fkl')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_shape_bcast_list('strides', 's')
    op.arg_shape_bcast_list('dilations', 'd')
    op.arg_unchecked('name')

    op.valid_dtypes('input', ('int32', 'float', 'bfloat16'))
    op.equate_dtypes('filters', 'input')

    op.exclude_dtypes(
            ('input', 'i', ':layout'),
            ('int32', 3, 0),    # 3D int32 channel-first layout not implemented 
            ('int32', None, 1),   # all int32 channel-last not implemented 
            ('bfloat16', 1, None), # 1D bfloat16, any layout
            ('bfloat16', 3, None)  # 3D bfloat16, any layout
            )

    op.add_index_predicate('s-d exclusion', flib.not_both_over_one, 'sd')
    op.add_index_generator('sd', flib.gen_not_both_over_one, 'sd', 1, 3)
    op.add_index_generator('f', flib.gen_range, 'f', 3, 10)
    
    # compute output spatial dimension 
    def odims(i, f, s, d, padding):
        if padding == 'VALID':
            pad_filter_dims = (f - 1) * d + 1
            tmp = i - pad_filter_dims + 1
            out = flib.ceildiv(tmp, s)
        else:
            out = flib.ceildiv(i, s)
        return out

    def odims_template(i, f, s, d, padding):
        if padding == 'VALID':
            tem = f'ceil(({i} - ({f} - 1) * {d}) / {s})'
        else:
            tem = f'ceil({i} / {s})' 
        return tem

    op.computed_index('o', odims, odims_template, 'ifsd', 0, 'padding')
    op.return_tensor('blo', 'bol')

opcheck.register('tf.nn.convolution', init_schema)

