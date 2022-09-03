import opcheck
from schema import Kind, flib

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
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
    op.arg_shape_list('strides', 's')
    op.arg_shape_list('dilations', 'd')

    op.valid_dtypes('input', ('int32', 'float32'))
    op.equate_dtypes('filters', 'input')

    op.add_index_predicate('stride-dilation exclusion', flib.not_both_over_one,
            'sd')
    op.add_index_generator(flib.gen_not_both_over_one, 'sd', 'sd', 1, 10)
    op.add_index_generator(flib.gen_range, 'f', 'f', 3, 10)
    
    # compute output spatial dimension 
    def odims(dims_map, padding):
        idims = dims_map['i']
        fdims = dims_map['f']
        strides = dims_map['s']
        dilations = dims_map['d']

        if padding == 'VALID':
            pad_filter_dims = (fdims - 1) * dilations + 1
            tmp = idims - pad_filter_dims + 1
            out = flib.ceildiv(tmp, strides)
        else:
            out = flib.ceildiv(idims, strides)
        return out 

    op.computed_index('o', odims, Kind.IDIMS, 'padding')
    op.return_tensor('blo', 'bol')

opcheck.register('tf.nn.convolution', init_schema)

