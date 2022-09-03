import opcheck
from schema import Kind

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 2, 2)
    op.add_index('k', 'input channel', 1, 1)
    op.add_index('d', 'depthwise spatial')
    op.add_index('c', 'channel multiplier', 1, 1)
    op.add_index('z', 'constant dimension', 2, 2)
    op.add_index('o', 'output spatial')
    op.add_index('l', 'output channel', 1, 1)
    op.add_index('p', 'pointwise channel', 1, 1)
    op.add_index('s', 'strides')
    op.add_index('e', 'dilations')

    op.equate_ranks('d', 'i')
    op.equate_ranks('o', 'i')
    op.equate_ranks('s', 'i')
    op.equate_ranks('e', 'i')

    layouts = [ { 2: 'NHWC' }, { 2: 'NCHW' } ]
    op.arg_layout('data_format', layouts, 'i')
    op.arg_tensor('input', 'bik', 'bki')
    op.arg_tensor('depthwise_filter', 'dkc')
    op.arg_tensor('pointwise_filter', 'zpl')

    op.valid_dtypes('input', ('int32', 'float32'))
    op.equate_dtypes('depthwise_filter', 'input')
    op.equate_dtypes('pointwise_filter', 'input')

    def pdims(idims_map):
        cmul = idims_map['c']
        ichan = idims_map['k']
        flat = cmul * ichan
        return flat

    op.computed_index('p', pdims, Kind.IDIMS)
    op.computed_index('z', lambda: [1, 1])

    op.arg_option('padding', ('VALID', 'SAME'))

    op.arg_shape_list('strides', 's')
    op.arg_shape_list('dilations', 'e')

    op.arg_unchecked('name')

opcheck.register('tf.nn.separable_conv2d', init_schema)





