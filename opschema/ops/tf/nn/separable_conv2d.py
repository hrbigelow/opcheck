from opschema.genlib import stride_dil

# Requires a polymorphic 'padding' API which accepts either a string 'VALID' or
# 'SAME', or a python list of lists

def init_schema(op):
    op.add_index('b', 'batch', 1)
    op.add_index('i', 'input spatial', 2)
    op.add_index('k', 'input channel', 1)
    op.add_index('p', 'padded input spatial', 'i') 
    op.add_index('d', 'depthwise spatial', 'i')
    op.add_index('c', 'channel multiplier', 1)
    op.add_index('z', 'constant 1', 2)
    op.add_index('o', 'output spatial', 'i')
    op.add_index('l', 'output channel', 1)
    op.add_index('w', 'pointwise channel', 1)
    op.add_index('s', 'strides', 'i')
    op.add_index('e', 'dilations', 'i')
    formats = {
            'NCHW': (0, 2),
            'NHWC': (1, 2)
            }
    op.arg_layout('data_format', formats, 'i')
    op.arg_tensor('input', 'bki', 'bik')
    op.arg_tensor('depthwise_filter', 'dkc')
    op.arg_tensor('pointwise_filter', 'zwl')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_shape_bcast_list('strides', 's')
    op.arg_shape_bcast_list('dilations', 'e')
    op.arg_unchecked('name')
    op.return_tensor('blo', 'bol')

    op.valid_dtypes('input', ('int32', 'float32'))
    op.equate_dtypes('depthwise_filter', 'input')
    op.equate_dtypes('pointwise_filter', 'input')

    op.gen_dims('b', 100)
    op.gen_dims('i', 50)
    op.gen_dims('k', 30)
    op.gen_dims('c', 30)
    op.gen_dims_rng('z', 1, 1)
    op.gen_dims('l', 30)
    op.gen_dims_func('s', stride_dil, '', 30, True)
    op.gen_dims_func('d', stride_dil, '', 30, True)

    def wdims(c, k):
        return c * k

    def wdims_t(c, k):
        return f'{c} * {k}'

    op.comp_dims_cw('p', dilate, dilate_t, 'i
    op.comp_dims_cw('w', wdims, wdims_t, 'ck')

