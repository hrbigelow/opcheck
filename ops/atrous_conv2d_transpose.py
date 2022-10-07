import opcheck

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 2, 2)
    op.add_index('k', 'input channel', 1, 1)
    op.add_index('f', 'filter spatial', 2, 2)
    op.add_index('l', 'output channel', 1, 1)
    op.add_index('o', 'output spatial', 2, 2)

    op.arg_tensor('value', 'bik')
    op.arg_tensor('filters', 'flk')
    op.arg_shape_tensor('output_shape', 'bol')
    op.arg_int('rate', 1, 10)
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_unchecked('name')
    op.return_tensor('bol')

    op.valid_dtypes('value', ('float32',))
    op.equate_dtypes('filters', 'value')


opcheck.register('tf.nn.atrous_conv2d_transpose', init_schema)

