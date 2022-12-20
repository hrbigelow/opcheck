def init_schema(op):
    op.add_index('b', 'batch', 1)
    op.add_index('i', 'input', 1)

    op.arg_tensor('x', 'bi')
    op.arg_tensor('cs_prev', 'bi')
    op.arg_tensor('h_prev', 'bi')
    op.arg_tensor('w', 'ii')
    op.arg_tensor('wci', 'ii')
    op.arg_tensor('wcf', 'ii')
    op.arg_tensor('wco', 'ii')
    op.arg_tensor('b', 'i')

    op.arg_unchecked('name')

    op.valid_dtypes('x', ('float16', 'float32'))
    op.equate_dtypes('cs_prev', 'x')
    op.equate_dtypes('h_prev', 'x')
    op.equate_dtypes('w', 'x')
    op.equate_dtypes('wci', 'x')
    op.equate_dtypes('wcf', 'x')
    op.equate_dtypes('wco', 'x')
    op.equate_dtypes('b', 'x')

    op.gen_dims('b', 500)
    op.gen_dims('i', 500)

