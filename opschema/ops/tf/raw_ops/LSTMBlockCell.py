def init_schema(op):
    op.add_index('b', 'batch', 1)
    op.add_index('i', 'input', 1)
    op.add_index('c', 'cell', 1)
    op.add_index('j', 'input + cell', 1)
    op.add_index('d', 'four cells', 1)

    op.arg_tensor('x', 'bi')
    op.arg_tensor('cs_prev', 'bc')
    op.arg_tensor('h_prev', 'bc')
    op.arg_tensor('w', 'jd')
    op.arg_tensor('b', 'd')
    op.arg_tensor('wci', 'c')
    op.arg_tensor('wcf', 'c')
    op.arg_tensor('wco', 'c')

    op.arg_unchecked('name')

    op.valid_dtypes('x', ('float16', 'float32'))
    op.equate_dtypes('cs_prev', 'x')
    op.equate_dtypes('h_prev', 'x')
    op.equate_dtypes('w', 'x')
    op.equate_dtypes('wci', 'x')
    op.equate_dtypes('wcf', 'x')
    op.equate_dtypes('wco', 'x')
    op.equate_dtypes('b', 'x')

    op.gen_dims('b', 1, 500, 500, True)
    op.gen_dims('i', 1, 500, 500, True)
    op.gen_dims('c', 1, 300, 300, True)

    def add(i, c):
        val = i+c
        yield val, val

    op.gen_dims_calc('j', add, 'ic')

    def four(c):
        val = 4 * c
        yield val, val

    op.gen_dims_calc('d', four, 'c')

    jpred = lambda j, i, c: j == i + c
    jpred_t = lambda j, i, c: f'{j} == {i} + {c}'
    op.dims_pred_cw('j == i + c', jpred, jpred_t, 'jic')

    dpred = lambda d, c: d == c * 4
    dpred_t = lambda d, c: f'{d} == {c} * 4'

    op.dims_pred_cw('d == c * 4', dpred, dpred_t, 'dc')


