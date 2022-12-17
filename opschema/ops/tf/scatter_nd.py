def init_schema(op):
    op.add_index('r', 'read address', (1, 4))
    op.add_index('e', 'slice element', (0, 4))
    op.add_index('c', 'write address component', 1)
    op.add_index('w', 'write address', (1, 5))

    op.gen_dims('r', 100)
    op.gen_dims('e', 100)
    op.gen_dims('c', 8)
    op.gen_dims('w', 100)

    # op.limit_ranks('rc', 1, 10)
    # op.limit_ranks('re', 1, 10)
    # op.limit_ranks('we', 1, 10)

    op.arg_tensor('indices', 'rc')
    op.arg_tensor('updates', 're')
    op.arg_shape_list('shape', 'we')  
    op.arg_unchecked('name')
    op.return_tensor('we')

    def rankw(indices_shape):
        if len(indices_shape) == 0:
            return None
        else:
            return indices_shape[-1]
    op.rank_dims_constraint(rankw, 'w', 'indices')

    op.valid_dtypes('indices', ('int32+',))
    # op.valid_dtypes('updates', ('int32', 'float32'))
    op.valid_dtypes('updates', ('int', 'float', 'complex', 'bool'))

