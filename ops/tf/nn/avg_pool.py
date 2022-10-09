def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 1, 3)
    op.add_index('c', 'channel', 1, 1)
    op.add_index('k', 'ksize')
    op.add_index('s', 'strides')

    op.equate_ranks('k', 'i')
    op.equate_ranks('s', 'i')

    data_formats = [ 
            { 1: 'NCW', 2: 'NCHW', 3: 'NCDHW' },
            { 1: 'NWC', 2: 'NHWC', 3: 'NDHWC' }
            ]
    
    op.arg_layout('data_format', data_formats, 'i')
    op.arg_tensor('input', 'bic', 'bci')
    op.arg_shape_bcast_list('ksize', 'k')
    op.arg_shape_bcast_list('strides', 's')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_unchecked('name')

    op.valid_dtypes('input', ('bfloat16', 'float',))

    op.exclude_dtypes(
            ('input', 'i'),
            ('float64', 3),
            ('bfloat16', 3)
            )

    op.return_tensor('bic')

    
