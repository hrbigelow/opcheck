def init_schema(op):
    op.add_index('b', 'batch', 1)
    op.add_index('l', 'leading', (0,4))
    op.add_index('c', 'channel', 1)

    formats = { 
            'NC..': (0, None),
            'N..C': (1, None)
            }

    op.arg_layout('data_format', formats, 'l')
    op.arg_tensor('value', 'bcl', 'blc')
    op.arg_tensor('bias', 'c')
    op.arg_unchecked('name')

    op.gen_dims('b', 100)
    op.gen_dims('l', 100)
    op.gen_dims('c', 100)

    op.valid_dtypes('value', ('int', 'complex', 'float', 'uint8'))
    op.equate_dtypes('bias', 'value')

    op.return_tensor('bcl', 'blc')

