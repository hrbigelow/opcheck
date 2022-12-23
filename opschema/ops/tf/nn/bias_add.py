from opschema.base import LAYOUT

def init_schema(op):
    op.add_index('b', 'batch', 1)
    op.add_index('s', 'spatial', (0,3))
    op.add_index('c', 'channel', 1)

    formats = { 
            'NC..': (0, None),
            'N..C': (1, None),
            None: (1, None)
            }

    op.arg_layout('data_format', formats, 's')
    op.arg_tensor('value', 'bcs', 'bsc')
    op.arg_tensor('bias', 'c')
    op.arg_unchecked('name')

    op.gen_dims('b', 1, 100, 100, True)
    op.gen_dims('s', 1, 100, 100, True)
    op.gen_dims('c', 1, 100, 100, True)

    op.valid_dtypes('value', ('int', 'uint', 'float', 'bfloat', 'complex'))
    op.equate_dtypes('bias', 'value')

    excluded = ('int8', 'int16', 'int64', 'bfloat', 'uint', 'complex')
    op.exclude_combos('value', excluded, 's', 0, LAYOUT, 0)

    op.return_tensor('bcs', 'bsc')

