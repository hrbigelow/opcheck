import opcheck
from schema import Kind

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 2, 2)
    op.add_index('k', 'input channel', 1, 1)
    op.add_index('f', 'filter spatial', 2, 2)
    op.add_index('l', 'output channel', 1, 1)
    op.add_index('o', 'output spatial', 2, 2)
    op.add_index('r', 'rate', 1, 1) 

    op.arg_tensor('value', 'bik')
    op.arg_tensor('filters', 'fkl')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_shape_int('rate', 'r')
    op.arg_unchecked('name')

    op.valid_dtypes('value', ('float32',))
    op.equate_dtypes('filters', 'value')

    def odims(dims_map, padding):
        idims = dims_map['i']
        fdims = dims_map['f']
        rate = dims_map['r']
        if padding == 'VALID':
            out = idims - (fdims - 1) * 2
        else:
            out = idims
        return out

    op.computed_index('o', odims, Kind.IDIMS, 'padding')
    # op.return_tensor()

opcheck.register('tf.nn.atrous_conv2d', init_schema)


