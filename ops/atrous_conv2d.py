import opcheck
from schema import Broadcastable, Kind

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 2, 2)
    op.add_index('k', 'input channel', 1, 1)
    op.add_index('f', 'filter spatial', 2, 2)
    op.add_index('l', 'output channel', 1, 1)
    op.add_index('o', 'output spatial', 2, 2)

    op.arg_tensor('value', 'bik')
    op.arg_tensor('filters', 'fkl')
    op.arg_int('rate', 1, 10)
    op.arg_option('padding', ('VALID', 'SAME'))

    op.valid_dtypes('value', ('float32',))
    op.equate_dtypes('filters', 'value')

    def odims(dims_map, rate, padding):
        idims = Broadcastable(dims_map['i'])
        fdims = Broadcastable(dims_map['f'])
        if padding == 'VALID':
            out = idims - (fdims - 1) * 2
        else:
            out = idims
        return Broadcastable.getval(out)

    op.computed_dims('o', odims, Kind.IDIMS, 'rate', 'padding')

opcheck.register('tf.nn.atrous_conv2d', init_schema)


