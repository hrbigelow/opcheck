import opcheck
from schema import Broadcastable
import numpy as np

def init_schema(op):
    op.index('b', 'batch')
    op.index('i', 'input spatial')
    op.index('k', 'input channel')
    op.index('o', 'output spatial')
    op.index('f', 'output flattened')
    op.index('c', 'vect c channel')

    op.arg_tensor('input', 'bik')
    op.arg_option('data_format', ('NHWC', 'NCHW', 'NCHW_VECT_C')) 
    op.append_return_tensor('bof')

    op.limit_ranks('b', 1, 1)
    op.limit_ranks('i', 2, 2)
    op.limit_ranks('k', 1, 1)
    op.limit_ranks('o', 2, 2) 
    op.limit_ranks('f', 1, 1)
    op.limit_ranks('c', 1, 1)

    def output_dims(_op):
        block_size = Broadcastable(_op.get_arg('block_size'))
        idims = _op.get_index_dims('i')
        return idims // block_size

    op.index_dims_func('o', output_dims)

    def flattened_dims(_op):
        block_size = Broadcastable(_op.get_arg('block_size'))
        idims = _op.get_index_dims('i')
        kdims = _op.get_index_dims('k')
        block = idims % block_size
        return np.prod(block.getval() + kdims)

    op.index_dims_func('f', flattened_dims)
    op.index_dims_func('c', lambda op: 4)

def calltime_func(op):
    data_format = op.get_arg('data_format', default='NHWC')
    insig, outsig = None, None
    if data_format == 'NHWC':
        insig, outsig = 'bik', 'bof'
    elif data_format == 'NCHW':
        insig, outsig = 'bki', 'bfo'
    elif data_format == 'NCHW_VECT_C':
        pass
    op.set_shape_signature('input', insig)
    op.set_output_signature(0, outsig)

opcheck.register('tf.nn.space_to_depth', init_schema, calltime_func)




