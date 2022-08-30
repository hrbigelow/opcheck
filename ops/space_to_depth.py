import opcheck
from schema import Broadcastable, Kind
import numpy as np

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 2, 2)
    op.add_index('k', 'input channel', 1, 1)
    op.add_index('z', 'input channel / 4', 1, 1)
    op.add_index('o', 'output spatial', 2, 2)
    op.add_index('f', 'output flattened', 1, 1)
    op.add_index('c', 'vect c channel', 1, 1)
    op.add_index('s', 'block size', 2, None)

    data_formats = [ 
            { 2: 'NHWC' }, 
            { 2: 'NCHW' }, 
            { 2: 'NCHW_VECT_C' } 
            ]

    op.arg_layout('data_format', data_formats, 'i')
    op.arg_tensor('input', 'bik', 'bki', 'bzic')
    op.arg_shape_int('block_size', 's') 
    op.return_tensor('bof', 'bfo', 'bfoc')

    op.valid_dtypes('input', ('int32', 'float32'))

    def output_dims(dims_map, block_size):
        block_size = Broadcastable(block_size)
        idims = Broadcastable(dims_map['i'])
        odims = idims // block_size
        return Broadcastable.getval(odims)

    op.computed_dims('o', output_dims, Kind.IDIMS, 'block_size')

    def flattened_dims(dims_map, block_size):
        idims = Broadcastable(dims_map['i'])
        kdims = dims_map['k']
        block = idims % block_size
        flat = np.prod((*Broadcastable.getval(block), *kdims))
        return [int(flat)]

    op.computed_dims('f', flattened_dims, Kind.IDIMS, 'block_size')
    op.computed_dims('c', lambda: [4])

opcheck.register('tf.nn.space_to_depth', init_schema)




