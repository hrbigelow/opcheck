import opcheck
from schema import Kind, flib
import numpy as np

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 2, 2)
    op.add_index('k', 'input channel', 1, 1)
    op.add_index('z', 'input channel / 4', 1, 1)
    op.add_index('o', 'output spatial', 2, 2)
    op.add_index('f', 'output flattened', 1, 1)
    op.add_index('c', 'vect c channel', 1, 1)
    op.add_index('s', 'block size', 1, 1)

    op.add_index_predicate('i % s == 0', flib.divis_by, 'is')

    def cdims(dummy):
        return [([4],)]
    op.add_index_generator(cdims, 'c', 'c')
    op.add_index_generator(flib.gen_blocked_sizes, 'is', 'i', 2, 8, 10, 100)

    data_formats = [ 
            { 2: 'NHWC' }, 
            { 2: 'NCHW' }, 
            { 2: 'NCHW_VECT_C' } 
            ]

    op.arg_layout('data_format', data_formats, 'i')
    op.arg_tensor('input', 'bik', 'bki', 'bzic')
    op.arg_shape_int('block_size', 's') 
    op.arg_unchecked('name')
    op.return_tensor('bof', 'bfo', 'bfoc')

    op.valid_dtypes('input', ('int32', 'float32'))

    def output_dims(dims_map):
        idims = dims_map['i']
        block_size = dims_map['s']
        odims = flib.floordiv(idims, block_size)
        return odims

    op.computed_index('o', output_dims, Kind.IDIMS)

    def flattened_dims(dims_map, layout):
        if layout == 'NCHW_VECT_C':
            zdims = dims_map['z']
            cdims = dims_map['c']
            block_size = dims_map['s']
            flat = block_size * block_size * zdims * cdims
        else:
            kdims = dims_map['k']
            block_size = dims_map['s']
            flat = block_size * block_size * flib.reduce_prod(kdims)
        return flat

    op.computed_index('f', flattened_dims, Kind.IDIMS, Kind.LAYOUT)

opcheck.register('tf.nn.space_to_depth', init_schema)




