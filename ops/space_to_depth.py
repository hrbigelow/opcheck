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
    op.add_index_generator('c', cdims, '')
    op.add_index_generator('is', flib.gen_blocked_sizes, 'i', 2, 8, 10, 100)

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

    def odims(i, s):
        return flib.floordiv(i, s)

    def odims_template(i, s):
        return f'{i} // {s}'

    op.computed_index('o', odims, odims_template, 'is', 1)

    def fdims(z, c, s, k, layout):
        if layout == 'NCHW_VECT_C':
            flat = s * s * z * c
        else:
            flat = s * s * flib.reduce_prod(k)
        return flat

    def fdims_template(z, c, s, k, layout):
        if layout == 'NCHW_VECT_C':
            tmp = f'{s} * {s} * {z} * {c}'
        else:
            tmp = f'{s} * {s} * product({k})'
        return tmp

    op.computed_index('f', fdims, fdims_template, 'zcsk', 1, Kind.DATA_FORMAT)

opcheck.register('tf.nn.space_to_depth', init_schema)

