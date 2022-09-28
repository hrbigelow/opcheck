import opcheck
from schema import flib

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 2, 2)
    op.add_index('k', 'input channel', 1, 1)
    op.add_index('z', 'input channel / 4', 1, 1)
    op.add_index('o', 'output spatial', 2, 2)
    op.add_index('f', 'output flattened', 1, 1)
    op.add_index('c', 'vect c channel', 1, 1)
    op.add_index('s', 'block size', 1, 1)
    op.add_index('t', 'squared block size', 1, 1)

    def tdims(s):
        return s * s

    def tdims_txt(s):
        return f'{s} * {s}'

    # depth of the input tensor must be divisible by block_size * block_size.
    op.computed_index('t', tdims, tdims_txt, 's', 1)
    op.add_index_predicate('k % t == 0', flib.divis_by, 'it')

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
    op.valid_dtypes('input', ('int32', 'float32'))

    # width of the output tensor is input spatial * block_size
    def odims(i, s):
        return i * s

    def odims_txt(i, s):
        return f'{i} * {s}'

    op.computed_index('o', odims, odims_txt, 'is', 1)

opcheck.register('tf.nn.depth_to_space', init_schema)
