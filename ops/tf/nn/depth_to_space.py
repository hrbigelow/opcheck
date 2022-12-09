from opschema.predlib import divis_by, divis_by_t
from opschema import genlib

def init_schema(op):
    op.add_index('b', 'batch', 1)
    op.add_index('i', 'input spatial', 2)
    op.add_index('s', 'block size', 1)
    op.add_index('k', 'input channel', 1)
    op.add_index('c', 'const dim 4', 1)
    op.add_index('t', 'squared block size', 1)
    op.add_index('o', 'output spatial', 'i')
    op.add_index('f', 'output flattened', 1)

    op.dims_pred_cw('k % t == 0', divis_by, divis_by_t, 'kt')
    op.dims_pred_rng('c', 4, 4)

    formats = {
            'NHWC': (0, 2),
            'NCHW': (1, 2),
            'NCHW_VECT_C': (2, 2)
            }

    op.arg_layout('data_format', formats, 'i')
    op.arg_tensor('input', 'bik', 'bki', 'bkic')
    op.arg_shape_int('block_size', 's', 2, None) 
    op.arg_unchecked('name')
    op.return_tensor('bof', 'bfo', 'bfoc')
    op.valid_dtypes('input', ('int32', 'float32'))

    sq, sqt = lambda s: s * s, lambda s: f'{s} * {s}'
    mul, mult = lambda a, b: a * b, lambda a, b: f'{a} * {b}'
    div, divt = lambda a, b: a // b, lambda a, b: f'{a} // {b}'

    op.gen_dims('b', 100)
    op.gen_dims('i', 500)
    op.gen_dims_func('s', genlib.interval, '', 100, False, 1, 8)
    op.gen_dims_func('c', genlib.interval, '', 4, False, 3, 5)
    op.comp_dims_cw('o', mul, mult, 'is')
    op.comp_dims_cw('t', sq, sqt, 's')
    op.gen_dims_func('k', genlib.divis_by, 't', 100, False, 100)
    op.comp_dims_cw('f', div, divt, 'kt')

