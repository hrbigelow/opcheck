from opschema.base import LAYOUT
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
    op.valid_dtypes('input', ('int', 'float', 'uint', 'qint', 'bfloat', 'bool', 'complex'))

    op.exclude_combos('input', 'int', LAYOUT, (1,2))
    op.exclude_combos('input', 'qint', LAYOUT, (0,1))
    op.exclude_combos('input', ('float16', 'float32'), LAYOUT, 2)
    op.exclude_combos('input', 'float64', LAYOUT, (1,2))
    op.exclude_combos('input', 'uint', LAYOUT, (1,2))
    op.exclude_combos('input', 'bfloat16', LAYOUT, 1)
    op.exclude_combos('input', 'bool', LAYOUT, 1)
    op.exclude_combos('input', 'complex', LAYOUT, 1)

    sq, sqt = lambda s: s * s, lambda s: f'{s} * {s}'
    mul, mult = lambda a, b: a * b, lambda a, b: f'{a} * {b}'
    div, divt = lambda a, b: a // b, lambda a, b: f'{a} // {b}'

    op.gen_dims('b', 1, 100, 100, True)
    op.gen_dims('i', 1, 500, 500, True)

    interval18 = genlib.WrapParams(genlib.interval, 1, 8)
    op.gen_dims_func('s', interval18, '', 100, False)

    interval35 = genlib.WrapParams(genlib.interval, 3, 5)
    op.gen_dims_func('c', interval35, '', 4, False)

    op.comp_dims_cw('o', mul, mult, 'is')
    op.comp_dims_cw('t', sq, sqt, 's')

    divis_by100 = genlib.WrapParams(genlib.divis_by, 100)
    op.gen_dims_func('k', divis_by100, 't', 100, False)

    op.comp_dims_cw('f', div, divt, 'kt')

