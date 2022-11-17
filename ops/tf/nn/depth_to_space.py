from schema import flib

def init_schema(op):
    op.add_index('b', 'batch', 1)
    op.add_index('i', 'input spatial', 2)
    op.add_index('s', 'block size', 1, (2, None))
    op.add_index('k', 'input channel', 1)
    op.add_index('c', 'const dim 4', 1, 4)
    op.add_index('t', 'squared block size', 1)
    op.add_index('o', 'output spatial', 'i')
    op.add_index('f', 'output flattened', 1)
    op.add_index('g', 'output flattened / 4', 1)
    op.add_index('z', 'input channel / 4', 1)

    formats = {
            'NHWC': (0, 2),
            'NCHW': (1, 2),
            'NCHW_VECT_C': (2, 2)
            }

    op.arg_layout('data_format', formats, 'i')
    op.arg_tensor('input', 'bik', 'bki', 'bzic')
    op.arg_shape_int('block_size', 's') 
    op.arg_unchecked('name')
    op.return_tensor('bof', 'bfo', 'bgoc')
    op.valid_dtypes('input', ('int32', 'float32'))

    op.add_index_generator('s', flib.gen_range, '', 2, 8)
    # op.add_index_generator('is', flib.gen_blocked_sizes, 'i', 2, 8, 10, 100)
    op.add_index_predicate('k % t == 0', flib.divis_by, flib.divis_by_templ, 'kt')

    sq, sqt = lambda s: s * s, lambda s: f'{s} * {s}'
    mul, mult = lambda a, b: a * b, lambda a, b: f'{a} * {b}'
    div, divt = lambda a, b: a // b, lambda a, b: f'{a} // {b}'
    div4, div4t = lambda a: a // 4), lambda a: f'{a} // 4'

    op.computed_index('o', mul, mult, 'is')
    op.computed_index('t', sq, sqt, 's')
    op.computed_index('f', div, divt, 'kt')
    op.computed_index('g', div4, div4t, 'f')
    op.computed_index('z', div4, div4t, 'k')

