from schema import flib, LAYOUT
from schema.flib import divis_by, divis_by_t

def init_schema(op):
    op.add_index('b', 'batch', 1)
    op.add_index('i', 'input spatial', 2)
    op.add_index('k', 'input channel', 1)
    op.add_index('o', 'output spatial', 'i')
    op.add_index('f', 'output flattened', 1)
    op.add_index('c', 'vect c channel', 1)
    op.add_index('s', 'block size', 1)
    op.add_index('t', 'squared block size', 1)

    op.add_index_predicate('i % s == 0', divis_by, divis_by_t, 'is')

    formats = {
            'NHWC': (0, 2),
            'NCHW': (1, 2),
            'NCHW_VECT_C': (2, 2)
            }

    op.arg_layout('data_format', formats, 'i')
    op.arg_tensor('input', 'bik', 'bki', 'bkic')
    op.arg_shape_int('block_size', 's') 
    op.arg_unchecked('name')
    op.return_tensor('bof', 'bfo', 'bfoc')

    valid_dt = ('bool', 'complex', 'qint8-', 'bfloat', 'int', 'float', 'uint')
    op.valid_dtypes('input', valid_dt)

    non_vect = ('int', 'uint16+', 'float64', 'bool', 'bfloat')
    op.exclude_combos('input', non_vect, LAYOUT, (1,2))
    op.exclude_combos('input', 'complex', LAYOUT, 1)
    op.exclude_combos('input', 'complex128', LAYOUT, 2)

    div4, div4t = lambda a: a // 4, lambda a: f'{a} // 4'
    sq, sqt = lambda s: s * s, lambda s: f'{s} * {s}'

    op.gen_dims('i', 100)
    op.gen_dims_rng('s', 10, 100)
    op.comp_dims('t', sq, sqt, 's')
    op.gen_dims('b', 100)
    op.gen_dims('k', 20)
    op.gen_dims_rng('c', 4, 4)

    def odims(i, s):
        return flib.floordiv(i, s)

    def odims_t(i, s):
        return f'{i} // {s}'

    op.comp_dims('o', odims, odims_t, 'is')

    def fdims(c, t, k, layout):
        if layout == 2:
            flat = t * k * c
        else:
            flat = t * flib.reduce_prod(k)
        return flat

    def fdims_t(c, t, k, layout):
        if layout == 2:
            tmp = f'{t} * {k} * {c}'
        else:
            tmp = f'{t} * product({k})'
        return tmp

    op.comp_dims('f', fdims, fdims_t, 'ctk', LAYOUT)

