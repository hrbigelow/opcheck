from opschema import genlib
from opschema.complib import dilate, dilate_t, tconv, tconv_t
from opschema.predlib import divis_by, divis_by_t
from opschema import genlib

def init_schema(op):
    op.add_index('b', 'batch', (1, 1))
    op.add_index('i', 'input spatial', (1,3))
    op.add_index('k', 'input channel', 1)
    op.add_index('n', 'strided input spatial', 'i')
    op.add_index('f', 'filter spatial', 'i')
    op.add_index('g', 'dilated filter spatial', 'i')
    op.add_index('s', 'strides', 'i')
    op.add_index('d', 'dilations', 'i')
    op.add_index('l', 'output channel', 1)
    op.add_index('j', 'filter output channel', 1)
    op.add_index('o', 'output spatial declared', 'i')
    op.add_index('q', 'output spatial computed', 'i')

    formats = {
            'NCW': (0, 1),
            'NCHW': (0, 2),
            'NCDHW': (0, 3),
            'NWC': (1, 1),
            'NHWC': (1, 2),
            'NDHWC': (1, 3),
            None: (1, None),
            }
    op.arg_layout('data_format', formats, 'i')
    op.arg_tensor('input', 'bki', 'bik')
    op.arg_tensor('filters', 'fjk')
    # op.arg_tensor('filters', 'flk')
    op.arg_shape_tensor('output_shape', 'blo', 'bol')
    op.arg_shape_bcast_list('strides', 's')
    op.arg_shape_bcast_list('dilations', 'd')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_unchecked('name')
    op.return_tensor('blq', 'bql')

    op.gen_dims('b', 100)
    op.gen_dims('f', 100)
    op.gen_dims_func('i', genlib.below_above, 'f', 1000, False)  
    op.gen_dims_func('s', genlib.stride_dil, '', 10, True) 
    op.gen_dims_func('d', genlib.stride_dil, '', 10, True) 
    op.gen_dims('k', 30)
    op.gen_dims('j', 30)
    # op.gen_dims('l', 30)

    wrap_divis_by = genlib.WrapParams(genlib.divis_by, 300)
    op.gen_dims_func('l', wrap_divis_by, 'j', 300, False)

    # input is dilated with 'strides' 
    op.comp_dims_cw('n', dilate, dilate_t, 'is') 
    # filter is dilated with 'dilations'
    op.comp_dims_cw('g', dilate, dilate_t, 'fd')

    def odims_gen(n, g, padding):
        if padding == 'VALID':
            val = n + g - 1
        else:
            val = n
        yield val, val

    op.gen_dims_calc('o', odims_gen, 'ng', 'padding')   
    op.comp_dims_cw('q', tconv, tconv_t, 'ng', 'padding')

    def oq_pred(o, q):
        return o == q

    def oq_pred_t(o, q):
        return f'{o} must equal {q}'

    op.dims_pred_cw('o == q', oq_pred, oq_pred_t, 'oq')
    op.dims_pred_cw('l % j == 0', divis_by, divis_by_t, 'lj')

    op.valid_dtypes('input', ('bfloat16', 'float16+',))
    op.equate_dtypes('filters', 'input')

