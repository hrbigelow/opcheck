import opcheck
from schema import Broadcastable

def init_schema(op):
    op.index('b', 'batch')
    op.index('i', 'input spatial')
    op.index('f', 'filter spatial')
    op.index('o', 'output spatial')
    op.index('k', 'input channel')
    op.index('l', 'output channel')

    # inputs
    op.arg_tensor('input', 'bik')
    op.arg_tensor('filters', 'fkl')
    op.add_input_sigrank('strides', 'i', 1, 10, 2)
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_option('data_format', ('NWC', 'NHWC', 'NDHWC', 'NCW', 'NCHW', 'NCDHW'))
    op.add_input_sigrank('dilations', 'i', 1, 10, 2)

    # outputs
    op.append_return_tensor('bol')

    # constraints
    op.limit_ranks('b', 1, 1)
    op.limit_ranks('i', 1, 3)
    op.equate_ranks('i', 'f')
    op.equate_ranks('i', 'o')
    op.limit_ranks('k', 1, 1)
    op.limit_ranks('l', 1, 1)
    
    def dcons(_op):
        idims = Broadcastable(_op.get_index_dims('i'))
        fdims = Broadcastable(_op.get_index_dims('f'))
        stride = _op.get_arg('strides', default=1) 
        dilation = _op.get_arg('dilations', default=1)

        if _op.get_arg('padding') == 'VALID':
            pad_filter_dims = (fdims - 1) * dilation + 1
            out = (idims - pad_filter_dims + 1).ceildiv(stride)
        else:
            out = idims.ceildiv(stride)
        return out.val

    op.index_dims_func('o', dcons)

def process_data_format(op):
    df = op.get_arg('data_format', default='NWC')
    if df in ('NWC', 'NHWC', 'NDHWC'):
        in_sig = 'bik'
    else:
        in_sig = 'bki'
    op.set_shape_signature('input', in_sig)

opcheck.register('tf.nn.convolution', init_schema, process_data_format)


