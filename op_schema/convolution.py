import opcheck
from schema import Broadcastable

op = opcheck.register('tf.nn.convolution')

def init_schema(op):
    op.add_index('b', 'batch')
    op.add_index('i', 'input spatial')
    op.add_index('f', 'filter spatial')
    op.add_index('o', 'output spatial')
    op.add_index('k', 'input channel')
    op.add_index('l', 'output channel')

    # inputs
    op.add_input_tensor('input', 'bik')
    op.add_input_tensor('filters', 'fkl')
    op.add_input_sigrank('strides', 'i', 1, 10, 2)
    op.add_input_static('padding', ('VALID', 'SAME'))
    op.add_input_static('data_format', ('NWC', 'NHWC', 'NDHWC', 'NCW', 'NCHW', 'NCDHW'))
    op.add_input_sigrank('dilations', 'i', 1, 10, 2)

    # outputs
    op.append_output_tensor('bol')

    # constraints
    op.set_rank_range('i', range(1, 4))
    op.equate_index_ranks('i', 'f')
    op.equate_index_ranks('i', 'o')
    
    def dcons(_op):
        idims = Broadcastable(_op.get_index('i').dims())
        fdims = Broadcastable(_op.get_index('f').dims())
        stride = _op.get_arg('strides', default=1) 
        dilation = _op.get_arg('dilations', default=1)

        if _op.get_arg('padding') == 'VALID':
            pad_filter_dims = (fdims - 1) * dilation + 1
            out = (idims - pad_filter_dims + 1).ceildiv(stride)
        else:
            out = idims.ceildiv(stride)
        return out.val

    op.set_index_dims_constraint('o', dcons)

def process_data_format(op):
    df = op.get_arg('data_format', default='NWC')
    if df in ('NWC', 'NHWC', 'NDHWC'):
        in_sig = 'bik'
    else:
        in_sig = 'bki'
    op.set_tensor_signature('input', in_sig)


op.set_init(init_schema)
op.set_calltime_config(process_data_format)


