import opcheck
op = opcheck.register('tf.nn.convolution')

def init_schema(op, pars):
    op.add_index('b', 'batch')
    op.add_index('i', 'input spatial')
    op.add_index('f', 'filter spatial')
    op.add_index('o', 'output spatial')
    op.add_index('k', 'input channel')
    op.add_index('l', 'output channel')

    if pars['data_format'] in (None, 'NWC', 'NHWC', 'NDHWC'):
        in_sig = 'bik'
    else:
        in_sig = 'bki'
    op.add_input_tensor('input', in_sig)
    op.add_input_tensor('filters', 'fkl')
    op.append_output_tensor('output', 'bol')

    op.set_rank_range('i', range(1, 4))
    op.equate_rank('i', 'f')
    op.equate_rank('i', 'o')
    
    def dcons(schema):
        idims = opcheck.Broadcastable(schema.index['i'].dims())
        fdims = opcheck.Broadcastable(schema.index['f'].dims())
        stride = schema.get_arg('strides') 
        dilation = schema.get_arg('dilations')
        if stride is None:
            stride = 1
        if dilation is None:
            dilation = 1

        if schema.arguments['padding'] == 'VALID':
            pad_filter_dims = (fdims - 1) * dilation + 1
            out = (idims - pad_filter_dims + 1).ceildiv(stride)
        else:
            out = idims.ceildiv(stride)
        return out.val

    op.add_dims_constraint('o', dcons)

op.set_init(init_schema)


