import opcheck
op = opcheck.register('tf.nn.convolution')

# print('got here in convolution.py')

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
    op.add_signature('input', in_sig)
    op.add_signature('filters', 'fkl')

    op.append_output_signature('output', 'bol')

    op.set_rank_range('i', range(1, 4))
    op.equate_rank('i', 'f')
    op.equate_rank('i', 'o')
    
    """
    def dcons():
        if pars['padding'] == 'VALID':
            pad_filter_dims = (op.dims('f') - 1) * op.arg('dilation') + 1
            return ( (op.dims('i') - pad_filter_dims + 1) //^ op.arg('stride'))
        else:
            return op.dims('i') //^ op.arg('stride')

    op.add_dims_constraint('o', dcons)
    """

op.set_init(init_schema)


