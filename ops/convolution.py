import opcheck
from schema import Broadcastable, IName

def init_schema(op):
    op.index('b', 'batch')
    op.index('i', 'input spatial')
    op.index('f', 'filter spatial')
    op.index('o', 'output spatial')
    op.index('k', 'input channel')
    op.index('l', 'output channel')

    # keys are rank, channel_first
    data_formats = { 
            (1,True): 'NCW',
            (1,False): 'NWC',
            (2,True): 'NCHW',
            (2,False): 'NHWC',
            (3,True): 'NCDHW',
            (3,False): 'NDHWC'
            }

    def gen_layout():
        return [True, False]
    def val_layout(data_format):
        return data_format[:2] == 'NC'
    op.arg_pseudo('channel_first', gen_layout, val_layout, 'data_format')

    def input_sig(channel_first):
        if channel_first:
            return 'bki'
        else:
            return 'bik'
    op.arg_tensor_func('input', input_sig, 'channel_first')

    def df_gen(rank_map, channel_first):
        return [data_formats[rank_map['i'],channel_first]]

    def df_pred(arg_val, rank_map, channel_first):
        return arg_val == data_formats[rank_map['i'],channel_first]
    op.arg_func('data_format', df_gen, df_pred, IName.RANKS, 'channel_first')

    def return_sig(channel_first):
        if channel_first:
            return 'blo'
        else:
            return 'bol'
    op.append_return_tensor_func(return_sig, 'channel_first')

    op.arg_tensor('filters', 'fkl')
    op.add_input_sigrank('strides', 'i', 1, 10)
    op.arg_option('padding', ('VALID', 'SAME'))
    op.add_input_sigrank('dilations', 'i', 1, 1)

    op.tensor_valid_dtypes('input', ('int32', 'float32'))
    op.tensor_equate_dtypes('filters', 'input')

    # constraints
    op.limit_ranks('b', 1, 1)
    op.limit_ranks('i', 1, 3)
    op.equate_ranks('f', 'i')
    op.equate_ranks('o', 'i')
    op.limit_ranks('k', 1, 1)
    op.limit_ranks('l', 1, 1)

    
    # compute output spatial dimension 
    def odims(dims_map, strides, dilations, padding):
        idims = Broadcastable(dims_map['i'])
        fdims = Broadcastable(dims_map['f'])
        if padding == 'VALID':
            pad_filter_dims = (fdims - 1) * dilations + 1
            out = (idims - pad_filter_dims + 1).ceildiv(strides)
        else:
            out = idims.ceildiv(strides)
        return out.val

    op.index_dims_func('o', odims, 'strides', 'dilations', 'padding')

opcheck.register('tf.nn.convolution', init_schema)

