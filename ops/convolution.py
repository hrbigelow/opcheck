import opcheck
from schema import Broadcastable, Kind

def init_schema(op):
    op.add_index('b', 'batch')
    op.add_index('i', 'input spatial')
    op.add_index('f', 'filter spatial')
    op.add_index('o', 'output spatial')
    op.add_index('k', 'input channel')
    op.add_index('l', 'output channel')

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
    def pred_layout(data_format):
        return True, data_format[:2] == 'NC'
    
    op.arg_pseudo('channel_first', pred_layout, gen_layout, 'data_format')
    def input_sig(channel_first):
        if channel_first:
            return 'bki'
        else:
            return 'bik'
    op.arg_tensor('input', input_sig, 'channel_first')

    def df_pred(arg_val, rank_map, channel_first):
        valid = (arg_val == data_formats[rank_map['i'],channel_first])
        if valid:
            return True, arg_val
        else:
            return False, ArgValueError('data_format', arg_val) 

    def df_gen(rank_map, channel_first):
        return [data_formats[rank_map['i'],channel_first]]
    op.arg_func('data_format', df_pred, df_gen, Kind.RANKS, 'channel_first')

    op.arg_tensor('filters', lambda: 'fkl')
    op.arg_option('padding', ('VALID', 'SAME'))
    op.arg_sigrank('strides', 'i', 1, 10)
    op.arg_sigrank('dilations', 'i', 1, 1)

    op.valid_dtypes('input', ('int32', 'float32'))
    op.equate_dtypes('filters', 'input')

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

    op.computed_dims('o', odims, Kind.IDIMS, 'strides', 'dilations', 'padding')

    def return_sig(channel_first):
        if channel_first:
            return 'blo'
        else:
            return 'bol'
    op.return_tensor(return_sig, 'channel_first')

opcheck.register('tf.nn.convolution', init_schema)

