import opcheck

def init_schema(op):
    op.index('b', 'batch')
    op.index('r', 'read location')
    op.index('w', 'write location')
    op.index('e', 'slice element')
    op.index('c', 'read address component')

    # allowed rank combinations
    op.limit_ranks('c', 1, 1)
    op.limit_ranks('r', 1, None)
    op.limit_ranks('w', 1, None)
    # op.limit_ranks('bre', None, 3)
    # op.limit_ranks('bwc', None, 3)

    op.limit_ranks('bre', None, 7)
    op.limit_ranks('bwc', None, 7)

    # argument interpretations
    op.arg_tensor('indices', 'bwc')
    op.arg_tensor('params', 'bre')
    op.arg_rank('batch_dims', 'b')
    op.arg_unchecked('name')

    # dtypes
    op.tensor_valid_dtypes('indices', ('int32', 'int64'))
    op.tensor_valid_dtypes('params', ('int32', 'float32'))

    def rankr(indices):
        return indices.shape[-1]
    op.arg_rank_func('r', rankr, 'indices')

    # allowed dims combinations (see below)
    def dimsc(rank_map):
        return [rank_map['r']]
    op.index_rank_func('c', dimsc)

    # output shape prediction
    op.append_return_tensor('bwe')

    
opcheck.register('tf.gather_nd', init_schema)

"""
Rank Inference is unambiguous:
rank(c) = 1
rank(b) = batch_dims
rank(w) = rank(indices) - rank(c) - rank(b)
rank(r) = dims(c)[0]

rank inference constraints - necessary to infer the actual rank combos from a
given call

from TensorFlow docs
(https://www.tensorflow.org/api_docs/python/tf/gather_nd)
index_depth = indices.shape[-1]
outer_shape = indices.shape[:-1]
assert index_depth <= params.shape.rank
inner_shape = params.shape[index_depth:]
output_shape = outer_shape + inner_shape

Interpretation:
inner_shape = e (slice element)  
outer_shape = bw (batch + write location) 
output_shape = bwe (outer_shape + inner_shape)
"""

