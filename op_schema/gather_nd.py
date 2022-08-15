import opcheck
op = opcheck.register('tf.gather_nd')

def init_schema(op):
    op.index('b', 'batch')
    op.index('r', 'read location')
    op.index('w', 'write location')
    op.index('e', 'slice element')
    op.index('c', 'read address component')

    # allowed rank combinations
    op.limit_ranks('c', 1, 1)
    op.limit_ranks('b', 1, None)
    op.limit_ranks('r', 1, None)
    op.limit_ranks('w', 1, None)
    op.limit_ranks('bre', None, 7)
    op.limit_ranks('bwc', None, 7)

    # argument interpretations
    op.arg_tensor('indices', 'bwc')
    op.arg_tensor('params', 'bre')
    op.arg_rank('batch_dims', 'b')

    def rankw(indices):
        return indices.shape[-1]
    op.arg_rank_func('indices', 'w', rankw)

    # output shape prediction
    op.append_output_tensor('bwe')

    # allowed dims combinations (see below)
    def dimsc(_op):
        return [_op.get_index_rank('r')]
    op.set_index_dims_constraint('c', dimsc)
    
op.set_init(init_schema)

"""
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

