def rankr(indices_shape):
    if len(indices_shape) == 0:
        return None
    else:
        return indices_shape[-1]

def init_schema(op):
    op.add_index('b', 'batch')
    op.add_index('r', 'read location', (1, 7))
    op.add_index('w', 'write location')
    op.add_index('e', 'slice element')
    op.add_index('c', 'read address component', 1)

    # allowed rank combinations
    op.limit_ranks('bwc', 0, 10)
    op.limit_ranks('bre', 0, 10)

    # generators
    op.gen_dims('b', 1, 500, 500, True)
    op.gen_dims('r', 1, 100, 100, True)
    op.gen_dims('w', 1, 100, 100, True)
    op.gen_dims('e', 1, 100, 100, True)
    op.gen_dims('c', 1, 8, 8, True)

    # argument interpretations
    op.arg_tensor('indices', 'bwc')
    op.arg_tensor('params', 'bre')
    op.arg_rank('batch_dims', 'b')
    op.arg_unchecked('name')

    # dtypes
    op.valid_dtypes('indices', ('int32+',))
    op.valid_dtypes('params', ('int32+', 'float'))

    op.rank_dims_constraint(rankr, 'r', 'indices')

    op.dims_pred_rng('b', 1, None)
    op.dims_pred_rng('r', 1, None)
    op.dims_pred_rng('e', 1, None)

    # output shape prediction
    op.return_tensor('bwe')

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

