import opcheck
op = opcheck.register('tf.gather_nd')

def init_schema(op):
    op.add_index('b', 'batch')
    op.add_index('r', 'read location')
    op.add_index('w', 'write location')
    op.add_index('e', 'slice element')
    op.add_index('c', 'read address component')

    # outer_shape = bw  (batch + write location) 
    # inner_shape = e (slice element)  
    # output_shape = bwe (outer_shape + inner_shape)
    op.add_input_tensor('indices', 'bwc')
    op.add_input_tensor('params', 'bre')
    op.append_output_tensor('bwe')
    op.add_input_rank('batch_dims', 'b')
    op.set_index_rank('c', 1)
    op.set_index_rank_range('r', range(1, 4))
    op.set_index_rank_range('w', range(0, 4))
    op.set_index_rank_range('e', range(0, 4))

    # number of elements for 'read location' are determined by the
    # size of the last dimension of the indices.
    # think of indices as a tensor of 1D slices, each one a 'read address'
    def dimsc(_op):
        return [_op.get_index('r').rank()]
    op.set_index_dims_constraint('c', dimsc)
    
op.set_init(init_schema)

