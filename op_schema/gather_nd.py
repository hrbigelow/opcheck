import opcheck
op = opcheck.register('tf.gather_nd')

"""
This is a
"""

def init_schema(op, pars):
    op.add_index('b', 'batch')
    op.add_index('r', 'read location')
    op.add_index('w', 'write location')
    op.add_index('e', 'slice element')
    op.add_index('c', 'read address component')

    # outer_shape = bw  (batch + write location) 
    # inner_shape = e (slice element)  
    # output_shape = bwe (outer_shape + inner_shape)
    op.add_signature('indices', 'bwc')
    op.add_signature('params', 'bre')

    op.append_output_signature('output', 'bwe')

    nb = pars['batch_dims']
    op.set_rank_range('b', range(nb, nb+1))
    op.set_rank_range('w', range(0, 4))
    op.set_rank_range('e', range(0, 4))
    # op.set_rank_range('c', range(1, 2))

    index_depth = pars['indices'].shape[-1]
    op.set_rank_range('r', range(index_depth, index_depth+1))

    def read_rank(schema):
        return schema.index['r'].rank()

    op.add_dims_constraint('c', read_rank)
    
op.set_init(init_schema)

