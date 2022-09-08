import opcheck
from schema import Kind, kname

def init_schema(op):
    op.add_index('r', 'read address')
    op.add_index('c', 'write address component')
    op.add_index('e', 'slice element')
    op.add_index('w', 'write address')

    def genc(rank_list):
        return [([rank_list[0]],)]
    op.add_index_generator(genc, 'c', 'w')

    op.limit_ranks('r', 1, 3)
    op.limit_ranks('c', 1, 1)
    op.limit_ranks('e', 0, 4)
    op.limit_ranks('w', 1, 3)

    op.arg_tensor('indices', 'rc')
    op.arg_tensor('updates', 're')
    op.arg_shape_list('shape', 'we')  
    op.arg_unchecked('name')
    op.return_tensor('we')

    def rankw(indices_shape):
        return indices_shape[-1]
    inds_kname = kname('indices', Kind.SHAPE) 
    op.rank_constraint('rank(w) == dims(c)', 'w', rankw, inds_kname)

    op.valid_dtypes('indices', ('int32',))
    # op.valid_dtypes('updates', ('int32', 'float32'))
    op.valid_dtypes('updates', ('float32',))

opcheck.register('tf.scatter_nd', init_schema)

"""
This schema determines index ranks and dims as follows:

1. The rank setting algorithm is executed.  In this case:
   rank(c) is implicitly set to 1 (since it has no rank constraint)
   rank(w) is set to indices.shape[-1]
   rank(r) and rank(e) are chosen as the first combo to satisfy constraints:
       rank(r) + rank(c) = rank(indices)
       rank(r) + rank(e) = rank(updates)
       rank(r) in range(1, 4)
       rank(e) in range(0, 4)
2. The dimension checking / setting algorithm is executed.  In this case:
   dims(r) is set to indices.shape[:rank(r)]
   dims(c) is set to indices.shape[rank(r):]
   dims(r) is checked against updates.shape[:rank(r)]
   dims(e) is set to updates.shape[rank(r):]
   dims(w) is set from the dims constraint function w_cons. (Equal to
     shape[:rank(e)])
   dims(e) is checked against the dims constraint function e_cons. (Equal to
     shape[rank(w):])

This can fail in step 1, if no consistent ranks are found.  This logically
means the user has provided invalid shapes to the operation.

If it succeeds, it is assumed that the signature interpretation is unambiguous.
Logically, this must be the case, otherwise the framework op would have no way
of dedicing which of multiple codepaths to execute.

The second step can fail if any of the 'checking' steps fails.  This means that
certain dimension constraints are violated.
   
"""
