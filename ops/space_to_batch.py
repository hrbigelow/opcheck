import opcheck
from schema import Broadcastable as B, Kind

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 1, None)
    op.add_index('k', 'block shape', 1, None)
    op.add_index('r', 'remaining', 0, None)
    op.add_index('s', 'padding start')
    op.add_index('e', 'padding end')
    op.add_index('o', 'output spatial')
    op.add_index('p', 'output batch')

    op.arg_tensor('input', 'bir')
    op.arg_shape_tensor('block_shape', 'i')
    op.arg_shape_tensor2d('paddings', 's', 'e')

    op.valid_dtypes('input', ('int32', 'float32'))

    def odims(idims_map):
        input_spatial = B(idims_map['i'])
        block_shape = B(idims_map['k'])
        pad_start = B(idims_map['s'])
        pad_end = B(idims_map['e'])
        padded = input_spatial + pad_start + pad_end
        output_spatial = padded // block_shape
        return B.getval(output_spatial)

    op.computed_dims('o', odims, Kind.IDIMS)

    def pdims(idims_map):
        block_elems = np.prod(idims_map['k'])
        batch = idims_map['b'][0]
        out_batch = [block_elems * batch]

    op.computed_dims('p', pdims, Kind.IDIMS)
    op.return_tensor('po')

opcheck.register('tf.nn.space_to_batch', init_schema)

