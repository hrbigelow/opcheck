import opcheck
from schema import Kind, flib

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 1, 3)
    op.add_index('k', 'block shape')
    op.add_index('r', 'remaining', 0, 10)
    op.add_index('s', 'padding start')
    op.add_index('e', 'padding end')
    op.add_index('o', 'output spatial')
    op.add_index('p', 'output batch', 1, 1)

    op.equate_ranks('s', 'i')
    op.equate_ranks('e', 'i')
    op.equate_ranks('o', 'i')
    op.equate_ranks('k', 'i')

    op.arg_tensor('input', 'bir')
    op.arg_shape_tensor('block_shape', 'k')
    op.arg_shape_tensor2d('paddings', 's', 'e')
    op.arg_unchecked('name')

    # ensure that padded input is divisible by block size
    op.add_index_predicate('pad_input_block', flib.pad_input_blocked, 'isek')

    # generates i, s, e, and k dimensions compatible with the predicate
    op.add_index_generator(flib.gen_pad_input_blocked, 'isek', 'i', 10, 50)

    op.valid_dtypes('input', ('int32', 'float32'))

    def odims(idims_map):
        input_spatial = idims_map['i']
        block_shape = idims_map['k']
        pad_start = idims_map['s']
        pad_end = idims_map['e']
        padded = input_spatial + pad_start + pad_end
        output_spatial = flib.floordiv(padded, block_shape)
        return output_spatial

    op.computed_index('o', odims, Kind.IDIMS)

    def pdims(idims_map):
        block_shape = idims_map['k']
        block_elems = flib.reduce_prod(block_shape)
        batch = idims_map['b']
        out_batch = block_elems * batch
        return out_batch

    op.computed_index('p', pdims, Kind.IDIMS)
    op.return_tensor('por')

opcheck.register('tf.nn.space_to_batch', init_schema)

