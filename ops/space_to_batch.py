import opcheck
from schema import flib

def init_schema(op):
    op.add_index('b', 'batch', 1, 1)
    op.add_index('i', 'input spatial', 1, 3)
    op.add_index('j', 'padded input spatial')
    op.add_index('k', 'block shape')
    op.add_index('r', 'remaining', 0, 10)
    op.add_index('s', 'padding start')
    op.add_index('e', 'padding end')
    op.add_index('o', 'output spatial')
    op.add_index('p', 'output batch', 1, 1)

    op.equate_ranks('s', 'i')
    op.equate_ranks('j', 'i')
    op.equate_ranks('e', 'i')
    op.equate_ranks('o', 'i')
    op.equate_ranks('k', 'i')

    op.arg_tensor('input', 'bir')
    op.arg_shape_tensor('block_shape', 'k')
    op.arg_shape_tensor2d('paddings', 's', 'e')
    op.arg_unchecked('name')

    # ensure that padded input is divisible by block size
    op.add_index_predicate('pad_input_block', flib.divis_by, 'jk')
    # op.add_index_predicate('pad_input_block', flib.pad_input_blocked, 'isek')

    # generates i, s, e, and k dimensions compatible with the predicate
    op.add_index_generator('isek', flib.gen_pad_input_blocked, 'i', 10, 50)

    op.valid_dtypes('input', ('int32', 'float32'))

    def jdims(s, e, i):
        return s + e + i

    def jdims_txt(s, e, i):
        return f'{s} + {i} + {e}'

    op.computed_index('j', jdims, jdims_txt, 'sei', 1)

    def odims(padded, block_shape):
        return flib.floordiv(padded, block_shape)

    def odims_txt(padded, block_shape):
        return f'{padded} // {block_shape}'

    op.computed_index('o', odims, odims_txt, 'jk', 1)

    def pdims(block_shape, batch):
        block_elems = flib.reduce_prod(block_shape)
        return block_elems * batch

    def pdims_txt(block_shape, batch):
        return f'product({block_shape}) * {batch}'

    op.computed_index('p', pdims, pdims_txt, 'kb', 1)

    op.return_tensor('por')

opcheck.register('tf.nn.space_to_batch', init_schema)

