import tensorflow as tf
from sly import Lexer, Parser

class ArgLexer(Lexer):
    tokens = { ID, DIMS, RANK, LPAREN, RPAREN }
    ignore = '\t'
    ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
    ID['DIMS'] = DIMS
    ID['RANK'] = RANK
    LPAREN  = r'\('
    RPAREN  = r'\)'


class ArgParser(Parser):
    tokens = ArgLexer.tokens

    def __init__(self, array_map, dims_map):
        self.array_map = array_map
        self.dims_map = dims_map
        self.lexer = ArgLexer()

    @_('ID',
       'DIMS LPAREN ID RPAREN',
       'RANK LPAREN ID RPAREN'
       )
    def argument(self, p):
        if hasattr(p, 'DIMS') and p.ID in self.dims_map:
            return self.dims_map[p.ID]
        elif hasattr(p, 'RANK') and p.ID in self.dims_map:
            return len(self.dims_map[p.ID])
        elif p.ID in self.array_map:
            return tf.convert_to_tensor(self.array_map[p.ID].ary)
        else:
            raise RuntimeError('argument must be DIMS(), RANK() or array name,'
                    f' got {list(p)}')

    def parse(self, arg_string):
        return super().parse(self.lexer.tokenize(arg_string))


