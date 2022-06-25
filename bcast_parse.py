from sly import Lexer, Parser
from bcast_ast import *

class BCLexer(Lexer):
    # Set of token names.   This is always required
    tokens = { ID, LPAREN, RPAREN, LBRACK, RBRACK, PLUS, MINUS, TIMES, DIVIDE,
            INT, ASSIGN, ACCUM, COMMA, COLON, DIMS, RANGE, RANK, RANDOM, DTYPE }

    # String containing ignored characters between tokens
    ignore = ' \t'

    # Regular expression rules for tokens
    ID    = r'[a-z]+'
    COMMA   = r','
    COLON   = r':'
    INT     = r'[0-9]+'
    ASSIGN  = r'='
    ACCUM   = r'\+='
    LPAREN  = r'\('
    RPAREN  = r'\)'
    LBRACK  = r'\['
    RBRACK  = r'\]'
    PLUS    = r'\+'
    MINUS   = r'\-'
    TIMES   = r'\*'
    DIVIDE  = r'\/'
    DIMS    = 'DIMS'
    RANGE   = 'RANGE'
    RANK    = 'RANK'
    RANDOM  = 'RANDOM'
    DTYPE   = r'(FLOAT|INT)'

class BCParser(Parser):
    tokens = BCLexer.tokens
    precedence = (
       ('left', PLUS, MINUS),
       ('left', TIMES, DIVIDE),
    )

    def __init__(self, cfg):
        self.lexer = BCLexer()
        self.cfg = cfg 

    @_('lval_array ASSIGN rval_expr',
       'lval_array ACCUM rval_expr')
    def assignment(self, p):
        do_accum = hasattr(p, 'ACCUM')
        return Assign(p.lval_array, p.rval_expr, do_accum)

    @_('ID LBRACK index_list RBRACK')
    def lval_array(self, p):
        return LValueArray(self.cfg, p.ID, p.index_list)

    @_('rval_array', 
       'rand_call', 
       'range_array')
    def rval_unit(self, p):
        return p[0]

    @_('rval_unit',
       'rval_unit oper rval_unit')
    def rval_expr(self, p):
        if hasattr(p, 'oper'):
            return ArrayBinOp(p.rval_unit0, p.rval_unit1, p.oper)
        else:
            return p.rval_unit

    @_('PLUS', 'MINUS', 'TIMES', 'DIVIDE')
    def oper(self, p):
        return p[0]

    @_('ID LBRACK star_index_list RBRACK')
    def rval_array(self, p):
        return RValueArray(self.cfg, p.ID, p.star_index_list)

    @_('RANDOM LPAREN array_slice COMMA array_slice COMMA DTYPE RPAREN')
    def rand_call(self, p):
        return RandomCall(self.cfg, p.array_slice0, p.array_slice1, p.DTYPE)

    @_('RANGE LBRACK ID COMMA ID RBRACK')
    def range_array(self, p):
        return RangeExpr(self.cfg, p.ID0, p.ID1)

    @_('index_expr', 
       'index_list COMMA index_expr')
    def index_list(self, p):
        if hasattr(p, 'COMMA'):
            return p.index_list + [p.index_expr]
        else:
            return [p.index_expr]

    @_('ID',
       'array_slice')
    def index_expr(self, p):
        return p[0]

    @_('DIMS LPAREN ID RPAREN LBRACK star_or_index RBRACK',
       'RANK LPAREN ID RPAREN',
       'INT',
       'ID LBRACK star_index_list RBRACK')
    def array_slice(self, p):
        if hasattr(p, 'DIMS'):
            return Dims(self.cfg, p.ID, p.star_or_index) 
        elif hasattr(p, 'RANK'):
            return Rank(self.cfg, p.ID)
        elif hasattr(p, 'INT'):
            return IntExpr(p.INT)
        else:
            return RValueArray(self.cfg, p.ID, p.star_index_list)

    @_('star_index_expr',
       'star_index_list COMMA star_index_expr')
    def star_index_list(self, p):
        if hasattr(p, 'COMMA'):
            return p.star_index_list + [p.star_index_expr]
        else:
            return [p.star_index_expr]

    @_('ID',
       'COLON')
    def star_or_index(self, p):
        return p[0]

    @_('star_or_index',
       'array_slice')
    def star_index_expr(self, p):
        return p[0]

    def parse(self, arg_string):
        return super().parse(self.lexer.tokenize(arg_string))

if __name__ == '__main__':
    import config
    import sys
    cfg = config.Config()
    parser = BCParser(cfg)
    statement = sys.argv[1]
    ast = parser.parse(statement)
    print(ast)

