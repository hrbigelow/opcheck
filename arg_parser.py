from sly import Lexer, Parser
from ein_ast import *

class ArgLexer(Lexer):
    tokens = { ID, DIMS, RANK, LPAREN, RPAREN, LBRACK, RBRACK, INT }
    ignore = ' \t'
    ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
    ID['DIMS'] = DIMS
    ID['RANK'] = RANK
    LPAREN  = r'\('
    RPAREN  = r'\)'
    LBRACK  = r'\['
    RBRACK  = r'\]'
    INT     = r'[0-9]+'

class ArgParser(Parser):
    tokens = ArgLexer.tokens

    def __init__(self, cfg):
        self.lexer = ArgLexer()
        self.cfg = cfg

    @_('tensor_arg', 'rank', 'dims_access')
    def arg(self, p):
        return p[0]

    @_('ID')
    def tensor_arg(self, p):
        return TensorArg(self.cfg, p.ID)

    @_('RANK LPAREN ID RPAREN')
    def rank(self, p):
        return Rank(self.cfg, p.ID)

    @_('DIMS LPAREN ID RPAREN LBRACK INT RBRACK')
    def dims_access(self, p):
        return DimsAccess(self.cfg, p.ID, p.INT)

    def parse(self, arg_string):
        return super().parse(self.lexer.tokenize(arg_string))

