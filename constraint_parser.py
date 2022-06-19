import tensorflow as tf
from sly import Lexer, Parser
from ein_ast import *

class ConsLexer(Lexer):
    tokens = { ID, DIMS, RANK, PLUS, MINUS, LPAREN, RPAREN, LBRACK, RBRACK,
            INT, COMP }
    ignore = ' \t'
    ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
    ID['DIMS'] = DIMS
    ID['RANK'] = RANK
    PLUS    = r'\+'
    MINUS   = r'\-'
    LPAREN  = r'\('
    RPAREN  = r'\)'
    LBRACK  = r'\['
    RBRACK  = r'\]'
    INT     = r'[0-9]+'
    COMP = r'(<|>|>=|==|<=)'

class ConsParser(Parser):
    tokens = ConsLexer.tokens

    def __init__(self, cfg):
        self.lexer = ConsLexer()
        self.cfg = cfg

    @_('shape_access COMP shape_access')
    def predicate(self, p):
        return LogicalOp(p[0], p[2], p[1])

    @_('shape_access PLUS shape_access',
       'shape_access MINUS shape_access')
    def shape_access(self, p):
        return ShapeAccessBinOp(p.shape_access0, p.shape_access1, p[1])

    @_('numeric', 'rank', 'dims_access')
    def shape_access(self, p):
        return p[0]

    @_('INT')
    def numeric(self, p):
        return IntNode(p.INT)

    @_('RANK LPAREN ID RPAREN')
    def rank(self, p):
        return Rank(self.cfg, p.ID)

    @_('DIMS LPAREN ID RPAREN LBRACK INT RBRACK')
    def dims_access(self, p):
        return DimsAccess(self.cfg, p.ID, p.INT)

    def parse(self, arg_string):
        return super().parse(self.lexer.tokenize(arg_string))
