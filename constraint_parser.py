import tensorflow as tf
from sly import Lexer, Parser
from ein_ast import *

class ConsLexer(Lexer):
    tokens = { ID, DIMS, RANK, MIN, MAX, PLUS, MINUS, LPAREN, RPAREN, LBRACK,
            RBRACK, INT, COMP, ASSIGN }
    ignore = ' \t'
    ID = r'[a-zA-Z_][a-zA-Z0-9_]*'
    ID['DIMS'] = DIMS
    ID['RANK'] = RANK
    ID['MIN'] = MIN
    ID['MAX'] = MAX
    PLUS    = r'\+'
    MINUS   = r'\-'
    LPAREN  = r'\('
    RPAREN  = r'\)'
    LBRACK  = r'\['
    RBRACK  = r'\]'
    INT     = r'[0-9]+'
    COMP = r'(>=|>|<=|<|==)'
    ASSIGN = r':='

class ConsParser(Parser):
    tokens = ConsLexer.tokens

    def __init__(self, cfg):
        self.lexer = ConsLexer()
        self.cfg = cfg

    @_('range_constraint', 'logical_op')
    def constraint(self, p):
        return p[0]

    @_('shape_access COMP shape_access',
       'eintup_binop COMP shape_access',
       'eintup COMP shape_access')
    def logical_op(self, p):
        return LogicalOp(p[0], p[2], p[1])

    @_('MIN LPAREN eintup_binop RPAREN ASSIGN shape_access',
       'MAX LPAREN eintup_binop RPAREN ASSIGN shape_access')
    def range_constraint(self, p):
        return RangeConstraint(p.eintup_binop, p[0], p.shape_access)

    @_('shape_access PLUS shape_access',
       'shape_access MINUS shape_access')
    def shape_access(self, p):
        return ArithmeticBinOp(p.shape_access0, p.shape_access1, p[1])

    @_('numeric', 'rank', 'dims', 'dims_access')
    def shape_access(self, p):
        return p[0]

    @_('eintup PLUS eintup',
       'eintup MINUS eintup')
    def eintup_binop(self, p):
        return EinTupBinOp(p[0], p[2], p[1])

    @_('ID')
    def eintup(self, p):
        return EinTup(self.cfg, p.ID)

    @_('INT')
    def numeric(self, p):
        return IntNode(self.cfg, p.INT)

    @_('RANK LPAREN ID RPAREN')
    def rank(self, p):
        return Rank(self.cfg, p.ID)

    @_('DIMS LPAREN ID RPAREN')
    def dims(self, p):
        return Dims(self.cfg, p.ID)

    @_('DIMS LPAREN ID RPAREN LBRACK INT RBRACK')
    def dims_access(self, p):
        return DimsAccess(self.cfg, p.ID, p.INT)

    def parse(self, arg_string):
        return super().parse(self.lexer.tokenize(arg_string))

