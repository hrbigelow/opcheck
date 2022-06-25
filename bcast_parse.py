from enum import Enum
from sly import Lexer, Parser
from bcast_ast import *

class BCLexer(Lexer):
    # Set of token names.   This is always required
    tokens = { ID, LPAREN, RPAREN, LBRACK, RBRACK, PLUS, MINUS, TIMES, DIVIDE,
            INT, ASSIGN, ACCUM, COMMA, COLON, DIMS, RANGE, RANK, RANDOM, MIN,
            MAX, DTYPE }

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
    MIN     = 'MIN'
    MAX     = 'MAX'
    DTYPE   = r'(FLOAT|INT)'

class ParserMode(Enum):
    Constraint = 0
    Statement = 1
    Argument = 2

class BCParser(Parser):
    tokens = BCLexer.tokens
    precedence = (
       ('left', PLUS, MINUS),
       ('left', TIMES, DIVIDE),
    )

    def __init__(self, cfg):
        self.lexer = BCLexer()
        self.cfg = cfg 

    def set_constraint_mode(self):
        self.mode = ParserMode.Constraint

    def set_statement_mode(self):
        self.mode = ParserMode.Statement

    def set_argument_mode(self):
        self.mode = ParserMode.Argument

    @_('constraint', 'statement', 'argument')
    def toplevel(self, p):
        if self.mode == ParserMode.Constraint and hasattr(p, 'constraint'):
            return p.constraint
        elif self.mode == ParserMode.Statement and hasattr(p, 'statement'):
            return p.statement
        elif self.mode == ParserMode.Argument and hasattr(p, 'argument'):
            return p.argument
        else:
            raise RuntimeError('Parse Error at top level')

    @_('ID')
    def argument(self, p):
        return TensorArg(self.cfg, p.ID)
    
    @_('shape_test',
       'tup_limit')
    def constraint(self, p):
        pass

    @_('shape_expr COMP shape_expr')
    def shape_test(self, p):
        return LogicalOp(p.shape_expr0, p.shape_expr1, p.COMP)

    @_('limit_expr ASSIGN shape')
    def tup_limit(self, p):
        pass

    @_('shape',
       'shape OP shape')
    def shape_expr(self, p):
        if hasattr(p, 'OP'):
            return ArithmeticBinOp(p.shape0, p.shape1, p[1])
        else:
            return p.shape

    @_('integer', 'rank', 'dims_colon', 'dims_int')
    def shape(self, p):
        return p[0]

    @_('INT')
    def integer(self, p):
        return IntExpr(p.INT)

    @_('RANK LPAREN ID RPAREN')
    def rank(self, p):
        return Rank(self.cfg, p.ID)

    @_('DIMS LPAREN ID RPAREN LBRACK COLON RBRACK')
    def dims_colon(self, p):
        return Dims(self.cfg, p.ID, p.COLON) 

    @_('DIMS LPAREN ID RPAREN LBRACK INT RBRACK')
    def dims_int(self, p):
        return Dims(self.cfg, p.ID, p.INT)

    @_('limit_type LPAREN simple_index_expr RPAREN ASSIGN shape_expr')
    def range_constraint(self, p):
        return RangeConstraint(p.simple_index_expr, p.limit_type, p.shape_expr)

    @_('MIN', 'MAX')
    def limit_type(self, p):
        return p[0]

    @_('ID',
       'simple_index_expr OP ID')
    def simple_index_expr(self, p):
        pass

# ----------------------------------
    @_('lval_array ASSIGN rval_expr',
       'lval_array ACCUM rval_expr')
    def statement(self, p):
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

