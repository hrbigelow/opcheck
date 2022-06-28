from enum import Enum
from sly import Lexer, Parser
from ast_nodes import *

class BCLexer(Lexer):
    # Set of token names.   This is always required
    tokens = { ID, LPAREN, RPAREN, LBRACK, RBRACK, PLUS, MINUS, TIMES, DIVIDE,
            TRUEDIV, INT, COMP, ASSIGN, ACCUM, COMMA, COLON, DIMS, RANGE, RANK,
            RANDOM, MIN, MAX, DTYPE }

    # String containing ignored characters between tokens
    ignore = ' \t'

    # Regular expression rules for tokens
    ID    = r'[a-z]+'
    COMMA   = r','
    COLON   = r':'
    INT     = r'[0-9]+'
    COMP = r'(>=|>|<=|<|==)'
    ASSIGN  = r'='
    ACCUM   = r'\+='
    LPAREN  = r'\('
    RPAREN  = r'\)'
    LBRACK  = r'\['
    RBRACK  = r'\]'
    PLUS    = r'\+'
    MINUS   = r'\-'
    TIMES   = r'\*'
    TRUEDIV = r'\/\/'
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

    @_('argument', 'constraint', 'statement')
    def toplevel(self, p):
        if self.mode == ParserMode.Argument and hasattr(p, 'argument'):
            return p.argument
        elif self.mode == ParserMode.Constraint and hasattr(p, 'constraint'):
            return p.constraint
        elif self.mode == ParserMode.Statement and hasattr(p, 'statement'):
            return p.statement
        else:
            raise RuntimeError('Parse Error at top level')

    @_('ID')
    def argument(self, p):
        return TensorArg(self.cfg, p.ID)
    
    @_('shape_test', 'tup_limit')
    def constraint(self, p):
        return p[0]

    @_('lval_array ASSIGN rval_expr',
       'lval_array ACCUM rval_expr')
    def statement(self, p):
        do_accum = hasattr(p, 'ACCUM')
        return Assign(p.lval_array, p.rval_expr, do_accum)

    @_('shape_expr COMP shape_expr')
    def shape_test(self, p):
        return LogicalOp(p.shape_expr0, p.shape_expr1, p.COMP)

    @_('limit_type LPAREN tup_expr RPAREN ASSIGN shape_expr')
    def tup_limit(self, p):
        return RangeConstraint(p.tup_expr, p.limit_type, p.shape_expr)

    @_('MIN', 'MAX')
    def limit_type(self, p):
        return p[0]

    @_('shape',
       'shape_expr arith_five_op shape')
    def shape_expr(self, p):
        if hasattr(p, 'arith_five_op'):
            return ArithmeticBinOp(p.shape_expr, p.shape, p.arith_five_op)
        else:
            return p.shape

    @_('PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'TRUEDIV')
    def arith_five_op(self, p):
        return p[0]

    @_('integer', 'rank', 'dims_star', 'dims_int')
    def shape(self, p):
        return p[0]

    @_('INT')
    def integer(self, p):
        return IntExpr(self.cfg, p.INT)

    @_('RANK LPAREN tup_name RPAREN')
    def rank(self, p):
        return Rank(self.cfg, p.tup_name)

    @_('DIMS LPAREN tup_name RPAREN LBRACK COLON RBRACK')
    def dims_star(self, p):
        return Dims(self.cfg, p.tup_name, DimKind.Star) 

    @_('DIMS LPAREN tup_name RPAREN LBRACK INT RBRACK')
    def dims_int(self, p):
        return Dims(self.cfg, p.tup_name, DimKind.Int, p.INT)

    @_('DIMS LPAREN tup_name RPAREN LBRACK tup_name RBRACK')
    def dims_index(self, p):
        return Dims(self.cfg, p.tup_name0, DimKind.Index, p.tup_name1)

    @_('tup_name',
       'tup_expr arith_five_op tup_name')
    def tup_expr(self, p):
        if hasattr(p, 'arith_five_op'):
            return ArithmeticBinOp(p.tup_expr, p.tup_name, p.arith_five_op)
        else:
            return p.tup_name

    @_('ID LBRACK index_list RBRACK')
    def lval_array(self, p):
        return LValueArray(self.cfg, p.ID, p.index_list)

    @_('rval_array', 
       'rand_call',
       'range_array',
       'dims_int',
       'dims_index',
       'integer')
    def rval_unit(self, p):
        return p[0]

    @_('rval_unit',
       'rval_expr arith_five_op rval_unit')
    def rval_expr(self, p):
        if hasattr(p, 'arith_five_op'):
            return ArrayBinOp(self.cfg, p.rval_expr, p.rval_unit,
                    p.arith_five_op)
        else:
            return p.rval_unit

    @_('array_name LBRACK star_index_list RBRACK')
    def slice_node(self, p):
        return Slice(self.cfg, p.array_name, p.star_index_list)

    @_('array_name LBRACK nested_index_list RBRACK')
    def rval_array(self, p):
        return RValueArray(self.cfg, p.array_name, p.nested_index_list)

    @_('RANDOM LPAREN array_slice COMMA array_slice COMMA DTYPE RPAREN')
    def rand_call(self, p):
        return RandomCall(self.cfg, p.array_slice0, p.array_slice1, p.DTYPE)

    @_('RANGE LBRACK tup_name COMMA tup_name RBRACK')
    def range_array(self, p):
        return RangeExpr(self.cfg, p.tup_name0, p.tup_name1)

    @_('index_expr', 
       'index_list COMMA index_expr')
    def index_list(self, p):
        if hasattr(p, 'COMMA'):
            return p.index_list + [p.index_expr]
        else:
            return [p.index_expr]

    @_('tup_name', 'array_slice')
    def index_expr(self, p):
        return p[0]

    @_('integer', 'rank', 'dims_index', 'rval_array')
    def array_slice(self, p):
        return p[0]

    @_('star_index_expr',
       'star_index_list COMMA star_index_expr')
    def star_index_list(self, p):
        if hasattr(p, 'COMMA'):
            return p.star_index_list + [p.star_index_expr]
        else:
            return [p.star_index_expr]

    @_('COLON', 'tup_name')
    def star_index_expr(self, p):
        return p[0]

    @_('nested_index_expr',
       'nested_index_list COMMA nested_index_expr')
    def nested_index_list(self, p):
        if hasattr(p, 'COMMA'):
            return p.nested_index_list + [p.nested_index_expr]
        else:
            return [p.nested_index_expr]

    @_('tup_name', 'slice_node')
    def nested_index_expr(self, p):
        return p[0]

    @_('ID')
    def array_name(self, p):
        return p.ID

    @_('ID')
    def tup_name(self, p):
        return p.ID

    def parse(self, arg_string):
        return super().parse(self.lexer.tokenize(arg_string))

if __name__ == '__main__':
    import config
    import sys
    import json

    cfg = config.Config(5, 20)
    parser = BCParser(cfg)
    with open('parse_tests.json', 'r') as fp:
        all_tests = json.load(fp)

    test_string = sys.argv[1]
    tests = all_tests[test_string]

    print('Parsing statements')
    asts = []
    parser.set_statement_mode()
    for st in tests['statements']:
        ast = parser.parse(st)
        print(f'Statement: {st}\nParsed as: {ast}\n')
        asts.append(ast)

    cfg.set_dims(tests['rank'])
    # cfg.set_one_dim('', 0, cfg.rank('s'))
    for ast in asts:
        print(f'Evaluating {ast}')
        ast.prepare()
        ast.evaluate()

    print(cfg)

    """
    print('Parsing constraints')
    parser.set_constraint_mode()
    for con in tests['constraints']:
        ast = parser.parse(con)
        print(f'Constraint: {con}\nParsed as:  {ast}\n')
    """

