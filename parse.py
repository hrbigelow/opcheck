from enum import Enum
from sly import Lexer, Parser
from ast_nodes import *

class BCLexer(Lexer):
    # Set of token names.   This is always required
    tokens = { IDENT, QUAL_NM, COMMA, COLON, SQSTR, DQSTR, DOT, INT, NUMBER, COMP,
            ASSIGN, ACCUM, LPAREN, RPAREN, LBRACK, RBRACK, PLUS, MINUS, TIMES,
            TRUEDIV, DIVIDE, DIMS, RANGE, RANK, RANDOM, MIN, MAX, L, DTYPE }

    # String containing ignored characters between tokens
    ignore = ' \t'

    # Regular expression rules for tokens
    DIMS    = 'DIMS'
    RANGE   = 'RANGE'
    RANK    = 'RANK'
    RANDOM  = 'RANDOM'
    MIN     = 'MIN'
    MAX     = 'MAX'
    L       = 'L'
    DTYPE   = r'(FLOAT|INT)'
    QUAL_NM = r'[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+'
    IDENT   = r'[a-zA-Z_][a-zA-Z0-9_]*' 
    COMMA   = r','
    COLON   = r':'
    SQSTR   = "'(?:\\'|[^'])*'"
    DQSTR   = '"(?:\\"|[^"])*"' 
    DOT     = r'\.'
    INT     = r'[0-9]+'
    NUMBER  = r'[\-\+]?[0-9]+(\.[0-9]+)?'
    COMP    = r'(>=|>|<=|<|==)'
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

class ParserMode(Enum):
    Constraint = 0
    Statement = 1
    TensorOutput = 2
    TFCall = 3

class BCParser(Parser):
    tokens = BCLexer.tokens
    precedence = (
       ('left', PLUS, MINUS),
       ('left', TIMES, DIVIDE),
    )

    def __init__(self):
        self.lexer = BCLexer()

    def set_config(self, cfg):
        self.cfg = cfg

    def set_constraint_mode(self):
        self.mode = ParserMode.Constraint

    def set_statement_mode(self):
        self.mode = ParserMode.Statement

    def set_output_mode(self):
        self.mode = ParserMode.TensorOutput

    def set_tfcall_mode(self):
        self.mode = ParserMode.TFCall

    @_('tensor_list', 'constraint', 'statement', 'tf_call')
    def toplevel(self, p):
        if self.mode == ParserMode.TensorOutput and hasattr(p, 'tensor_list'):
            return p.tensor_list
        elif self.mode == ParserMode.Constraint and hasattr(p, 'constraint'):
            return p.constraint
        elif self.mode == ParserMode.Statement and hasattr(p, 'statement'):
            return p.statement
        elif self.mode == ParserMode.TFCall and hasattr(p, 'tf_call'):
            return p.tf_call
        else:
            raise RuntimeError('Parse Error at top level')
    
    @_('tensor_arg',
       'tensor_list COMMA tensor_arg')
    def tensor_list(self, p):
        if hasattr(p, 'COMMA'):
            p.tensor_list.append(p.tensor_arg)
            return p.tensor_list
        else:
            return [p.tensor_arg]

    @_('IDENT')
    def tensor_arg(self, p):
        return TensorArg(self.cfg, p.IDENT)
    
    @_('shape_test', 'tup_limit')
    def constraint(self, p):
        return p[0]

    @_('lval_array ASSIGN rval_expr',
       'lval_array ACCUM rval_expr')
    def statement(self, p):
        do_accum = hasattr(p, 'ACCUM')
        return Assign(p.lval_array, p.rval_expr, do_accum)

    @_('qualified_name LPAREN tf_call_list RPAREN')
    def tf_call(self, p):
        return TFCall(p.qualified_name, p.tf_call_list)

    @_('QUAL_NM', 'IDENT')
    def qualified_name(self, p):
        return p[0]

    @_('L LPAREN python_value RPAREN')
    def python_literal(self, p):
        return p.python_value

    @_('string_literal', 'number')
    def python_value(self, p):
        return p[0]

    @_('SQSTR', 'DQSTR')
    def string_literal(self, p):
        return p[0] 

    @_('NUMBER')
    def number(self, p):
        try:
            return int(p.NUMBER)
        except ValueError:
            pass
        try:
            return float(p.NUMBER)
        except ValueError:
            raise RuntimeError(
                f'Could not convert {p.INT} to int or float')

    @_('tf_call_arg',
       'tf_call_list COMMA tf_call_arg')
    def tf_call_list(self, p):
        if hasattr(p, 'COMMA'):
            p.tf_call_list.append(p.tf_call_arg)
            return p.tf_call_list
        else:
            return [p.tf_call_arg]

    @_('named_tf_call_arg', 'bare_tf_call_arg')
    def tf_call_arg(self, p):
        return p[0]

    @_('IDENT ASSIGN bare_tf_call_arg')
    def named_tf_call_arg(self, p):
        return (p.IDENT, p.bare_tf_call_arg)

    @_('python_literal', 'tensor_arg')
    def bare_tf_call_arg(self, p):
        return p[0]

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

    @_('IDENT LBRACK index_list RBRACK')
    def lval_array(self, p):
        return LValueArray(self.cfg, p.IDENT, p.index_list)

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

    @_('IDENT')
    def array_name(self, p):
        return p.IDENT

    @_('IDENT')
    def tup_name(self, p):
        return p.IDENT

    def parse(self, arg_string):
        return super().parse(self.lexer.tokenize(arg_string))

if __name__ == '__main__':
    import config
    import sys
    import json

    cfg = config.Config()
    parser = BCParser(cfg)
    with open('ops/tests.json', 'r') as fp:
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

    if 'rank' in tests:
        cfg.set_ranks(tests['rank'])

    if 'dims' in tests:
        cfg.set_dims(tests['dims'])

    # specific requirement for the gather test
    # cfg.set_one_dim('coord', 0, cfg.tup('elem').rank())

    for ast in asts:
        ast.post_parse_init()

    print(cfg)

    for ast in asts:
        print(f'Evaluating {ast}')
        ast.evaluate()


    """
    print('Parsing constraints')
    parser.set_constraint_mode()
    for con in tests['constraints']:
        ast = parser.parse(con)
        print(f'Constraint: {con}\nParsed as:  {ast}\n')
    """

