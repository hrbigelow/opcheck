from enum import Enum
from sly import Lexer, Parser
from ast_nodes import *

class BCLexer(Lexer):
    # Set of token names.   This is always required
    tokens = { IDENT, QUAL_NM, COMMA, COLON, SQSTR, DQSTR, NUMBER, COMP,
            ASSIGN, ACCUM, LPAREN, RPAREN, LBRACK, RBRACK, PLUS, MINUS, TIMES,
            TRUEDIV, DIVIDE, DIMS, RANGE, RANK, RANDOM, TENSOR, L, DTYPE }

    # String containing ignored characters between tokens
    ignore = ' \t'

    # Regular expression rules for tokens
    DIMS    = 'DIMS'
    RANGE   = 'RANGE'
    RANK    = 'RANK'
    RANDOM  = 'RANDOM'
    TENSOR  = 'TENSOR'
    L       = 'L'
    DTYPE   = r'(FLOAT|INT)'
    QUAL_NM = r'[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+'
    IDENT   = r'[a-zA-Z_][a-zA-Z0-9_]*' 
    COMMA   = r','
    COLON   = r':'
    SQSTR   = "'(?:\\'|[^'])*'"
    DQSTR   = '"(?:\\"|[^"])*"' 
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
    DIVIDE  = r'\/\/'
    TRUEDIV = r'\/'

    def error(self, t):
        print("Illegal character '%s'" % t.value[0])
        self.index += 1

class ParserMode(Enum):
    Constraint = 0
    Statement = 1
    TensorOutput = 2
    TFCall = 3

class BCParser(Parser):
    tokens = BCLexer.tokens
    precedence = (
       ('left', TIMES, DIVIDE, TRUEDIV),
       ('left', PLUS, MINUS),
    )

    def __init__(self):
        self.lexer = BCLexer()

    def set_runtime(self, runtime):
        self.runtime = runtime

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
        return TensorArg(self.runtime, p.IDENT)
    
    @_('shape_test')
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
        # strip leading and trailing quote
        return p[0][1:-1]

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
                f'Could not convert {p.NUMBER} to int or float')

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

    @_('python_literal', 'tensor_arg', 'rank', 'dims_int', 'dims_star',
       'tensor_wrap')
    def bare_tf_call_arg(self, p):
        return p[0]

    @_('shape_expr COMP shape_expr')
    def shape_test(self, p):
        return LogicalOp(p.shape_expr0, p.shape_expr1, p.COMP)

    @_('shape_term',
       'shape_expr expr_op shape_term')
    def shape_expr(self, p):
        if hasattr(p, 'shape_expr'):
            return ArithmeticBinOp(p.shape_expr, p.shape_term, p.expr_op)
        else:
            return p.shape_term
    
    @_('shape_factor',
       'shape_term int_term_op shape_factor')
    def shape_term(self, p):
        if hasattr(p, 'int_term_op'):
            return ArithmeticBinOp(p.shape_term, p.int_term_op, p.shape_factor)
        else:
            return p.shape_factor

    @_('shape',
       'LPAREN shape_expr RPAREN')
    def shape_factor(self, p):
        if hasattr(p, 'shape_expr'):
            return p.shape_expr
        else:
            return p.shape

    @_('PLUS', 'MINUS')
    def expr_op(self, p):
        return p[0]

    @_('TIMES', 'DIVIDE')
    def int_term_op(self, p):
        return p[0]

    @_('TIMES', 'DIVIDE', 'TRUEDIV')
    def term_op(self, p):
        return p[0]


    @_('integer_node', 'rank', 'dims_star', 'dims_int')
    def shape(self, p):
        return p[0]

    @_('number_node')
    def integer_node(self, p):
        if isinstance(p.number_node, FloatExpr):
            raise RuntimeError(f'Expected an IntExpr here')
        return p.number_node

    @_('number')
    def number_node(self, p):
        if isinstance(p.number, int):
            return IntExpr(self.runtime, p.number)
        elif isinstance(p.number, float):
            return FloatExpr(self.runtime, p.number)

    @_('RANK LPAREN tup_name_list RPAREN')
    def rank(self, p):
        return Rank(self.runtime, p.tup_name_list)

    @_('DIMS LPAREN tup_name_list RPAREN LBRACK COLON RBRACK')
    def dims_star(self, p):
        return Dims(self.runtime, DimKind.Star, p.tup_name_list) 

    @_('DIMS LPAREN tup_name_list RPAREN LBRACK number RBRACK')
    def dims_int(self, p):
        if not isinstance(p.number, int):
            raise RuntimeError(
                f'Expected an integer for Int Dims, got {p.number}')
        return Dims(self.runtime, DimKind.Int, p.tup_name_list, p.number)

    @_('DIMS LPAREN tup_name_list RPAREN LBRACK tup_name RBRACK')
    def dims_index(self, p):
        return Dims(self.runtime, DimKind.Index, p.tup_name_list, p.tup_name)

    @_('TENSOR LPAREN static_node RPAREN')
    def tensor_wrap(self, p):
        return TensorWrap(self.runtime, p.static_node)

    @_('dims_star', 'rank')
    def static_node(self, p):
        return p[0]

    @_('tup_name',
       'tup_name_list COMMA tup_name')
    def tup_name_list(self, p):
        if hasattr(p, 'COMMA'):
            p.tup_name_list.append(p.tup_name)
            return p.tup_name_list
        else:
            return [p.tup_name]

    @_('IDENT LBRACK top_index_list RBRACK')
    def lval_array(self, p):
        return LValueArray(self.runtime, p.IDENT, p.top_index_list)

    @_('rval_array', 
       'rand_call',
       'range_array',
       'dims_int',
       'dims_index',
       'number_node')
    def rval_unit(self, p):
        return p[0]

    @_('rval_term',
       'rval_expr expr_op rval_term')
    def rval_expr(self, p):
        if hasattr(p, 'expr_op'):
            return ArrayBinOp(self.runtime, p.rval_expr, p.rval_term, p.expr_op)
        else:
            return p.rval_term

    @_('rval_factor',
       'rval_term term_op rval_factor')
    def rval_term(self, p):
        if hasattr(p, 'term_op'):
            return ArrayBinOp(self.runtime, p.rval_term, p.rval_factor, 
                    p.term_op)
        else:
            return p.rval_factor


    @_('rval_unit',
       'LPAREN rval_expr RPAREN')
    def rval_factor(self, p):
        if hasattr(p, 'rval_expr'):
            return p.rval_expr
        else:
            return p.rval_unit

    @_('array_name LBRACK sub_index_list RBRACK')
    def slice_node(self, p):
        return Slice(self.runtime, p.array_name, p.sub_index_list)

    @_('array_name LBRACK top_index_list RBRACK')
    def rval_array(self, p):
        return RValueArray(self.runtime, p.array_name, p.top_index_list)

    @_('RANDOM LPAREN array_slice COMMA array_slice COMMA DTYPE RPAREN')
    def rand_call(self, p):
        return RandomCall(self.runtime, p.array_slice0, p.array_slice1, p.DTYPE)

    @_('RANGE LBRACK tup_name COMMA tup_name RBRACK')
    def range_array(self, p):
        return RangeExpr(self.runtime, p.tup_name0, p.tup_name1)

    @_('top_index_expr', 
       'top_index_list COMMA top_index_expr')
    def top_index_list(self, p):
        if hasattr(p, 'COMMA'):
            return p.top_index_list + [p.top_index_expr]
        else:
            return [p.top_index_expr]

    @_('tup_name', 'slice_node')
    def top_index_expr(self, p):
        return p[0]

    @_('number_node', 'rank', 'dims_index', 'rval_array')
    def array_slice(self, p):
        return p[0]

    @_('sub_index_expr',
       'sub_index_list COMMA sub_index_expr')
    def sub_index_list(self, p):
        if hasattr(p, 'COMMA'):
            return p.sub_index_list + [p.sub_index_expr]
        else:
            return [p.sub_index_expr]

    @_('COLON', 'tup_name')
    def sub_index_expr(self, p):
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
    import runtime
    import sys
    import json

    rt = runtime.Runtime()
    parser = BCParser()
    parser.set_runtime(rt)
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
        rt.set_ranks(tests['rank'])

    if 'dims' in tests:
        rt.set_dims(tests['dims'])

    # specific requirement for the gather test
    # rt.set_one_dim('coord', 0, rt.tup('elem').rank())

    for ast in asts:
        ast.post_parse_init()

    print(rt)

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

