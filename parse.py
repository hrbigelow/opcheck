from enum import Enum
from sly import Lexer, Parser
from ast_nodes import *

class BCLexer(Lexer):
    # Set of token names.   This is always required
    tokens = { IDENT, QUAL_NM, COMMA, COLON, SQSTR, DQSTR, UFLOAT, UINT,
            ASSIGN, ACCUM, LPAREN, RPAREN, LBRACK, RBRACK, PLUS, MINUS, TIMES,
            MODULO, TRUEDIV, TRUNCDIV, CEILDIV, DIMS, IN, RANGE, RANK, RANDOM,
            TENSOR, FLAT, L, DTYPE }

    # String containing ignored characters between tokens
    ignore = ' \t'

    # Regular expression rules for tokens
    DIMS    = 'DIMS'
    RANGE   = 'RANGE'
    RANK    = 'RANK'
    RANDOM  = 'RANDOM'
    TENSOR  = 'TENSOR'
    FLAT    = 'FLAT'
    DTYPE   = r'(FLOAT|INT)'
    IN      = 'IN'
    L       = 'L'
    QUAL_NM = r'[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+'
    IDENT   = r'[a-zA-Z_][a-zA-Z0-9_]*' 
    COMMA   = r','
    COLON   = r':'
    SQSTR   = r"'(?:\\'|[^'])*'"
    DQSTR   = r'"(?:\\"|[^"])*"' 
    UFLOAT  = r'[0-9]+(\.[0-9]+)'
    UINT    = r'[0-9]+' 
    # COMP    = r'(>=|>|<=|<|==)'
    ASSIGN  = r'='
    ACCUM   = r'\+='
    LPAREN  = r'\('
    RPAREN  = r'\)'
    LBRACK  = r'\['
    RBRACK  = r'\]'
    PLUS    = r'\+'
    MINUS   = r'\-'
    TIMES   = r'\*'
    MODULO  = r'%'
    CEILDIV = r'\/\/\^'
    TRUNCDIV= r'\/\/'
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
       ('left', PLUS, MINUS),
       ('left', TIMES, TRUNCDIV, TRUEDIV, CEILDIV),
       ('right', UMINUS)
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

    @_('rank_constraint', 'dims_constraint')
    def constraint(self, p):
        return p[0]
    
    @_('rank_range', 'rank_equals')
    def rank_constraint(self, p):
        return p[0]

    @_('dims_range', 'dims_equals')
    def dims_constraint(self, p):
        return p[0]

    @_('RANK LPAREN tup_name RPAREN IN closed_interval')
    def rank_range(self, p):
        tup = self.runtime.tup(p.tup_name)
        tup.add_rank_expr(p.closed_interval)
        return None

    @_('RANK LPAREN tup_name RPAREN ASSIGN rcons_expr')
    def rank_equals(self, p):
        tup = self.runtime.tup(p.tup_name)
        tup.add_rank_expr(p.rcons_expr)
        return None

    @_('DIMS LPAREN tup_name RPAREN IN closed_interval')
    def dims_range(self, p):
        lo, hi = p.closed_interval
        tup = self.runtime.tup(p.tup_name)
        rc = RangeConstraint(lo, hi, tup)
        rc.tup.add_gen_expr(rc)
        return None

    @_('DIMS LPAREN tup_name RPAREN ASSIGN dcons_expr')
    def dims_equals(self, p):
        tup = self.runtime.tup(p.tup_name)
        tup.add_gen_expr(p.dcons_expr)
        return None

    @_('LBRACK unsigned_int COMMA unsigned_int RBRACK')
    def closed_interval(self, p):
        return p.unsigned_int0, p.unsigned_int1

    @_('rcons_term',
       'rcons_expr expr_op rcons_term')
    def rcons_expr(self, p):
        if hasattr(p, 'rcons_expr'):
            return ArithmeticBinOp(p.rcons_expr, p.rcons_term, p.expr_op)
        else:
            return p.rcons_term
    
    @_('rcons_factor',
       'rcons_term int_term_op rcons_factor')
    def rcons_term(self, p):
        if hasattr(p, 'int_term_op'):
            return ArithmeticBinOp(p.rcons_term, p.rcons_factor, p.int_term_op)
        else:
            return p.rcons_factor

    @_('integer_node',
       'rank_cons',
       'LPAREN rcons_expr RPAREN')
    def rcons_factor(self, p):
        if hasattr(p, 'rcons_expr'):
            return p.rcons_expr
        else:
            return p[0]

    @_('dcons_term',
       'dcons_expr expr_op dcons_term')
    def dcons_expr(self, p):
        if hasattr(p, 'dcons_expr'):
            return ArithmeticBinOp(p.dcons_expr, p.dcons_term, p.expr_op)
        else:
            return p.dcons_term
    
    @_('dcons_factor',
       'dcons_term int_term_op dcons_factor')
    def dcons_term(self, p):
        if hasattr(p, 'int_term_op'):
            return ArithmeticBinOp(p.dcons_term, p.dcons_factor, p.int_term_op)
        else:
            return p.dcons_factor

    @_('integer_node',
       'rank_cons',
       'dims_cons',
       'LPAREN dcons_expr RPAREN')
    def dcons_factor(self, p):
        if hasattr(p, 'dcons_expr'):
            return p.dcons_expr
        else:
            return p[0] 

    @_('DIMS LPAREN tup_name RPAREN')
    def dims_cons(self, p):
        return DimsConstraint(self.runtime, p.tup_name)

    @_('RANK LPAREN tup_name RPAREN')
    def rank_cons(self, p):
        return RankConstraint(self.runtime, p.tup_name)

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

    @_('UFLOAT')
    def unsigned_float(self, p):
        return float(p.UFLOAT)

    @_('UINT')
    def unsigned_int(self, p):
        return int(p.UINT)

    @_('MINUS unsigned_float %prec UMINUS',
       'unsigned_float')
    def float(self, p):
        if hasattr(p, 'MINUS'):
            return - p.unsigned_float
        else:
            return p.unsigned_float

    @_('MINUS unsigned_int %prec UMINUS',
       'unsigned_int')
    def integer(self, p):
        if hasattr(p, 'MINUS'):
            return - p.unsigned_int
        else:
            return p.unsigned_int

    @_('integer', 'float')
    def number(self, p):
        return p[0]

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

    @_('python_literal', 'tensor_arg', 'rank', 'dims_star', 'tensor_wrap')
    def bare_tf_call_arg(self, p):
        return p[0]

    @_('PLUS', 'MINUS')
    def expr_op(self, p):
        return p[0]

    @_('TIMES', 'TRUNCDIV', 'CEILDIV', 'MODULO')
    def int_term_op(self, p):
        return p[0]

    @_('TIMES', 'TRUNCDIV', 'CEILDIV', 'TRUEDIV')
    def term_op(self, p):
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
        return RankExpr(self.runtime, p.tup_name_list)

    @_('DIMS LPAREN tup_name_list RPAREN LBRACK tup_name RBRACK')
    def dims_index(self, p):
        return Dims(self.runtime, DimKind.Index, p.tup_name_list, p.tup_name)

    @_('DIMS LPAREN tup_name_list RPAREN')
    def dims_slice(self, p):
        return DimsSlice(self.runtime, p.tup_name_list)

    @_('DIMS LPAREN tup_name_list RPAREN')
    def dims_star(self, p):
        return Dims(self.runtime, DimKind.Star, p.tup_name_list) 

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
    def array_slice(self, p):
        return ArraySlice(self.runtime, p.array_name, p.sub_index_list)

    @_('array_name LBRACK top_index_list RBRACK')
    def rval_array(self, p):
        return RValueArray(self.runtime, p.array_name, p.top_index_list)

    @_('RANDOM LPAREN rand_arg COMMA rand_arg COMMA DTYPE RPAREN')
    def rand_call(self, p):
        return RandomCall(self.runtime, p.rand_arg0, p.rand_arg1, p.DTYPE)

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

    @_('tup_expr')
    def top_index_expr(self, p):
        return p[0]

    @_('number_node', 'rank', 'dims_index', 'rval_array')
    def rand_arg(self, p):
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

    def maybe_convert_eintup(self, item):
        if isinstance(item, EinTup):
            return EinTupSlice(self.runtime, item.name)
        else:
            return item

    @_('tup_term',
       'tup_expr expr_op tup_term')
    def tup_expr(self, p):
        if hasattr(p, 'expr_op'):
            tup_expr = self.maybe_convert_eintup(p.tup_expr)
            tup_term = self.maybe_convert_eintup(p.tup_term)
            return SliceBinOp(self.runtime, tup_expr, tup_term, p.expr_op)
        else:
            return p.tup_term

    @_('tup_factor',
       'tup_term int_term_op tup_factor')
    def tup_term(self, p):
        if hasattr(p, 'int_term_op'):
            tup_term = self.maybe_convert_eintup(p.tup_term)
            tup_factor = self.maybe_convert_eintup(p.tup_factor)
            return SliceBinOp(self.runtime, tup_term, tup_factor, p.int_term_op)
        else:
            return p.tup_factor

    @_('tup_name',
       'unsigned_int',
       'dims_slice',
       'rank',
       'array_slice',
       'flatten_slice',
       'LPAREN tup_expr RPAREN')
    def tup_factor(self, p):
        if hasattr(p, 'LPAREN'):
            return p.tup_expr
        elif hasattr(p, 'unsigned_int'):
            return IntSlice(self.runtime, p.unsigned_int)
        elif hasattr(p, 'dims_slice'):
            return p.dims_slice
        elif hasattr(p, 'rank'):
            return RankSlice(p.rank)
        elif hasattr(p, 'array_slice'):
            return p.array_slice
        elif hasattr(p, 'tup_name'):
            return self.runtime.maybe_add_tup(p.tup_name)
        elif hasattr(p, 'flatten_slice'):
            return p.flatten_slice
        else:
            raise RuntimeError(f'Parsing Error for rule tup_factor')

    @_('FLAT LPAREN top_index_list RPAREN')
    def flatten_slice(self, p):
        slice_list = [ self.maybe_convert_eintup(item) 
                for item in p.top_index_list ]
        return FlattenSlice(slice_list)

    @_('IDENT')
    def tup_name(self, p):
        return p.IDENT

    def parse(self, arg_string):
        return super().parse(self.lexer.tokenize(arg_string))

