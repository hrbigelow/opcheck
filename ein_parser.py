"""
assignment : slice = expr

expr : call
     | slice
     | expr + slice
     | expr - slice
     | expr / slice
     | expr * slice

arg : ID 
    | DIMS LPAREN ID RPAREN 
    | RANK LPAREN ID RPAREN



scalar : call
       | slice



"""
from ein_lexer import EinLexer
from ein_array import *
from ein_ast import *
from sly import Parser
import numpy as np


class EinParser(Parser):
    tokens = EinLexer.tokens
    precedence = (
       ('left', PLUS, MINUS),
       ('left', TIMES, DIVIDE),
    )

    def __init__(self):
        self.lexer = EinLexer()
        # map of EinArray objects
        self.arrays = { }
        # the main counter
        self.slice_tuple = SliceTuple()

    def update_shapes(self, shape_map):
        self.slice_tuple.reset(shape_map)
        for ary in self.arrays.values():
            ary.update_shape(shape_map)

    def maybe_add_array(self, name, sig):
        if name not in self.arrays:
            self.arrays[name] = EinArray(sig)
        return self.arrays[name]

    def init_statement(self, ast):
        inds = ast.get_eintup_names()
        self.slice_tuple.set_indices(inds)
        ast.reset()

    @_('slice ASSIGN slice')
    def assignment(self, p):
        return Assign(p.slice0, p.slice1)

    @_('slice ASSIGN call')
    def assignment(self, p):
        p.slice.maybe_convert(p.call.dtype)
        return Assign(p.slice, p.call) 

    @_('ID LBRACK index_list RBRACK')
    def slice(self, p):
        ary = self.maybe_add_array(p.ID, p.index_list.sig())
        array_slice = ArraySlice(p.ID, ary, p.index_list)
        return array_slice

    @_('slice PLUS slice',
       'slice MINUS slice',
       'slice TIMES slice',
       'slice DIVIDE slice')
    def slice(self, p):
        return SliceBinOp(p.slice0, p.slice1, p[1])

    @_('ID LPAREN index_list RPAREN',
       'ID LPAREN RPAREN')
    def call(self, p):
        if hasattr(p, 'index_list'):
            return Call(p.ID, p.index_list)
        else:
            return Call(p.ID, IndexList())

    @_('DIMS LPAREN ID RPAREN LBRACK ID RBRACK')
    def size_expr(self, p):
        tup1d = EinTup(self.slice_tuple, p.ID1)
        return SizeExpr(self.slice_tuple, p.ID0, tup1d)

    @_('index_expr')
    def index_list(self, p):
        il = IndexList([p.index_expr])
        return il

    @_('index_list COMMA index_expr')
    def index_list(self, p):
        p.index_list.append(p.index_expr)
        return p.index_list

    @_('ID', 'INT', 'slice', 'COLON', 'size_expr')
    def index_expr(self, p):
        if hasattr(p, 'ID'):
            return EinTup(self.slice_tuple, p.ID)
        elif hasattr(p, 'INT'):
            return IntNode(p.INT)
        elif hasattr(p, 'slice'):
            return p.slice
        elif hasattr(p, 'COLON'):
            return StarNode(self.slice_tuple) 
        elif hasattr(p, 'size_expr'):
            return p.size_expr
                       
    def parse(self, arg_string):
        return super().parse(self.lexer.tokenize(arg_string))


if __name__ == '__main__':
    parser = EinParser()

    statements = [
            'index[b,s,c] = RANDINT(0, DIMS(e)[c])',
            'params[b,e] = RANDOM()',
            'result[b,s] = params[b,index[b,s,:]]'
            ]

    print(statements)
    programs = [ parser.parse(st) for st in statements ]

    # global settings for shapes
    size_map = { 'b': 3, 'e': 3, 's': 3, 'c': 1 }

    shape_map = { 'b': [2,2,3], 'e': [8,4,3,2], 's': [2,2], 'c': [4] }
    parser.update_shapes(shape_map)

    for stmt in statements:
        ast = parser.parse(stmt)
        parser.init_statement(ast)
        while parser.slice_tuple.advance():
            ast.evaluate()

    print(parser.arrays['result'].ary)

