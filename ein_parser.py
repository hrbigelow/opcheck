"""
assignment : ndarray = ndarray
           | ndarray = call

expr : call
     | ndarray
     | expr + ndarray
     | expr - ndarray
     | expr / ndarray
     | expr * ndarray

arg : ID 
    | DIMS LPAREN ID RPAREN 
    | RANK LPAREN ID RPAREN




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

    # map of EinArray objects
    arrays = { }

    # the main counter
    slice_tuple = SliceTuple()

    def __new__(clsname):
        cls = super().__new__(clsname)
        return cls

    @classmethod
    def update_shapes(cls, shape_map):
        cls.slice_tuple.reset(shape_map)
        for ary in cls.arrays.values():
            ary.update_shape(shape_map)

    @classmethod
    def maybe_add_array(cls, name, sig):
        if name not in cls.arrays:
            cls.arrays[name] = EinArray(sig)
        return cls.arrays[name]

    @classmethod
    def init_program(cls, program):
        inds = program.get_eintup_names()
        cls.slice_tuple.set_indices(inds)

    @_('ndarray ASSIGN ndarray')
    def assignment(self, p):
        return Assign(p.ndarray0, p.ndarray1)

    @_('ndarray ASSIGN call')
    def assignment(self, p):
        p.ndarray.maybe_convert(p.call.dtype)
        return Assign(p.ndarray, p.call) 

    @_('ID LBRACK index_list RBRACK')
    def ndarray(self, p):
        ary = EinParser.maybe_add_array(p.ID, p.index_list.sig())
        array_slice = ArraySlice(p.ID, ary, p.index_list)
        return array_slice

    @_('ID LPAREN index_list RPAREN',
       'ID LPAREN RPAREN')
    def call(self, p):
        if hasattr(p, 'index_list'):
            return Call(p.ID, p.index_list)
        else:
            return Call(p.ID, IndexList())

    @_('DIMS LPAREN ID RPAREN LBRACK ID RBRACK')
    def size_expr(self, p):
        tup1d = EinTup(EinParser.slice_tuple, p.ID1)
        return SizeExpr(EinParser.slice_tuple, p.ID0, tup1d)

    @_('index_expr')
    def index_list(self, p):
        il = IndexList([p.index_expr])
        return il

    @_('index_list COMMA index_expr')
    def index_list(self, p):
        p.index_list.append(p.index_expr)
        return p.index_list

    @_('ID', 'INT', 'ndarray', 'COLON', 'size_expr')
    def index_expr(self, p):
        if hasattr(p, 'ID'):
            return EinTup(EinParser.slice_tuple, p.ID)
        elif hasattr(p, 'INT'):
            return IntNode(p.INT)
        elif hasattr(p, 'ndarray'):
            return p.ndarray
        elif hasattr(p, 'COLON'):
            return StarNode(EinParser.slice_tuple) 
        elif hasattr(p, 'size_expr'):
            return p.size_expr
                       


if __name__ == '__main__':
    lexer = EinLexer()
    parser = EinParser()

    statements = [
            'index[b,s,c] = randint(0, size("e",c))',
            'params[b,e] = random()',
            'result[b,s] = params[b,index[b,s,:]]'
            ]

    print(statements)
    programs = [ parser.parse(lexer.tokenize(st)) for st in statements ]

    # global settings for shapes
    size_map = { 'b': 3, 'e': 3, 's': 3, 'c': 1 }

    shape_map = { 'b': [2,2,3], 'e': [8,4,3,2], 's': [2,2], 'c': [4] }
    parser.update_shapes(shape_map)

    for program in programs:
        parser.init_program(program)
        while parser.slice_tuple.advance():
            program.evaluate()

    print(parser.arrays['result'].ary)

