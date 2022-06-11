"""
BNF Grammar for tuple-einsum notation

The parsing phase builds an AST.
At the end, the parser returns this AST, but must also build
a context object which contains the symbol table.

The symbol table will contain the instantiated np.ndarray objects

Each node of the AST is a derived class with an overloaded 'value()'
function.  Together, the tree builds a 

assignment : ndarray = expr

expr : call
     | ndarray
     | expr + ndarray
     | expr - ndarray
     | expr / ndarray
     | expr * ndarray

call : ID ( expr )

ndarray : ID [ index_list ]

index_list : index
           | ndarray
           | index_list , index
           | index_list , ndarray

index : ID

"""

from ein_lexer import EinLexer
from sly import Parser
import numpy as np

# simple wrapper class
class Tup(object):
    # subranks is { b: 2, a: 3, ... }
    # a dictionary of the ranks of each tuple-ein index
    def __init__(self, subranks):
        self.value = None
        self.slices = {}
        self.rank = 0

        for index_name, subrank in subranks.values():
            self.slices[index_name] = slice(self.rank, self.rank + subrank)
            self.rank += subrank

    def get_value(self, name):
        return self.value[self.slices[name]]

    
class AST(object):
    def __init__(self):
        pass

    def value(self):
        pass

class Index(AST):
    # beg, end denote the position in the main tuple
    def __init__(self, tup, name):
        super().__init__()
        self.tup = tup 
        self.name = name

    def value(self):
        return self.tup.value[self.beg:self.end]

class IndexList(AST):
    def __init__(self):
        super().__init__()
        self.elements = []

    def append(self, index_node):
        self.elements.append(index_node)

    def value(self):
        return sum(self.elements, tuple())

class ArraySlice(AST):
    def __init__(self, array, index_list_node):
        super().__init__()
        self.array = array
        self.indices = index_list_node

    def value(self):
        return self.array[self.indices.value()]


class EinParser(Parser):
    tokens = EinLexer.tokens
    symbol_table = {}

    def __new__(clsname):
        cls = super().__new__(clsname)
        return cls


    @classmethod
    def register(cls, name, obj):
        cls.symbol_table[name] = obj

    """
    # Grammar rules and actions
    @_('ID LPAREN expr RPAREN')
    def call(self, p):
        print(p)
        pass

    """
    @_('ID LBRACKET index_list RBRACKET')
    def ndarray(self, p):
        print('p.ID: ', p.ID)
        print('p.index_list: ', p.index_list)
        self.register(p.ID, np.ndarray((5,2)))
        return 1

    @_('ID')
    def index_list(self, p):
        self.register(p.ID, 
        print(p.index)

    @_('ndarray')
    def index_list(self, p):
        print(p.ndarray)
        return 5

    @_('index_list COMMA ID')
    def index_list(self, p):
        print(p.index_list)
        return 2

    @_('index_list COMMA ndarray')
    def index_list(self, p):
        print(p.ndarray)
        return 3




if __name__ == '__main__':
    lexer = EinLexer()
    parser = EinParser()

    while True:
        try:
            text = input('ein > ')
            result = parser.parse(lexer.tokenize(text))
            print(result)
            print('symbols: ', parser.symbol_table)
        except EOFError:
            break

