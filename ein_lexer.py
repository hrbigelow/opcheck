from sly import Lexer
import sys

class EinLexer(Lexer):
    # Set of token names.   This is always required
    tokens = { ID, LPAREN, RPAREN, LBRACKET, RBRACKET,
            PLUS, MINUS, TIMES, DIVIDE, ASSIGN,
            COMMA }

    # String containing ignored characters between tokens
    ignore = ' \t'

    # Regular expression rules for tokens
    ID      = r'[a-zA-Z_][a-zA-Z0-9_]*'
    COMMA   = r','
    PLUS    = r'\+'
    MINUS   = r'-'
    TIMES   = r'\*'
    DIVIDE  = r'/'
    ASSIGN  = r'='
    LPAREN  = r'\('
    RPAREN  = r'\)'
    LBRACKET  = r'\['
    RBRACKET  = r'\]'

if __name__ == '__main__':
    lexer = EinLexer()
    while True:
        try:
            text = input('ein_lexer > ')
            for tok in lexer.tokenize(text):
                print('type=%r, value=%r' % (tok.type, tok.value))
        except EOFError:
            break

