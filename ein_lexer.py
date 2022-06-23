from sly import Lexer

class EinLexer(Lexer):
    # Set of token names.   This is always required
    tokens = { ID, LPAREN, RPAREN, LBRACK, RBRACK, PLUS, MINUS, TIMES, DIVIDE,
            INT, ASSIGN, ACCUM, COMMA, COLON, DIMS }

    # String containing ignored characters between tokens
    ignore = ' \t'

    # Regular expression rules for tokens
    ID      = r'[a-zA-Z_][a-zA-Z0-9_]*'
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

    ID['DIMS'] = DIMS

if __name__ == '__main__':
    lexer = EinLexer()
    while True:
        try:
            text = input('ein_lexer > ')
            for tok in lexer.tokenize(text):
                print('type=%r, value=%r' % (tok.type, tok.value))
        except EOFError:
            break
