import cStringIO
import tokenize


def remove_comments(src):
    """
    This reads tokens using tokenize.generate_tokens and recombines them
    using tokenize.untokenize, and skipping comment/docstring tokens in between
    """
    f = cStringIO.StringIO(src)

    class SkipException(Exception):
        pass
    processed_tokens = []
    # go through all the tokens and try to skip comments
    for tok in tokenize.generate_tokens(f.readline):
        t_type, t_string, t_srow_scol, t_erow_ecol, t_line = tok
        try:
            if t_type == tokenize.COMMENT:
                raise SkipException()
        except SkipException:
            pass
        else:
            processed_tokens.append(tok)

    return tokenize.untokenize(processed_tokens)


class Scanner(object):
    """
    Class to maintain tokenized string.
    """
    def __init__(self, string):
        src = cStringIO.StringIO(string).readline
        self.tokens = tokenize.generate_tokens(src)
        self.cur = None

    def next(self):
        while True:
            self.cur = self.tokens.next()
            if self.cur[0] not in [54, tokenize.NEWLINE] and self.cur[1] != ' ':
                break
        return self.cur

    def check_token(self, token, message):
        if type(token) == int:
            if self.cur[0] != token:
                raise SyntaxError(message + '; token: %s' % str(self.cur))
        else:
            if self.cur[1] != token:
                raise SyntaxError(message + '; token: %s' % str(self.cur))
