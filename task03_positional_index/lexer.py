import re
from dataclasses import dataclass

from utils.text import normalize, tokenize

WORD_RE = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*", re.IGNORECASE)
NEAR_RE = re.compile(r"NEAR/(\d+)")
OPERATORS = {"AND", "OR", "NOT"}


class QuerySyntaxError(ValueError):
    pass


@dataclass
class Token:
    kind: str     # WORD, PHRASE, AND, OR, NOT, NEAR, LPAREN, RPAREN
    value: object
    pos: int      # позиция символа в строке запроса


def read_phrase(query, start):
    end = query.find('"', start + 1)
    if end == -1:
        raise QuerySyntaxError(f"незакрытая кавычка в позиции {start}")
    words = tokenize(query[start + 1:end])
    if not words:
        raise QuerySyntaxError(f"пустая фраза в позиции {start}")
    return Token("PHRASE", words, start), end + 1


def read_near(query, start):
    match = NEAR_RE.match(query, start)
    if not match:
        raise QuerySyntaxError(f"после NEAR нужно /k, позиция {start}")
    k = int(match.group(1))
    if k < 1:
        raise QuerySyntaxError(f"в NEAR/k должно быть k >= 1, позиция {start}")
    return Token("NEAR", k, start), match.end()


def tokenize_query(query):
    tokens = []
    i = 0
    while i < len(query):
        ch = query[i]
        if ch.isspace():
            i += 1
        elif ch in "()":
            tokens.append(Token("LPAREN" if ch == "(" else "RPAREN", ch, i))
            i += 1
        elif ch == '"':
            token, i = read_phrase(query, i)
            tokens.append(token)
        else:
            match = WORD_RE.match(query, i)
            if not match:
                raise QuerySyntaxError(f"неожиданный символ {ch!r} в позиции {i}")
            text = match.group()
            if text == "NEAR":
                token, i = read_near(query, i)
                tokens.append(token)
                continue
            if text in OPERATORS:
                tokens.append(Token(text, text, i))
            else:
                tokens.append(Token("WORD", normalize(text), i))
            i = match.end()
    return tokens
