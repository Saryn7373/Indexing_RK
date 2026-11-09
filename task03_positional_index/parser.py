"""Рекурсивный спуск. Грамматика (от низкого приоритета к высокому):

query    := EOF | or_expr EOF
or_expr  := and_expr ("OR" and_expr)*
and_expr := not_expr ("AND" not_expr)*
not_expr := "NOT" not_expr | near_expr
near_expr:= primary ("NEAR/k" primary)*
primary  := WORD | PHRASE | "(" or_expr ")"
"""
from dataclasses import dataclass

from task03_positional_index.lexer import QuerySyntaxError, tokenize_query


@dataclass
class Empty:
    pass


@dataclass
class Term:
    word: str


@dataclass
class Phrase:
    words: list


@dataclass
class Near:
    left: object
    right: object
    k: int


@dataclass
class And:
    left: object
    right: object


@dataclass
class Or:
    left: object
    right: object


@dataclass
class Not:
    child: object


POSITIONAL = (Term, Phrase, Near)


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def peek(self):
        return self.tokens[self.i].kind if self.i < len(self.tokens) else "EOF"

    def advance(self):
        token = self.tokens[self.i]
        self.i += 1
        return token

    def error(self, message):
        if self.i < len(self.tokens):
            token = self.tokens[self.i]
            raise QuerySyntaxError(f"{message}: {token.kind} {token.value!r} в позиции {token.pos}")
        raise QuerySyntaxError(f"{message}: неожиданный конец запроса")

    def parse(self):
        if not self.tokens:
            return Empty()
        node = self.parse_or()
        if self.peek() != "EOF":
            self.error("ожидался оператор AND/OR или конец запроса")
        return node

    def parse_or(self):
        node = self.parse_and()
        while self.peek() == "OR":
            self.advance()
            node = Or(node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_not()
        while self.peek() == "AND":
            self.advance()
            node = And(node, self.parse_not())
        return node

    def parse_not(self):
        if self.peek() == "NOT":
            self.advance()
            return Not(self.parse_not())
        return self.parse_near()

    def parse_near(self):
        node = self.parse_primary()
        while self.peek() == "NEAR":
            k = self.advance().value
            right = self.parse_primary()
            if not isinstance(node, POSITIONAL) or not isinstance(right, POSITIONAL):
                raise QuerySyntaxError("операнды NEAR/k должны быть словами или фразами, а не AND/OR/NOT-выражениями")
            node = Near(node, right, k)
        return node

    def parse_primary(self):
        kind = self.peek()
        if kind == "WORD":
            return Term(self.advance().value)
        if kind == "PHRASE":
            words = self.advance().value
            return Term(words[0]) if len(words) == 1 else Phrase(words)
        if kind == "LPAREN":
            self.advance()
            node = self.parse_or()
            if self.peek() != "RPAREN":
                self.error("ожидалась закрывающая скобка")
            self.advance()
            return node
        self.error("ожидалось слово, фраза или '('")


def parse_query(query):
    return Parser(tokenize_query(query)).parse()


def tree_lines(node, indent=0):
    pad = "  " * indent
    if isinstance(node, Empty):
        return [pad + "EMPTY"]
    if isinstance(node, Term):
        return [pad + f"TERM {node.word}"]
    if isinstance(node, Phrase):
        return [pad + 'PHRASE "' + " ".join(node.words) + '"']
    if isinstance(node, Not):
        return [pad + "NOT"] + tree_lines(node.child, indent + 1)
    name = {And: "AND", Or: "OR", Near: "NEAR"}[type(node)]
    if isinstance(node, Near):
        name += f"/{node.k}"
    return [pad + name] + tree_lines(node.left, indent + 1) + tree_lines(node.right, indent + 1)


def show_tree(node):
    return "\n".join(tree_lines(node))
