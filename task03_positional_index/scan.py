"""Эталон: последовательный просмотр текстов без индекса."""
from task03_positional_index.parser import And, Empty, Near, Not, Or, Phrase, Term, parse_query
from utils.text import tokenize


def find_spans(node, tokens):
    if isinstance(node, Term):
        return [(i, i) for i, t in enumerate(tokens) if t == node.word]
    if isinstance(node, Phrase):
        n = len(node.words)
        return [(i, i + n - 1) for i in range(len(tokens) - n + 1) if tokens[i:i + n] == node.words]
    left, right = find_spans(node.left, tokens), find_spans(node.right, tokens)
    return sorted({(s1, e2) for s1, e1 in left for s2, e2 in right if 0 < s2 - e1 <= node.k})


def matches(node, tokens):
    if isinstance(node, Empty):
        return False
    if isinstance(node, And):
        return matches(node.left, tokens) and matches(node.right, tokens)
    if isinstance(node, Or):
        return matches(node.left, tokens) or matches(node.right, tokens)
    if isinstance(node, Not):
        return not matches(node.child, tokens)
    return bool(find_spans(node, tokens))


def scan_search(docs, query):
    """docs: {doc_id: text}. Каждый текст токенизируется заново при каждом запросе."""
    tree = parse_query(query)
    return [d for d in sorted(docs) if matches(tree, tokenize(docs[d]))]
