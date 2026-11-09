"""Случайные запросы по реальным текстам корпуса для сравнения индекса и просмотра."""
import random

from utils.text import tokenize


def random_operand(rng, doc_tokens, vocabulary):
    tokens = rng.choice(doc_tokens)
    kind = rng.random()
    if kind < 0.4 or len(tokens) < 4:
        return rng.choice(tokens if rng.random() < 0.7 else vocabulary)
    start = rng.randrange(len(tokens) - 3)
    if kind < 0.75:
        length = rng.choice([2, 3])
        return '"' + " ".join(tokens[start:start + length]) + '"'
    gap = rng.randint(1, 3)
    k = rng.randint(1, 4)
    return f"{tokens[start]} NEAR/{k} {tokens[min(start + gap, len(tokens) - 1)]}"


def random_query(rng, doc_tokens, vocabulary, depth=0):
    if depth >= 2 or rng.random() < 0.3:
        operand = random_operand(rng, doc_tokens, vocabulary)
        return f"NOT {operand}" if rng.random() < 0.15 else operand
    op = rng.choice(["AND", "OR", "AND NOT"])
    left = random_query(rng, doc_tokens, vocabulary, depth + 1)
    right = random_query(rng, doc_tokens, vocabulary, depth + 1)
    return f"({left} {op} {right})"


def generate_queries(docs, count, seed=42):
    rng = random.Random(seed)
    doc_tokens = [tokenize(docs[d]) for d in sorted(docs)]
    vocabulary = sorted({t for tokens in doc_tokens for t in tokens})
    return [random_query(rng, doc_tokens, vocabulary) for _ in range(count)]
