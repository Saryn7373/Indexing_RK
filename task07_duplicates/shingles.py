from utils.text import tokenize


def word_shingles(tokens, k):
    """Множество словесных шинглов длины k; шингл - кортеж слов."""
    return {tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def text_shingles(text, k):
    """(множество шинглов, статус). Статус 'insufficient_tokens', если слов меньше k."""
    tokens = tokenize(text)
    if len(tokens) < k:
        return set(), "insufficient_tokens"
    return word_shingles(tokens, k), "ok"
