import math
from collections import Counter

# idf считается так же, как в sklearn при smooth_idf: ln((1 + N) / (1 + df)) + 1


def document_frequencies(term_lists):
    df = Counter()
    for terms in term_lists:
        df.update(set(terms))
    return df


def idf_table(term_lists):
    n = len(term_lists)
    return {t: math.log((1 + n) / (1 + d)) + 1 for t, d in document_frequencies(term_lists).items()}, n


def tfidf_weights(terms, idf, n_docs):
    """tf * idf для терминов документа; неизвестный термин получает максимальный idf."""
    default_idf = math.log(1 + n_docs) + 1
    counts = Counter(terms)
    return {t: c * idf.get(t, default_idf) for t, c in counts.items()}


def score_by_words(candidates, word_scores):
    """Оценка фразы - среднее оценок её слов (общая схема для TF-IDF и TextRank)."""
    return {c: sum(word_scores.get(w, 0.0) for w in c.split()) / len(c.split()) for c in candidates}


def score_by_phrase(candidate_counts, phrase_idf, n_docs):
    """Вариант: сама фраза как термин TF-IDF."""
    return tfidf_weights(list(candidate_counts.elements()), phrase_idf, n_docs)


def top_phrases(scores, counts, k=10, words_only=False):
    items = [(c, s) for c, s in scores.items() if not words_only or " " not in c]
    items.sort(key=lambda x: (-x[1], -counts.get(x[0], 0), x[0]))
    return items[:k]
