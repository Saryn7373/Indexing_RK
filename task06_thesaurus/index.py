import math
from collections import Counter, defaultdict

from utils.morph import lemmatize
from utils.text import tokenize


def doc_lemmas(doc):
    return [lemmatize(t) for t in tokenize(doc["title"] + "\n" + doc["text"])]


def build_index(docs):
    """Собственное поисковое представление: последовательности лемм и индекс лемма -> {doc_id: tf}."""
    sequences = {d["doc_id"]: doc_lemmas(d) for d in docs}
    postings = defaultdict(dict)
    for doc_id, lemmas in sequences.items():
        for lemma, tf in Counter(lemmas).items():
            postings[lemma][doc_id] = tf
    return {"sequences": sequences, "postings": dict(postings), "n_docs": len(docs), "phrase_cache": {}}


def count_phrase(sequence, key):
    n = len(key)
    return sum(1 for i in range(len(sequence) - n + 1) if tuple(sequence[i:i + n]) == key)


def term_postings(index, key):
    """key - кортеж лемм. Однословный термин берём из индекса, фразу ищем подряд в документах-кандидатах."""
    if len(key) == 1:
        return index["postings"].get(key[0], {})
    if key in index["phrase_cache"]:
        return index["phrase_cache"][key]
    candidates = None
    for lemma in key:
        docs = set(index["postings"].get(lemma, {}))
        candidates = docs if candidates is None else candidates & docs
    result = {}
    for doc_id in sorted(candidates or []):
        tf = count_phrase(index["sequences"][doc_id], key)
        if tf:
            result[doc_id] = tf
    index["phrase_cache"][key] = result
    return result


def idf(index, key):
    df = len(term_postings(index, key))
    return math.log(index["n_docs"] / df) if df else 0.0


def term_scores(index, key):
    """Вклад термина в документы: (1 + ln tf) * idf."""
    w = idf(index, key)
    return {doc_id: (1 + math.log(tf)) * w for doc_id, tf in term_postings(index, key).items()}
