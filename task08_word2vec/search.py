import numpy as np

from task08_word2vec.doc_vectors import document_vectors, query_vector
from utils.tfidf import build_tfidf


def cosine(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 0 else None


class VectorSearch:
    """Поиск по усреднённым векторам Word2Vec (mode: 'mean' или 'tfidf')."""

    def __init__(self, doc_ids, doc_terms, wv, vectorizer, matrix, mode):
        self.doc_ids = doc_ids
        self.wv = wv
        self.vectorizer = vectorizer
        self.mode = mode
        self.doc_vecs = document_vectors(doc_terms, wv, vectorizer, matrix, mode)

    def rank(self, terms, top=None):
        q = query_vector(terms, self.wv, self.vectorizer, self.mode)
        if q is None:
            return []  # пустой вектор запроса: cosine не вычисляется, ответ пустой
        pairs = []
        for doc_id, vec in zip(self.doc_ids, self.doc_vecs):
            sim = cosine(q, vec) if vec is not None else None
            if sim is not None:
                pairs.append((doc_id, sim))
        pairs.sort(key=lambda p: (-p[1], p[0]))
        return pairs[:top] if top else pairs


class TfidfSearch:
    """Базовый метод: TF-IDF + косинус, обучен внутри задачи 08."""

    def __init__(self, doc_ids, doc_terms):
        self.doc_ids = doc_ids
        self.vectorizer, self.matrix = build_tfidf(doc_terms)

    def rank(self, terms, top=None):
        known = [t for t in terms if t in self.vectorizer.vocabulary_]
        if not known:
            return []
        scores = (self.matrix @ self.vectorizer.transform([known]).T).toarray().ravel()
        pairs = [(d, float(s)) for d, s in zip(self.doc_ids, scores) if s > 0]
        pairs.sort(key=lambda p: (-p[1], p[0]))
        return pairs[:top] if top else pairs
