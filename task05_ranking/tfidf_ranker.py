import numpy as np

from utils.tfidf import build_tfidf


class TfidfField:
    """TF-IDF одного поля: сырой tf, сглаженный idf, L2-нормализация (схема sklearn)."""

    def __init__(self, token_lists):
        self.n_docs = len(token_lists)
        self.vocab = {}
        if any(token_lists):
            self.vectorizer, self.matrix = build_tfidf(token_lists)
            self.vocab = self.vectorizer.vocabulary_

    def scores(self, query_terms):
        known = [t for t in query_terms if t in self.vocab]
        if not known:
            # пустое поле или нет известных терминов -> нулевая оценка
            return np.zeros(self.n_docs)
        query_vec = self.vectorizer.transform([known])
        # строки уже нормированы по L2, поэтому скалярное произведение = косинус
        return (self.matrix @ query_vec.T).toarray().ravel()

    def explain(self, query_terms, doc_index):
        known = [t for t in query_terms if t in self.vocab]
        if not known:
            return []
        query_vec = self.vectorizer.transform([known]).toarray().ravel()
        doc_vec = self.matrix[doc_index].toarray().ravel()
        rows = []
        for term in sorted(set(known)):
            col = self.vocab[term]
            rows.append({"term": term, "q_weight": query_vec[col], "d_weight": doc_vec[col],
                         "contribution": query_vec[col] * doc_vec[col]})
        return rows


class TfidfRanker:
    def __init__(self, doc_ids, text_terms, title_terms, title_weight=0.0):
        self.doc_ids = doc_ids
        self.text = TfidfField(text_terms)
        self.title = TfidfField(title_terms)
        self.title_weight = title_weight

    def scores(self, query_terms):
        result = self.text.scores(query_terms)
        if self.title_weight:
            result = result + self.title_weight * self.title.scores(query_terms)
        return result

    def rank(self, query_terms, top=None):
        return rank_scores(self.doc_ids, self.scores(query_terms), top)


def rank_scores(doc_ids, scores, top=None):
    """Список (doc_id, score) по убыванию score; документы с нулевой оценкой не выдаются."""
    pairs = [(d, float(s)) for d, s in zip(doc_ids, scores) if s > 0]
    pairs.sort(key=lambda p: (-p[1], p[0]))
    return pairs[:top] if top else pairs
