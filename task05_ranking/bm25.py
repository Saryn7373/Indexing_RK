import math
from collections import Counter

import numpy as np

from task05_ranking.tfidf_ranker import rank_scores


class BM25Field:
    """Собственная реализация BM25 для одного поля документа."""

    def __init__(self, token_lists, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.tf = [Counter(tokens) for tokens in token_lists]
        self.doc_len = [len(tokens) for tokens in token_lists]
        self.n_docs = len(token_lists)
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        self.df = Counter()
        for counts in self.tf:
            self.df.update(counts.keys())

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def term_score(self, term, doc_index):
        tf = self.tf[doc_index].get(term, 0)
        dl = self.doc_len[doc_index]
        if tf == 0 or dl == 0:
            return 0.0
        norm = 1 - self.b + self.b * dl / self.avgdl
        return self.idf(term) * tf * (self.k1 + 1) / (tf + self.k1 * norm)

    def score(self, query_terms, doc_index):
        # сумма вкладов различных терминов запроса
        return sum(self.term_score(t, doc_index) for t in set(query_terms))

    def scores(self, query_terms):
        return np.array([self.score(query_terms, i) for i in range(self.n_docs)])

    def explain(self, query_terms, doc_index):
        rows = []
        for term in sorted(set(query_terms)):
            rows.append({"term": term, "tf": self.tf[doc_index].get(term, 0), "df": self.df.get(term, 0),
                         "idf": self.idf(term), "dl": self.doc_len[doc_index], "avgdl": self.avgdl,
                         "contribution": self.term_score(term, doc_index)})
        return rows


class BM25Ranker:
    def __init__(self, doc_ids, text_terms, title_terms, k1=1.2, b=0.75, title_weight=0.0):
        self.doc_ids = doc_ids
        self.text = BM25Field(text_terms, k1, b)
        self.title = BM25Field(title_terms, k1, b)
        self.title_weight = title_weight

    def set_params(self, k1, b, title_weight):
        for field in (self.text, self.title):
            field.k1, field.b = k1, b
        self.title_weight = title_weight

    def scores(self, query_terms):
        result = self.text.scores(query_terms)
        if self.title_weight:
            result = result + self.title_weight * self.title.scores(query_terms)
        return result

    def rank(self, query_terms, top=None):
        return rank_scores(self.doc_ids, self.scores(query_terms), top)
