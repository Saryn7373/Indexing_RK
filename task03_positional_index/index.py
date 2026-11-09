import json
from bisect import bisect_left, insort

from utils.text import tokenize


class PositionalIndex:
    """Позиционный индекс: термин -> doc_id -> отсортированный список позиций.

    Позиции считаются по полному потоку токенов документа (стоп-слова не удаляются),
    нумерация с 0 заново в каждом документе.
    """

    def __init__(self):
        self.doc_lists = {}   # термин -> отсортированный список doc_id
        self.positions = {}   # термин -> {doc_id: [позиции]}
        self.doc_terms = {}   # doc_id -> множество терминов (нужно для удаления)
        self.doc_lengths = {}  # doc_id -> число токенов
        self.universe = []    # отсортированный список всех актуальных doc_id

    @classmethod
    def build(cls, docs):
        """docs: {doc_id: text} -> индекс, построенный с нуля."""
        index = cls()
        for doc_id in sorted(docs):
            index.add_document(doc_id, docs[doc_id])
        return index

    def add_document(self, doc_id, text):
        if doc_id in self.doc_terms:
            self.remove_document(doc_id)
        tokens = tokenize(text)
        doc_positions = {}
        for pos, term in enumerate(tokens):
            doc_positions.setdefault(term, []).append(pos)
        for term, plist in doc_positions.items():
            insort(self.doc_lists.setdefault(term, []), doc_id)
            self.positions.setdefault(term, {})[doc_id] = plist
        self.doc_terms[doc_id] = set(doc_positions)
        self.doc_lengths[doc_id] = len(tokens)
        insort(self.universe, doc_id)

    def update_document(self, doc_id, text):
        """Старые вхождения удаляются внутри add_document."""
        self.add_document(doc_id, text)

    def remove_document(self, doc_id):
        for term in self.doc_terms.pop(doc_id):
            ids = self.doc_lists[term]
            del ids[bisect_left(ids, doc_id)]
            del self.positions[term][doc_id]
            if not ids:
                del self.doc_lists[term]
                del self.positions[term]
        del self.doc_lengths[doc_id]
        del self.universe[bisect_left(self.universe, doc_id)]

    def docs_with(self, term):
        return self.doc_lists.get(term, [])

    def positions_of(self, term, doc_id):
        return self.positions.get(term, {}).get(doc_id, [])

    def is_sorted(self):
        for term, ids in self.doc_lists.items():
            if ids != sorted(set(ids)) or set(self.positions[term]) != set(ids):
                return False
            for plist in self.positions[term].values():
                if plist != sorted(set(plist)):
                    return False
        return self.universe == sorted(set(self.universe))

    def to_dict(self):
        """Каноническое представление для сравнения и сохранения."""
        return {
            "universe": list(self.universe),
            "doc_lengths": {d: self.doc_lengths[d] for d in self.universe},
            "postings": {
                term: {d: self.positions[term][d] for d in self.doc_lists[term]}
                for term in sorted(self.doc_lists)
            },
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    def stats(self):
        n_postings = sum(len(ids) for ids in self.doc_lists.values())
        n_positions = sum(self.doc_lengths.values())
        return {"documents": len(self.universe), "terms": len(self.doc_lists),
                "postings": n_postings, "positions": n_positions}
