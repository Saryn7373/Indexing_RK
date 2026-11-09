from collections import defaultdict

from task02_morphology.normalizers import analyze
from utils.index import build_inverted_index


def build_index(docs, mode, drop_stopwords=False):
    """Инвертированный индекс term -> [doc_id] и исходные написания term -> {doc_id: формы}."""
    doc_terms = {}
    forms = defaultdict(lambda: defaultdict(set))
    for doc in docs:
        pairs = analyze(doc["title"] + "\n" + doc["text"], mode, drop_stopwords)
        doc_terms[doc["doc_id"]] = [term for term, _ in pairs]
        for term, original in pairs:
            forms[term][doc["doc_id"]].add(original)
    return {
        "mode": mode,
        "drop_stopwords": drop_stopwords,
        "postings": build_inverted_index(doc_terms),
        "forms": forms,
    }


def vocabulary_size(index):
    return len(index["postings"])


def build_all(docs, modes, drop_stopwords):
    return {mode: build_index(docs, mode, drop_stopwords) for mode in modes}


def postings_for_save(index):
    return {term: index["postings"][term] for term in sorted(index["postings"])}
