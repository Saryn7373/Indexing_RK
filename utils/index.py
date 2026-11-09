from collections import defaultdict


def build_inverted_index(doc_tokens):
    """doc_tokens: {doc_id: [термины]} -> {термин: отсортированный список doc_id}."""
    index = defaultdict(set)
    for doc_id, tokens in doc_tokens.items():
        for term in tokens:
            index[term].add(doc_id)
    return {term: sorted(ids) for term, ids in index.items()}
