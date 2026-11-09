"""Списки вхождений для задачи 04: документы = фрагменты текстов корпуса по CHUNK_SIZE токенов."""
from utils.corpus import load_corpus
from utils.text import tokenize

CHUNK_SIZE = 300      # токенов в одном фрагменте
MIN_TAIL = 100        # хвост короче этого присоединяется к предыдущему фрагменту


def split_tokens(tokens, size=CHUNK_SIZE, min_tail=MIN_TAIL):
    chunks = [tokens[i:i + size] for i in range(0, len(tokens), size)]
    if len(chunks) > 1 and len(chunks[-1]) < min_tail:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + tail
    return chunks


def build_chunks():
    """-> список (chunk_id, исходный doc_id, токены); chunk_id = 0, 1, 2, ... по порядку корпуса."""
    chunks = []
    for doc in load_corpus():
        for part in split_tokens(tokenize(doc["text"])):
            chunks.append((len(chunks), doc["doc_id"], part))
    return chunks


def sorted_unique(ids):
    return sorted(set(ids))


def build_postings(chunks):
    """-> {термин: отсортированный список chunk_id без повторов}."""
    postings = {}
    for chunk_id, _, tokens in chunks:
        for term in tokens:
            postings.setdefault(term, []).append(chunk_id)
    return {term: sorted_unique(ids) for term, ids in postings.items()}
