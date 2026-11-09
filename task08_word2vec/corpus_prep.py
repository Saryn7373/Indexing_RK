from collections import Counter

from utils.corpus import load_corpus
from utils.morph import lemmatize_tokens
from utils.text import load_stopwords, split_sentences, tokenize


def lemmas(text):
    """Очистка: нижний регистр, ё -> е, только слова (без пунктуации), лемматизация."""
    return lemmatize_tokens(tokenize(text))


def content_terms(text):
    """Термины для поиска и TF-IDF: леммы без стоп-слов. Одинаково для документов и запросов."""
    stop = load_stopwords()
    return [t for t in lemmas(text) if t not in stop]


def corpus_sentences(docs):
    """Каждое предложение - отдельная последовательность лемм (стоп-слова сохраняются как контекст)."""
    sentences = []
    for doc in docs:
        for sentence in split_sentences(doc["title"] + ".\n\n" + doc["text"]):
            tokens = lemmas(sentence)
            if len(tokens) > 1:
                sentences.append(tokens)
    return sentences


def corpus_stats(docs, sentences):
    raw_tokens = sum(len(tokenize(d["text"])) for d in docs)
    counts = Counter(t for s in sentences for t in s)
    return {"documents": len(docs), "raw_text_tokens": raw_tokens, "sentences": len(sentences),
            "training_tokens": sum(counts.values()), "distinct_lemmas": len(counts),
            "mean_sentence_len": sum(counts.values()) / len(sentences)}


def load_prepared():
    docs = load_corpus()
    sentences = corpus_sentences(docs)
    doc_terms = [content_terms(d["title"] + " " + d["text"]) for d in docs]
    return docs, sentences, doc_terms
