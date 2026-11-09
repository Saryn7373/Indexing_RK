from utils.morph import lemmatize_tokens
from utils.text import load_stopwords, remove_stopwords, tokenize


def analyze(text):
    """Одинаковая обработка документов и запросов: токены -> без стоп-слов -> леммы -> без стоп-слов."""
    if not text:
        return []
    stop = load_stopwords()
    lemmas = lemmatize_tokens(remove_stopwords(tokenize(text)))
    return [t for t in lemmas if t not in stop]


def prepare_docs(docs):
    """Термины основного текста и заголовка для каждого документа."""
    text_terms = [analyze(d.get("text", "")) for d in docs]
    title_terms = [analyze(d.get("title", "")) for d in docs]
    return text_terms, title_terms
