from utils.morph import lemmatize, stem
from utils.text import load_stopwords, tokenize_with_spans

MODES = ("form", "stem", "lemma")
MODE_NAMES = {"form": "словоформы", "stem": "стемминг", "lemma": "лемматизация"}


def normalize_word(word, mode):
    """word уже приведён к нижнему регистру, ё заменена на е."""
    if mode == "form":
        return word
    if mode == "stem":
        return stem(word)
    if mode == "lemma":
        return lemmatize(word)
    raise ValueError(f"неизвестный режим: {mode}")


def analyze(text, mode, drop_stopwords=False):
    """Одинаковая обработка документа и запроса: [(термин, исходное написание)]."""
    stop = load_stopwords() if drop_stopwords else frozenset()
    pairs = []
    for word, original, _, _ in tokenize_with_spans(text):
        if word in stop:
            continue
        pairs.append((normalize_word(word, mode), original))
    return pairs
