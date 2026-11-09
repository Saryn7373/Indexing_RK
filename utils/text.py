import re
from functools import lru_cache

from utils.paths import DATA_DIR

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*", re.IGNORECASE)
SENT_END_RE = re.compile(r"(?<=[.!?…])\s+|\n\s*\n")


def normalize(text):
    # длина строки не меняется, поэтому позиции символов совпадают с исходным текстом
    return text.lower().replace("ё", "е")


def tokenize(text):
    return TOKEN_RE.findall(normalize(text))


def tokenize_with_spans(text):
    """Токены вместе с исходным написанием и диапазоном [start, end)."""
    return [(normalize(m.group()), m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def split_sentences(text):
    return [s.strip() for s in SENT_END_RE.split(text) if s and s.strip()]


@lru_cache(maxsize=1)
def load_stopwords():
    path = DATA_DIR / "stopwords_ru.txt"
    return frozenset(normalize(w.strip()) for w in path.read_text(encoding="utf-8").split() if w.strip())


def remove_stopwords(tokens):
    stop = load_stopwords()
    return [t for t in tokens if t not in stop]
