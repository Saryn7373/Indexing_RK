from functools import lru_cache

import pymorphy3
from nltk.stem.snowball import SnowballStemmer

_stemmer = SnowballStemmer("russian")
_morph = pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=None)
def stem(word):
    return _stemmer.stem(word)


@lru_cache(maxsize=None)
def lemmatize(word):
    return _morph.parse(word)[0].normal_form.replace("ё", "е")


@lru_cache(maxsize=None)
def pos(word):
    return _morph.parse(word)[0].tag.POS


def stem_tokens(tokens):
    return [stem(t) for t in tokens]


def lemmatize_tokens(tokens):
    return [lemmatize(t) for t in tokens]
