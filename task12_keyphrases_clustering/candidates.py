from collections import Counter

from utils.morph import lemmatize, pos
from utils.text import load_stopwords, split_sentences, tokenize, tokenize_with_spans

ALLOWED_POS = {"NOUN", "ADJF"}
MAX_LEN = 3
# местоименные прилагательные pymorphy относит к ADJF, но в ключевые фразы они не входят
PRONOUN_ADJ = {"какой", "который", "свой", "наш", "ваш", "мой", "твой", "этот", "тот", "такой", "один",
               "другой", "каждый", "всякий", "весь", "самый", "сам", "некоторый", "иной", "любой",
               "никакой", "чей", "всё", "сей", "оный", "этакий", "никой"}


def strip_header(text, title="", author=""):
    """Убирает начальные строки с автором и названием (аналог авторских ключевых слов)."""
    known = set(tokenize(title)) | set(tokenize(author))
    lines = text.split("\n")
    while lines:
        words = tokenize(lines[0])
        if words and len(words) <= 6 and sum(w in known for w in words) >= len(words) / 2:
            lines.pop(0)
        else:
            break
    return "\n".join(lines)


def is_allowed(word):
    stop = load_stopwords()
    lemma = lemmatize(word)
    return (word not in stop and lemma not in stop and lemma not in PRONOUN_ADJ
            and len(word) > 1 and pos(word) in ALLOWED_POS)


def normalize_phrase(phrase):
    """Одинаковая нормализация для эталона и выдачи: леммы через пробел."""
    return " ".join(lemmatize(w) for w in tokenize(phrase))


def sentence_runs(sentence):
    """Непрерывные цепочки допустимых слов. Стоп-слово, знак препинания или перенос строки рвут цепочку."""
    runs, current, prev_end = [], [], None
    for norm, _, start, end in tokenize_with_spans(sentence):
        gap = sentence[prev_end:start] if prev_end is not None else ""
        if current and (gap.strip() or "\n" in gap):
            runs.append(current)
            current = []
        if is_allowed(norm):
            current.append((lemmatize(norm), pos(norm)))
        elif current:
            runs.append(current)
            current = []
        prev_end = end
    if current:
        runs.append(current)
    return runs


def run_candidates(run):
    """Фразы из 1-3 соседних слов цепочки; последнее слово - существительное."""
    phrases = []
    for i in range(len(run)):
        for n in range(1, MAX_LEN + 1):
            words = run[i:i + n]
            if len(words) == n and words[-1][1] == "NOUN":
                phrases.append(" ".join(lemma for lemma, _ in words))
    return phrases


def analyze(text):
    """Леммы допустимых слов по предложениям и частоты кандидатов."""
    sentences, candidates = [], Counter()
    for sentence in split_sentences(text):
        runs = sentence_runs(sentence)
        sentences.append([lemma for run in runs for lemma, _ in run])
        for run in runs:
            candidates.update(run_candidates(run))
    return sentences, candidates
