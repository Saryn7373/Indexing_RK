from utils.morph import lemmatize
from utils.text import load_stopwords, normalize

TERMS = ["роман", "повесть", "поэма", "стихотворение", "поэт", "писатель", "читатель", "критик", "библиотека",
         "любовь", "смерть", "война", "деревня", "крестьянин", "церковь", "революция",
         "онегин", "печорин", "швабрин", "раскольников"]

# a : b :: c : ?  ->  ответ = b - a + c, ожидается expected
ANALOGIES = [
    ("отец", "мать", "сын", "дочь"),
    ("муж", "жена", "брат", "сестра"),
    ("царь", "царица", "князь", "княгиня"),
    ("день", "ночь", "утро", "вечер"),
    ("поэт", "стихотворение", "писатель", "роман"),
    ("пушкин", "онегин", "лермонтов", "печорин"),
    ("гоголь", "чичиков", "тургенев", "базаров"),
    ("россия", "русский", "франция", "француз"),
    ("пушкин", "александр", "гоголь", "николай"),
    ("лермонтов", "михаил", "тургенев", "иван"),
    ("толстой", "каренин", "достоевский", "раскольников"),
]

# литературные имена: проверяем, попали ли они в словарь после лемматизации и min_count
NAMES = ["Мышкин", "Грушницкий", "Данко", "Изергиль", "Гринёв", "Швабрин", "Каштанка", "Муму",
         "Раскольников", "Обломов", "Базаров", "Алёша"]


def to_lemma(word):
    return lemmatize(normalize(word))


def neighbours(model, word, topn=5):
    """Ближайшие слова без стоп-слов или статус OOV (фиктивный вектор не создаётся)."""
    wv = model.wv
    term = to_lemma(word)
    if term not in wv.key_to_index:
        return {"term": term, "status": "OOV", "count": 0, "neighbours": []}
    stop = load_stopwords()
    similar = [(w, round(float(s), 3)) for w, s in wv.most_similar(term, topn=topn + 20) if w not in stop]
    return {"term": term, "status": "ok", "count": int(wv.get_vecattr(term, "count")), "neighbours": similar[:topn]}


def analogy(model, a, b, c, expected, topn=10):
    wv = model.wv
    a, b, c, expected = map(to_lemma, (a, b, c, expected))
    missing = [w for w in (a, b, c) if w not in wv.key_to_index]
    row = {"analogy": f"{a}:{b} :: {c}:?", "expected": expected}
    if missing:
        return {**row, "status": f"OOV во входе: {', '.join(missing)}", "answer": None, "rank": None}
    if expected not in wv.key_to_index:
        # ответ модели показываем, но правильным не считаем
        answer = wv.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]
        return {**row, "status": "ожидаемое слово OOV, не оценивается", "answer": answer, "rank": None}
    candidates = [w for w, _ in wv.most_similar(positive=[b, c], negative=[a], topn=topn)]
    rank = candidates.index(expected) + 1 if expected in candidates else None
    status = "верно" if rank == 1 else ("в top-10" if rank else "неверно")
    return {**row, "status": status, "answer": candidates[0], "rank": rank}


def name_coverage(model, corpus_counts):
    rows = []
    for name in NAMES:
        lemma = to_lemma(name)
        rows.append({"name": name, "lemma": lemma, "corpus_count": corpus_counts.get(lemma, 0),
                     "in_vocab": lemma in model.wv.key_to_index})
    return rows
