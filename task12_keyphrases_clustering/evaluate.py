import re
from collections import defaultdict

from sklearn.metrics import adjusted_rand_score

from task12_keyphrases_clustering.candidates import normalize_phrase


def precision_at_10(predicted, gold):
    """Знаменатель всегда 10, даже если фраз меньше; засчитывается только точное совпадение лемм."""
    return sum(1 for p in predicted[:10] if p in gold) / 10


def gold_sets(gold_json):
    return {d["doc_id"]: {normalize_phrase(p) for p in d["phrases"]} for d in gold_json["documents"]}


def ari(true_labels, predicted_labels):
    return float(adjusted_rand_score(true_labels, predicted_labels))


def work_key(title):
    """«Шинель (повесть)» и «Шинель» - одно произведение."""
    key = re.sub(r"\(.*?\)", " ", title.lower().replace("ё", "е"))
    key = re.sub(r"\bтом \d+", " ", key)
    return " ".join(re.sub(r"[^а-яa-z ]", " ", key).split())


def work_article_pairs(docs):
    """Пары (произведение, статья Википедии о нём): одна тема, но разные жанры."""
    groups = defaultdict(list)
    for i, d in enumerate(docs):
        groups[work_key(re.sub(r"\(фрагмент \d+\)", "", d["title"]))].append(i)
    pairs = []
    for idx in groups.values():
        works = [i for i in idx if docs[i]["genre"] != "критика"]
        articles = [i for i in idx if docs[i]["genre"] == "критика"]
        pairs.extend((w, a) for w in works for a in articles)
    return pairs


def crosstab(labels, genres):
    table = defaultdict(lambda: defaultdict(int))
    for c, g in zip(labels, genres):
        table[int(c)][g] += 1
    return table
