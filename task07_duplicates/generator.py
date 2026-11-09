"""Генератор размеченных пар для поиска дубликатов.

Правило разметки (записано до генерации, см. README):
duplicate = true, если текст B получен из A копированием или небольшими правками
(затронуто не более 10% слов), основное содержание и порядок фрагментов сохранены.
duplicate = false для всех остальных пар: переработка (затронуто больше 10% слов), другой текст того же произведения или о нём,
текст с цитатой из A, короткий фрагмент A (меньше половины), несвязанные тексты.
"""
import random
import re
from collections import defaultdict

from utils.corpus import load_corpus, save_json
from utils.paths import DATA_DIR

SEED = 42
BASE_WORDS = 400
PAIRS_PATH = DATA_DIR / "duplicate_pairs.json"

SYNONYMS = {
    "сказал": "произнёс", "очень": "весьма", "большой": "огромный", "большие": "огромные",
    "теперь": "сейчас", "тогда": "в то время", "только": "лишь", "всегда": "постоянно",
    "человек": "господин", "люди": "народ", "быстро": "скоро", "дом": "жилище",
    "говорил": "рассказывал", "время": "пора", "день": "денёк", "глаза": "очи",
    "ничего": "нисколько", "хорошо": "славно", "может": "способен", "после": "затем",
    "роман": "произведение", "писатель": "автор", "году": "г.", "также": "тоже",
    "однако": "но", "потом": "затем", "сразу": "тотчас", "лицо": "физиономия",
    "дело": "занятие", "жизнь": "существование", "мысль": "идея", "смотрел": "глядел",
}
FILLERS = ["вот", "даже", "уж", "именно", "просто", "всё-таки", "вдруг", "так"]


def base_text(doc, words=BASE_WORDS):
    return " ".join(doc["text"].split()[:words])


def work_key(title):
    """Ключ произведения: «Война и мир. Том 1» и «Война и мир» совпадают."""
    key = re.sub(r"\(.*?\)", " ", title.lower().replace("ё", "е"))
    key = re.sub(r"\bтом \d+", " ", key)
    key = re.sub(r"[^а-яa-z ]", " ", key)
    return " ".join(key.split())


def edit_words(words, rate, rng):
    """Небольшие правки: замена синонимом, удаление, вставка, перестановка соседних слов."""
    words = list(words)
    n_ops = max(1, round(rate * len(words)))
    for _ in range(n_ops):
        op = rng.choice(["synonym", "delete", "insert", "swap"])
        i = rng.randrange(len(words))
        if op == "synonym":
            candidates = [j for j, w in enumerate(words) if w.lower().strip(".,;:!?") in SYNONYMS]
            if candidates:
                i = rng.choice(candidates)
                words[i] = SYNONYMS[words[i].lower().strip(".,;:!?")]
            else:
                words[i] = rng.choice(FILLERS)
        elif op == "delete" and len(words) > 2:
            del words[i]
        elif op == "insert":
            words.insert(i, rng.choice(FILLERS))
        elif i + 1 < len(words):
            words[i], words[i + 1] = words[i + 1], words[i]
    return words


def insert_quote(host_text, source_text, rng, quote_words=40):
    """В текст host вставляется цитата из source в кавычках."""
    src = source_text.split()
    start = rng.randrange(max(1, len(src) - quote_words))
    quote = "«" + " ".join(src[start:start + quote_words]) + "»"
    host = host_text.split()
    pos = rng.randrange(len(host))
    return " ".join(host[:pos] + ["Как", "сказано", "в", "тексте:", quote] + host[pos:])


def build_groups(docs):
    groups = defaultdict(list)
    for d in docs:
        title = re.sub(r"\(фрагмент \d+\)", "", d["title"])
        groups[work_key(title)].append(d)
    return list(groups.values())


def split_groups(groups, rng):
    """Группы произведений делятся на tune/test, все пары строятся внутри своей части."""
    groups = list(groups)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    parts = {"tune": [], "test": []}
    for g in groups:
        target = "tune" if len(parts["tune"]) <= len(parts["test"]) else "test"
        parts[target].extend(g)
    return parts


def make_pair(kind, duplicate, sources, text_a, text_b, note=""):
    return {"type": kind, "duplicate": duplicate, "source_ids": sources,
            "text_a": text_a, "text_b": text_b, "note": note}


def pairs_for_part(docs, groups, rng):
    pairs = []
    by_id = {d["doc_id"]: d for d in docs}
    ids = sorted(by_id)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    for doc_id in shuffled[:10]:
        text = base_text(by_id[doc_id])
        pairs.append(make_pair("exact", True, [doc_id], text, text, "точная копия"))
    for doc_id in shuffled[10:22]:
        text = base_text(by_id[doc_id])
        edited = " ".join(edit_words(text.split(), 0.03, rng))
        pairs.append(make_pair("near_duplicate", True, [doc_id], text, edited, "правки 3% слов"))
    for doc_id in shuffled[22:34]:
        text = base_text(by_id[doc_id])
        edited = " ".join(edit_words(text.split(), 0.08, rng))
        pairs.append(make_pair("near_duplicate", True, [doc_id], text, edited, "правки 8% слов"))

    for doc_id in shuffled[45:51]:
        text = base_text(by_id[doc_id])
        edited = " ".join(edit_words(text.split(), 0.10, rng))
        pairs.append(make_pair("near_duplicate", True, [doc_id], text, edited, "правки 10% слов (граница правила)"))
    for doc_id in shuffled[51:57]:
        text = base_text(by_id[doc_id])
        edited = " ".join(edit_words(text.split(), 0.25, rng))
        pairs.append(make_pair("heavy_edit", False, [doc_id], text, edited, "переработка: правки 25% слов"))

    for g in groups:
        g = [d for d in g if d["doc_id"] in by_id]
        wiki = [d for d in g if d["genre"] == "критика"]
        works = [d for d in g if d["genre"] != "критика"]
        for w in works:
            for c in wiki:
                pairs.append(make_pair("topical", False, [w["doc_id"], c["doc_id"]],
                                       base_text(w), base_text(c), "произведение и статья о нём"))
        for a, b in zip(works, works[1:]):
            pairs.append(make_pair("same_work_fragment", False, [a["doc_id"], b["doc_id"]],
                                   base_text(a), base_text(b), "два фрагмента одного произведения / одно название"))

    for doc_id in shuffled[34:38]:
        same_author = [i for i in ids if i != doc_id and by_id[i]["author"] == by_id[doc_id]["author"]]
        if not same_author:
            continue
        other = rng.choice(same_author)
        pairs.append(make_pair("topical", False, [doc_id, other], base_text(by_id[doc_id]),
                               base_text(by_id[other]), "разные произведения одного автора"))

    quoted_groups = [g for g in groups if len({d["genre"] == "критика" for d in g}) == 2]
    for g in quoted_groups[:4]:
        work = next(d for d in g if d["genre"] != "критика")
        wiki = next(d for d in g if d["genre"] == "критика")
        quoted = insert_quote(base_text(wiki), base_text(work), rng)
        pairs.append(make_pair("quote", False, [work["doc_id"], wiki["doc_id"]], base_text(work), quoted,
                               "статья о произведении с цитатой из 40 слов текста A"))
    for doc_id in shuffled[42:45]:
        words = by_id[doc_id]["text"].split()[:BASE_WORDS]
        part = " ".join(words[:len(words) * 2 // 5])
        pairs.append(make_pair("fragment", False, [doc_id], " ".join(words), part,
                               "B - начальный фрагмент A (40% слов)"))

    for _ in range(10):
        a, b = rng.sample(ids, 2)
        while by_id[a]["genre"] == by_id[b]["genre"]:
            a, b = rng.sample(ids, 2)
        pairs.append(make_pair("unrelated", False, [a, b], base_text(by_id[a]), base_text(by_id[b]),
                               "разные жанры и произведения"))
    return pairs


def short_pairs(part_ids, docs):
    """Слишком короткие тексты: однословные названия и пустая строка."""
    by_title = {d["title"]: d["doc_id"] for d in docs if d["doc_id"] in part_ids}
    one_word = sorted(t for t in by_title if len(t.split()) == 1)
    pairs = []
    if one_word:
        t = one_word[0]
        pairs.append(make_pair("short", True, [by_title[t]], t, t, "копия однословного текста"))
        pairs.append(make_pair("short", False, [by_title[t]], t, "", "пустой текст"))
    if len(one_word) > 2:
        a, b = one_word[1], one_word[2]
        pairs.append(make_pair("short", False, [by_title[a], by_title[b]], a, b, "разные однословные тексты"))
    return pairs


def generate(seed=SEED):
    rng = random.Random(seed)
    docs = load_corpus()
    groups = build_groups(docs)
    parts = split_groups(groups, rng)
    all_pairs = []
    for split in ("tune", "test"):
        part_docs = parts[split]
        part_ids = {d["doc_id"] for d in part_docs}
        part_groups = [g for g in groups if g[0]["doc_id"] in part_ids]
        pairs = pairs_for_part(part_docs, part_groups, rng) + short_pairs(part_ids, part_docs)
        for p in pairs:
            p["split"] = split
        all_pairs.extend(pairs)
    for i, p in enumerate(all_pairs, start=1):
        p["pair_id"] = f"P{i:03d}"
    return all_pairs


def check_split(pairs):
    """Исходный документ не должен встречаться в обеих частях."""
    seen = {}
    for p in pairs:
        for doc_id in p["source_ids"]:
            assert seen.setdefault(doc_id, p["split"]) == p["split"], doc_id


def main():
    pairs = generate()
    check_split(pairs)
    save_json(pairs, PAIRS_PATH)
    sources = {d for p in pairs for d in p["source_ids"]}
    print(f"пар: {len(pairs)}, исходных документов: {len(sources)}, файл: {PAIRS_PATH}")


if __name__ == "__main__":
    main()
