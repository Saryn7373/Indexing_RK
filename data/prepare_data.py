"""Сборка общего корпуса data/corpus.jsonl из файлов, скачанных вручную.

Что должно лежать в data/raw/ (ссылки и порядок - в README.md, раздел «Данные»):
  * archive.zip   - архив датасета Kaggle "Russian literature" (d0rj3228/russian-literature, PDDL);
  * wiki/01.json ... wiki/40.json - ответы MediaWiki API русской Википедии (CC BY-SA 4.0)
    со статьями о произведениях, в порядке списка WIKI_TITLES.

Запуск из корня проекта:  python -m data.prepare_data
"""
import json
import re
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
KAGGLE_ZIP = RAW_DIR / "archive.zip"
WIKI_DIR = RAW_DIR / "wiki"
KAGGLE_PAGE = "https://www.kaggle.com/datasets/d0rj3228/russian-literature"

AUTHORS = {
    "Tolstoy": "Лев Толстой", "Dostoevsky": "Фёдор Достоевский", "Gogol": "Николай Гоголь",
    "Turgenev": "Иван Тургенев", "Pushkin": "Александр Пушкин", "Lermontov": "Михаил Лермонтов",
    "Chekhov": "Антон Чехов", "Gorky": "Максим Горький", "Herzen": "Александр Герцен",
    "Nekrasov": "Николай Некрасов", "Blok": "Александр Блок",
}

PROSE = [
    ("Tolstoy", "Анна Каренина"), ("Tolstoy", "Война и мир. Том 1"), ("Tolstoy", "Воскресение"),
    ("Tolstoy", "Смерть Ивана Ильича"), ("Tolstoy", "Крейцерова соната"), ("Tolstoy", "Хаджи-Мурат"),
    ("Tolstoy", "Детство"), ("Tolstoy", "Казаки"),
    ("Dostoevsky", "Бедные люди"), ("Dostoevsky", "Белые ночи"), ("Dostoevsky", "Бесы"),
    ("Dostoevsky", "Братья Карамазовы"), ("Dostoevsky", "Идиот"), ("Dostoevsky", "Игрок"),
    ("Gogol", "Мёртвые души"), ("Gogol", "Шинель"), ("Gogol", "Нос"), ("Gogol", "Вий"),
    ("Gogol", "Тарас Бульба"),
    ("Turgenev", "Отцы и дети"), ("Turgenev", "Дворянское гнездо"), ("Turgenev", "Муму"),
    ("Turgenev", "Ася"), ("Turgenev", "Записки охотника"),
    ("Pushkin", "Капитанская дочка"), ("Pushkin", "Пиковая дама"), ("Pushkin", "Дубровский"),
    ("Pushkin", "Повести Белкина"),
    ("Lermontov", "Герой нашего времени"),
    ("Chekhov", "Дама с собачкой"), ("Chekhov", "Палата №6"), ("Chekhov", "Человек в футляре"),
    ("Chekhov", "Крыжовник"), ("Chekhov", "Ионыч"), ("Chekhov", "Каштанка"),
    ("Gorky", "Мать"), ("Gorky", "Детство"), ("Gorky", "Старуха Изергиль"), ("Gorky", "Челкаш"),
    ("Herzen", "Кто виноват"),
]

# (автор, произведение, число фрагментов)
PUBLICISM_EXTRA = [
    ("Dostoevsky", "Дневник писателя", 5), ("Gorky", "Несвоевременные мысли", 4),
    ("Herzen", "С того берега", 3), ("Blok", "Сограждане", 1), ("Gorky", "Город жёлтого дьявола", 1),
]

WIKI_TITLES = [
    "Анна Каренина", "Война и мир", "Воскресение (роман)", "Смерть Ивана Ильича",
    "Крейцерова соната (повесть)", "Хаджи-Мурат (повесть)", "Идиот (роман)", "Бесы (роман)",
    "Братья Карамазовы", "Белые ночи (повесть)", "Бедные люди", "Мёртвые души",
    "Шинель (повесть)", "Повести Белкина", "Тарас Бульба", "Вий (повесть)",
    "Отцы и дети (роман)", "Муму", "Записки охотника", "Ася (повесть)",
    "Капитанская дочка", "Пиковая дама (повесть)", "Дубровский (роман)", "Евгений Онегин",
    "Медный всадник (поэма)", "Руслан и Людмила (поэма)", "Герой нашего времени",
    "Бородино (стихотворение)", "Кому на Руси жить хорошо", "Двенадцать (поэма)",
    "Дама с собачкой", "Палата № 6", "Человек в футляре", "Мать (роман)",
    "Старуха Изергиль", "Исповедь (Толстой)", "Дневник писателя", "Несвоевременные мысли",
    "Кто виноват? (роман)", "Каштанка",
]
WIKI_STOP_SECTIONS = ("Примечания", "Литература", "Ссылки", "См. также", "Источники", "Экранизации")

PROSE_WORDS, POEM_WORDS, WIKI_MAX_WORDS = 800, 500, 2500
POEMS_TOTAL = 40


def clean(text):
    text = text.replace("́", "").replace("\xa0", " ").replace("﻿", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def fragment(text, n_words, start_frac=0.0):
    """Фрагмент из целых строк (абзацев), начиная с доли start_frac текста."""
    lines = text.split("\n")
    start = int(len(lines) * start_frac)
    out, count = [], 0
    for line in lines[start:]:
        out.append(line)
        count += len(line.split())
        if count >= n_words:
            break
    return "\n".join(out)


def read_kaggle(path):
    """{('prose', 'Tolstoy', 'Анна Каренина'): текст} - имена в архиве в cp1251 без флага UTF-8."""
    works = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:
                raw = name.encode("cp437")
                try:
                    name = raw.decode("utf-8")
                except UnicodeDecodeError:
                    name = raw.decode("cp1251")
            parts = name.split("/")
            if len(parts) != 3 or not parts[2].endswith(".txt"):
                continue
            data = z.read(info)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("cp1251", errors="replace")
            works[(parts[0], parts[1], parts[2][:-4])] = clean(text)
    return works


def read_wiki(path):
    """Ответ API (prop=extracts|info, explaintext) -> (название, ссылка, текст без хвостовых разделов)."""
    response = json.loads(path.read_text(encoding="utf-8"))
    page = next(iter(response["query"]["pages"].values()))
    lines = []
    for line in page.get("extract", "").splitlines():
        heading = re.fullmatch(r"=+\s*(.*?)\s*=+", line.strip())
        if heading:
            if heading.group(1).startswith(WIKI_STOP_SECTIONS):
                break
            line = heading.group(1)
        lines.append(line)
    text = clean("\n".join(lines))
    if len(text.split()) > WIKI_MAX_WORDS:
        text = fragment(text, WIKI_MAX_WORDS)
    return page["title"], page["fullurl"], text


def make_doc(genre, author, title, text, source, url):
    return {"title": title, "author": author, "genre": genre, "source": source, "url": url, "text": text}


def build_docs(works):
    docs = []

    for key, title in PROSE:
        text = works[("prose", key, title)]
        docs.append(make_doc("проза", AUTHORS[key], title, fragment(text, PROSE_WORDS), "kaggle", KAGGLE_PAGE))

    poems = sorted(k for k in works if k[0] == "poems" and len(works[k].split()) >= 80)
    by_author = {}
    for k in poems:
        by_author.setdefault(k[1], []).append(k)
    picked = []
    while len(picked) < POEMS_TOTAL and any(by_author.values()):
        for author in sorted(by_author):
            if by_author[author] and len(picked) < POEMS_TOTAL:
                picked.append(by_author[author].pop(0))
    for k in sorted(picked):
        docs.append(make_doc("поэзия", AUTHORS[k[1]], k[2], fragment(works[k], POEM_WORDS), "kaggle", KAGGLE_PAGE))

    for k in sorted(k for k in works if k[0] == "publicism"):
        docs.append(make_doc("публицистика", AUTHORS[k[1]], k[2], fragment(works[k], PROSE_WORDS), "kaggle", KAGGLE_PAGE))
    for key, title, n in PUBLICISM_EXTRA:
        text = works[("prose", key, title)]
        for i in range(n):
            part = fragment(text, PROSE_WORDS, start_frac=i / n + (0.05 if n > 1 else 0.0))
            name = f"{title} (фрагмент {i + 1})" if n > 1 else title
            docs.append(make_doc("публицистика", AUTHORS[key], name, part, "kaggle", KAGGLE_PAGE))

    for i in range(1, len(WIKI_TITLES) + 1):
        title, url, text = read_wiki(WIKI_DIR / f"{i:02d}.json")
        docs.append(make_doc("критика", "Википедия", title, text, "wikipedia", url))

    for i, d in enumerate(docs, start=1):
        d["doc_id"] = f"D{i:03d}"
    return [{"doc_id": d["doc_id"], **{k: v for k, v in d.items() if k != "doc_id"}} for d in docs]


def main():
    missing = [p for p in [KAGGLE_ZIP] + [WIKI_DIR / f"{i:02d}.json" for i in range(1, len(WIKI_TITLES) + 1)]
               if not p.exists()]
    if missing:
        raise SystemExit("Не найдены файлы (см. README, раздел «Данные»): " + ", ".join(str(p) for p in missing))
    works = read_kaggle(KAGGLE_ZIP)
    docs = build_docs(works)
    with open(DATA_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    total = sum(len(d["text"].split()) for d in docs)
    print(f"Документов: {len(docs)}, слов: {total}")


if __name__ == "__main__":
    main()
