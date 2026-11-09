import json

from utils.paths import DATA_DIR


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_corpus():
    """Список документов: doc_id, title, author, genre, source, url, text."""
    return load_jsonl(DATA_DIR / "corpus.jsonl")


def load_queries(split=None):
    """Запросы с разметкой relevance {doc_id: 1 или 2}. split: 'tune', 'test' или None."""
    queries = load_json(DATA_DIR / "queries.json")
    if split:
        queries = [q for q in queries if q["split"] == split]
    return queries
