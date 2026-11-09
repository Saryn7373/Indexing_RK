import argparse

from task02_morphology.index import build_index
from task02_morphology.normalizers import MODE_NAMES, MODES, analyze
from utils.corpus import load_corpus


def query_terms(query, mode, drop_stopwords=False):
    """Уникальные термины запроса в порядке появления: [(термин, исходное слово)]."""
    seen = {}
    for term, original in analyze(query, mode, drop_stopwords):
        seen.setdefault(term, original)
    return list(seen.items())


def search(query, index):
    """Объединение списков терминов; ранг по числу разных совпавших терминов, затем по doc_id."""
    terms = query_terms(query, index["mode"], index["drop_stopwords"])
    matched = {}
    for term, _ in terms:
        for doc_id in index["postings"].get(term, []):
            matched.setdefault(doc_id, []).append(term)
    ranked = sorted(matched.items(), key=lambda item: (-len(item[1]), item[0]))
    return [{"doc_id": doc_id, "score": len(found), "terms": found} for doc_id, found in ranked]


def missing_terms(query, index):
    terms = query_terms(query, index["mode"], index["drop_stopwords"])
    return [term for term, _ in terms if term not in index["postings"]]


def explain(hit, index):
    """Какие исходные словоформы документа дали совпадение."""
    parts = []
    for term in hit["terms"]:
        originals = sorted(index["forms"][term][hit["doc_id"]])
        parts.append(f"{term} <- {', '.join(originals[:5])}")
    return "; ".join(parts)


def print_results(query, index, titles, top=10):
    terms = query_terms(query, index["mode"], index["drop_stopwords"])
    print(f"Режим: {MODE_NAMES[index['mode']]}, стоп-слова удалены: {index['drop_stopwords']}")
    print("Термины запроса:", ", ".join(f"{orig} -> {term}" for term, orig in terms))
    absent = missing_terms(query, index)
    if absent:
        print("Нет в индексе:", ", ".join(absent))
    hits = search(query, index)
    print(f"Найдено документов: {len(hits)}")
    for rank, hit in enumerate(hits[:top], start=1):
        print(f"{rank:>2}. {hit['doc_id']} [{hit['score']}] {titles[hit['doc_id']]}")
        print(f"      {explain(hit, index)}")


def main():
    parser = argparse.ArgumentParser(description="Булев поиск (OR) по трём морфологическим представлениям")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=MODES + ("all",), default="all")
    parser.add_argument("--stopwords", action="store_true", help="удалять стоп-слова")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    docs = load_corpus()
    titles = {d["doc_id"]: f"{d['author']} - {d['title']}" for d in docs}
    modes = MODES if args.mode == "all" else (args.mode,)
    for mode in modes:
        print_results(args.query, build_index(docs, mode, args.stopwords), titles, args.top)
        print()


if __name__ == "__main__":
    main()
