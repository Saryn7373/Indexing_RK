import argparse

from task06_thesaurus.expand import DEFAULT_PARAMS, MODE_NAMES, MODES, expand
from task06_thesaurus.index import build_index, term_scores
from task06_thesaurus.thesaurus import build_lookup, load_thesaurus
from utils.corpus import load_corpus


def search(expansion, index):
    """Сумма вкладов терминов: вес термина * (1 + ln tf) * idf. Ранг по score, при равенстве по doc_id."""
    scores, contrib = {}, {}
    for term in expansion["terms"]:
        for doc_id, s in term_scores(index, term["key"]).items():
            value = term["weight"] * s
            scores[doc_id] = scores.get(doc_id, 0.0) + value
            contrib.setdefault(doc_id, []).append({"term": term["term"], "relation": term["relation"],
                                                   "concept": term["concept"], "weight": term["weight"],
                                                   "value": round(value, 4)})
    ranked = sorted(scores, key=lambda d: (-scores[d], d))
    return [{"doc_id": d, "score": round(scores[d], 4),
             "contrib": sorted(contrib[d], key=lambda c: -c["value"])} for d in ranked]


def print_expansion(expansion, thesaurus):
    print(f"Запрос: {expansion['query']}  |  режим: {MODE_NAMES[expansion['mode']]}")
    for d in expansion["decisions"]:
        print(f"  термин «{d['term']}»: {d['status']}", end="")
        if d["chosen"]:
            c = thesaurus[d["chosen"]]
            print(f" -> {d['chosen']} ({c['label']}{', ' + c['note'] if c.get('note') else ''})")
        else:
            print()
        for alt in d["alternatives"]:
            print(f"      вариант {alt['concept_id']} ({alt['note'] or alt['label']}): score={alt['score']}, "
                  f"контекст={alt['context_hits']}, связи={alt['links']}")
    for entry in expansion["log"]:
        print(f"  [{entry['relation']}] {entry['term']} w={entry['weight']} понятие={entry['concept']} "
              f"из «{entry['via']}»: {entry['action']}")


def print_results(hits, titles, top):
    print(f"Найдено документов: {len(hits)}")
    for rank, hit in enumerate(hits[:top], start=1):
        parts = ", ".join(f"{c['term']}[{c['relation']}]={c['value']}" for c in hit["contrib"][:4])
        print(f"{rank:>2}. {hit['doc_id']} {hit['score']:.3f} {titles[hit['doc_id']]}")
        print(f"      {parts}")


def main():
    parser = argparse.ArgumentParser(description="Тезаурусный поиск с расширением запроса")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=MODES + ("all",), default="all")
    parser.add_argument("--w-syn", type=float, default=DEFAULT_PARAMS["w_syn"])
    parser.add_argument("--w-hier", type=float, default=DEFAULT_PARAMS["w_hier"])
    parser.add_argument("--depth", type=int, default=DEFAULT_PARAMS["max_depth"])
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    thesaurus = load_thesaurus()
    lookup = build_lookup(thesaurus)
    docs = load_corpus()
    index = build_index(docs)
    titles = {d["doc_id"]: f"{d['author']} - {d['title']}" for d in docs}
    params = {"w_syn": args.w_syn, "w_hier": args.w_hier, "max_depth": args.depth}
    for mode in (MODES if args.mode == "all" else (args.mode,)):
        expansion = expand(args.query, thesaurus, lookup, mode, params)
        print_expansion(expansion, thesaurus)
        print_results(search(expansion, index), titles, args.top)
        print()


if __name__ == "__main__":
    main()
