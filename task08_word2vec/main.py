import argparse
import platform
import sys
from collections import Counter

import gensim
import numpy
import sklearn

from task08_word2vec import inspect_words as iw
from task08_word2vec.corpus_prep import content_terms, corpus_stats, load_prepared
from task08_word2vec.doc_vectors import coverage
from task08_word2vec.evaluate import empty_answer_check, evaluate, ideal_p5, query_coverage
from task08_word2vec.report import md_table, ranking_str
from task08_word2vec.search import TfidfSearch, VectorSearch
from task08_word2vec.train import COMMON, CONFIGS, load_models, reproducibility_check, train_all
from utils.corpus import load_json, load_queries, save_json
from utils.paths import ROOT_DIR

RESULTS_DIR = ROOT_DIR / "task08_word2vec" / "results"
BEST_PATH = RESULTS_DIR / "best_config.json"
MODES = ["mean", "tfidf"]
SHOW_CONFIGS = ["cbow_w5_d100", "sg_w5_d100"]
EXAMPLE_QUERY = "произведения о жизни в деревне"
NAME_QUERIES = ["князь Мышкин", "Печорин и Грушницкий", "Гринёв и Швабрин", "Каштанка", "Раскольников"]


def environment():
    return {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
            "gensim": gensim.__version__, "numpy": numpy.__version__, "scikit-learn": sklearn.__version__}


def word_experiments(models, corpus_counts):
    neighbours = {name: [iw.neighbours(m, t) for t in iw.TERMS] for name, m in models.items()}
    analogies = {name: [iw.analogy(m, *a) for a in iw.ANALOGIES] for name, m in models.items()}
    analogy_summary = [{"config": name, "верно (top-1)": sum(r["status"] == "верно" for r in rows),
                        "в top-10": sum(r["rank"] is not None for r in rows),
                        "не оценивается (OOV)": sum("OOV" in r["status"] for r in rows), "всего": len(rows)}
                       for name, rows in analogies.items()]
    names = iw.name_coverage(models[SHOW_CONFIGS[0]], corpus_counts)
    return neighbours, analogies, analogy_summary, names


def search_experiments(models, doc_ids, doc_terms, tune_q, test_q):
    tfidf = TfidfSearch(doc_ids, doc_terms)
    searchers = {("TF-IDF", "-"): tfidf}
    for name, model in models.items():
        for mode in MODES:
            searchers[(name, mode)] = VectorSearch(doc_ids, doc_terms, model.wv, tfidf.vectorizer, tfidf.matrix, mode)
    rows, per_query = [], {}
    for (name, mode), searcher in searchers.items():
        tune_res, test_res = evaluate(searcher, tune_q), evaluate(searcher, test_q)
        rows.append({"config": name, "doc_vector": mode, "tune P@5": tune_res["p5"], "tune nDCG@5": tune_res["ndcg5"],
                     "test P@5": test_res["p5"], "test nDCG@5": test_res["ndcg5"], "test MRR@5": test_res["mrr5"]})
        per_query[(name, mode)] = test_res["per_query"]
    rows.append({"config": "верхняя граница", "doc_vector": "-", "tune P@5": ideal_p5(tune_q),
                 "test P@5": ideal_p5(test_q)})
    w2v_rows = [r for r in rows if r["config"] not in ("TF-IDF", "верхняя граница")]
    best = max(w2v_rows, key=lambda r: (round(r["tune P@5"], 6), round(r["tune nDCG@5"], 6)))
    return searchers, rows, per_query, best


def run_experiment():
    docs, sentences, doc_terms = load_prepared()
    doc_ids = [d["doc_id"] for d in docs]
    tune_q, test_q = load_queries("tune"), load_queries("test")
    stats = corpus_stats(docs, sentences)
    corpus_counts = Counter(t for s in sentences for t in s)
    print("Корпус:", stats)

    models = train_all(sentences)
    repro = reproducibility_check(sentences, CONFIGS[-1])
    vocab = [{"config": n, "vocab_size": len(m.wv), "epochs": m.epochs, "window": m.window,
              "vector_size": m.vector_size, "sg": m.sg} for n, m in models.items()]

    neighbours, analogies, analogy_summary, names = word_experiments(models, corpus_counts)
    searchers, search_rows, per_query, best = search_experiments(models, doc_ids, doc_terms, tune_q, test_q)
    best_key = (best["config"], best["doc_vector"])
    save_json({"config": best["config"], "mode": best["doc_vector"]}, BEST_PATH)

    wv = models[best["config"]].wv
    coverage_rows = query_coverage(tune_q + test_q, wv)
    per_query_rows = []
    for qid, r in per_query[best_key].items():
        base = per_query[("TF-IDF", "-")][qid]
        per_query_rows.append({"query_id": qid, "coverage": next(c["coverage"] for c in coverage_rows
                                                                 if c["query_id"] == qid),
                               "W2V P@5": r["p5"], "TF-IDF P@5": base["p5"],
                               "W2V top-5": r["top"], "TF-IDF top-5": base["top"]})

    example = {name: ranking_str(searchers[(name, mode)].rank(content_terms(EXAMPLE_QUERY), 5))
               for name, mode in [("TF-IDF", "-"), best_key, ("cbow_w5_d100", "mean"), ("sg_w5_d100", "mean")]}
    name_search = []
    for text in NAME_QUERIES:
        terms = content_terms(text)
        name_search.append({"query": text, "terms": terms, "coverage": coverage(terms, wv),
                            "W2V top-3": ranking_str(searchers[best_key].rank(terms, 3)),
                            "TF-IDF top-3": ranking_str(searchers[("TF-IDF", "-")].rank(terms, 3))})
    q29 = next(q for q in test_q if not q["relevance"])
    empty = {"TF-IDF": empty_answer_check(searchers[("TF-IDF", "-")], tune_q, q29),
             f"{best_key[0]}/{best_key[1]}": empty_answer_check(searchers[best_key], tune_q, q29)}

    results = {"environment": environment(), "common_params": COMMON, "corpus": stats, "vocab": vocab,
               "reproducibility": repro, "neighbours": neighbours, "analogies": analogies,
               "analogy_summary": analogy_summary, "names": names, "search": search_rows,
               "best": best, "per_query_best": per_query_rows, "coverage": coverage_rows,
               "example_query": {"query": EXAMPLE_QUERY, "results": example}, "name_search": name_search,
               "empty_answer": empty}
    save_json(results, RESULTS_DIR / "results.json")
    report = build_report(results)
    print(report)


def neighbours_table(neighbours):
    rows = []
    for i, term in enumerate(iw.TERMS):
        row = {"term": term}
        for name in SHOW_CONFIGS:
            n = neighbours[name][i]
            row[name] = "OOV" if n["status"] == "OOV" else ", ".join(f"{w} {s:.2f}" for w, s in n["neighbours"])
        row["count"] = neighbours[SHOW_CONFIGS[0]][i]["count"]
        rows.append(row)
    return rows


def roman_table(neighbours):
    i = iw.TERMS.index("роман")
    return [{"config": name, "роман, top-3": ", ".join(f"{w} {s:.2f}" for w, s in rows[i]["neighbours"][:3])}
            for name, rows in neighbours.items()]


def build_report(r):
    out = ["# Задача 08: результаты\n", f"Окружение: {r['environment']}\n",
           f"Общие параметры обучения: {r['common_params']}\n", f"Корпус: {r['corpus']}\n",
           f"Воспроизводимость (две тренировки {r['reproducibility']['config']}): {r['reproducibility']}\n",
           "## Конфигурации\n", md_table(r["vocab"]),
           f"## Ближайшие слова (top-5, без стоп-слов): {' и '.join(SHOW_CONFIGS)}\n",
           md_table(neighbours_table(r["neighbours"])),
           "## Пример из задания: термин «роман», topn=3, все конфигурации\n", md_table(roman_table(r["neighbours"])),
           "## Аналогии: сводка\n", md_table(r["analogy_summary"])]
    for name in SHOW_CONFIGS:
        out += [f"### Аналогии, {name}\n", md_table(r["analogies"][name])]
    out += ["## Особенность варианта: литературные имена и словарь\n", md_table(r["names"]),
            md_table(r["name_search"]),
            "## Поиск: Precision@5 (19 запросов с релевантными в test, 10 в tune)\n", md_table(r["search"]),
            f"Выбрано на tune: {r['best']['config']} / {r['best']['doc_vector']}\n",
            "## По запросам (test, выбранная конфигурация против TF-IDF)\n", md_table(r["per_query_best"]),
            "## Покрытие слов запросов словарём выбранной модели\n", md_table(r["coverage"]),
            f"Среднее покрытие: {sum(c['coverage'] for c in r['coverage']) / len(r['coverage']):.4f}\n",
            f"## Пример: «{r['example_query']['query']}»\n"]
    out += [f"- {name}: {value}\n" for name, value in r["example_query"]["results"].items()]
    out.append("\n## Q29 (нет релевантных)\n")
    for name, e in r["empty_answer"].items():
        out.append(f"- {name}: top-5 {ranking_str(e['top5'])}; порог {e['threshold']:.4f}; "
                   f"пустой ответ: {e['empty_answer']}\n")
    return "\n".join(out)


def cli(args):
    models = load_models()
    if not models:
        print("Модели не найдены, сначала запустите: python -m task08_word2vec.main")
        return
    best = load_json(BEST_PATH) if BEST_PATH.exists() else {"config": "sg_w5_d100", "mode": "mean"}
    name = args.config or best["config"]
    model = models[name]
    if args.word:
        print(f"{name}: {iw.neighbours(model, args.word, args.top)}")
    if args.query:
        docs, _, doc_terms = load_prepared()
        doc_ids = [d["doc_id"] for d in docs]
        titles = {d["doc_id"]: d["title"] for d in docs}
        tfidf = TfidfSearch(doc_ids, doc_terms)
        mode = args.mode or best["mode"]
        searcher = tfidf if mode == "baseline" else VectorSearch(doc_ids, doc_terms, model.wv,
                                                                 tfidf.vectorizer, tfidf.matrix, mode)
        terms = content_terms(args.query)
        print(f"Термины: {terms}; модель: {name}; режим: {mode}; покрытие: {coverage(terms, model.wv):.2f}; "
              f"OOV: {[t for t in terms if t not in model.wv.key_to_index]}")
        ranking = searcher.rank(terms, args.top)
        if not ranking:
            print("Пустой вектор запроса (все слова вне словаря) - поиск не выполняется")
        for i, (doc_id, score) in enumerate(ranking, start=1):
            print(f"{i:2}. {doc_id}  {score:.4f}  {titles[doc_id]}")


def main():
    parser = argparse.ArgumentParser(description="Задача 08: Word2Vec")
    parser.add_argument("--word", help="термин для поиска ближайших слов")
    parser.add_argument("--query", help="поисковый запрос")
    parser.add_argument("--config", choices=[c["name"] for c in CONFIGS])
    parser.add_argument("--mode", choices=MODES + ["baseline"], help="mean, tfidf или baseline (чистый TF-IDF)")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.word or args.query:
        cli(args)
    else:
        run_experiment()


if __name__ == "__main__":
    main()
