import argparse
import platform
import sys

import numpy
import sklearn

from task05_ranking import experiments as ex
from task05_ranking.bm25 import BM25Ranker
from task05_ranking.preprocess import analyze, prepare_docs
from task05_ranking.report import md_table, ranking_str
from task05_ranking.tfidf_ranker import TfidfRanker
from task05_ranking.tuning import evaluate, tune_bm25, tune_tfidf
from utils.corpus import load_corpus, load_json, load_queries, save_json
from utils.paths import ROOT_DIR

RESULTS_DIR = ROOT_DIR / "task05_ranking" / "results"
PARAMS_PATH = RESULTS_DIR / "best_params.json"
DEFAULT_PARAMS = {"k1": 1.2, "b": 0.75, "bm25_title_weight": 0, "tfidf_title_weight": 0}

# контролируемые правки: запрос Q12 и релевантный документ D004 ("Смерть Ивана Ильича")
EDIT_QUERY, EDIT_DOC, EDIT_TERM, DONOR_DOC = "Q12", "D004", "чиновник", "D029"
EXPLAIN_QUERY, EXPLAIN_DOC = "Q12", "D004"
ANNOTATION_PAIRS = [("Q11", ["D075", "D144"]), ("Q12", ["D004", "D124"]), ("Q27", ["D029", "D147"])]


def environment():
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "numpy": numpy.__version__, "scikit-learn": sklearn.__version__}


def final_configs(doc_ids, text_terms, title_terms, params):
    k1, b, wb, wt = params["k1"], params["b"], params["bm25_title_weight"], params["tfidf_title_weight"]
    return [
        ("TF-IDF, только текст", TfidfRanker(doc_ids, text_terms, title_terms, 0)),
        (f"TF-IDF, текст + {wt}·заголовок", TfidfRanker(doc_ids, text_terms, title_terms, wt)),
        ("BM25 k1=1.2 b=0.75, только текст", BM25Ranker(doc_ids, text_terms, title_terms, 1.2, 0.75, 0)),
        (f"BM25 k1={k1} b={b}, только текст", BM25Ranker(doc_ids, text_terms, title_terms, k1, b, 0)),
        (f"BM25 k1={k1} b={b}, текст + {wb}·заголовок", BM25Ranker(doc_ids, text_terms, title_terms, k1, b, wb)),
    ]


def run_experiment():
    docs = load_corpus()
    doc_ids = [d["doc_id"] for d in docs]
    tune_q, test_q = load_queries("tune"), load_queries("test")
    query_by_id = {q["query_id"]: q for q in tune_q + test_q}
    text_terms, title_terms = prepare_docs(docs)
    print(f"Документов: {len(docs)}, терминов в текстах после обработки: {sum(map(len, text_terms))}")

    tfidf_grid, tfidf_best = tune_tfidf(TfidfRanker(doc_ids, text_terms, title_terms), tune_q)
    bm25_grid, bm25_best = tune_bm25(BM25Ranker(doc_ids, text_terms, title_terms), tune_q)
    params = {"k1": bm25_best["k1"], "b": bm25_best["b"], "bm25_title_weight": bm25_best["title_weight"],
              "tfidf_title_weight": tfidf_best["title_weight"]}
    save_json(params, PARAMS_PATH)

    # итоговая проверка: параметры зафиксированы, дальше только оценка
    configs = final_configs(doc_ids, text_terms, title_terms, params)
    summary, per_query = [], {}
    for name, ranker in configs:
        tune_res, test_res = evaluate(ranker, tune_q), evaluate(ranker, test_q)
        summary.append({"method": name, "tune nDCG@5": tune_res["ndcg"], "tune MRR@5": tune_res["mrr"],
                        "test nDCG@5": test_res["ndcg"], "test MRR@5": test_res["mrr"]})
        per_query[name] = test_res["per_query"]
    (tfidf_name, tfidf_final), (bm25_name, bm25_final) = configs[1], configs[4]

    per_query_rows = []
    for qid, t in per_query[tfidf_name].items():
        b = per_query[bm25_name][qid]
        per_query_rows.append({"query_id": qid, "query": query_by_id[qid]["text"],
                               "TF-IDF nDCG": t["ndcg"], "TF-IDF MRR": t["mrr"],
                               "BM25 nDCG": b["ndcg"], "BM25 MRR": b["mrr"]})

    edit_q = query_by_id[EDIT_QUERY]["text"]
    results = {
        "environment": environment(), "params": params, "tfidf_grid": tfidf_grid, "bm25_grid": bm25_grid,
        "summary": summary, "per_query_test": per_query_rows,
        "order_differences": ex.order_differences(tfidf_final, bm25_final, test_q),
        "explain": ex.explain_document(tfidf_final, bm25_final, query_by_id[EXPLAIN_QUERY]["text"], EXPLAIN_DOC),
        "repetition": ex.repetition_experiment(doc_ids, text_terms, title_terms, params, edit_q, EDIT_DOC,
                                               EDIT_TERM, [0, 1, 2, 5, 10, 20, 50]),
        "unrelated_text": ex.unrelated_text_experiment(doc_ids, text_terms, title_terms, params, edit_q, EDIT_DOC,
                                                       DONOR_DOC, [0, 250, 500, 1000, 2000, 5000]),
        "toy_example": ex.toy_example(params["k1"], params["b"]),
        "annotations": ex.annotation_case(docs, text_terms, title_terms, params, test_q),
        "annotation_pairs": [ex.annotation_pair(docs, text_terms, title_terms, params, query_by_id[qid]["text"], pair)
                             for qid, pair in ANNOTATION_PAIRS],
        "empty_answer": {name: ex.empty_answer_check(r, tune_q, test_q, query_by_id["Q29"])
                         for name, r in [("TF-IDF", tfidf_final), ("BM25", bm25_final)]},
        "test_rankings": {q["query_id"]: {"tfidf": tfidf_final.rank(analyze(q["text"]), 5),
                                          "bm25": bm25_final.rank(analyze(q["text"]), 5)} for q in test_q},
    }
    save_json(results, RESULTS_DIR / "results.json")
    report = build_report(results)
    print(report)


def build_report(r):
    e = r["explain"]
    out = ["# Задача 05: результаты\n", f"Окружение: {r['environment']}\n",
           f"Выбранные параметры (только tune): {r['params']}\n",
           "## Настройка TF-IDF (tune, 10 запросов)\n", md_table(r["tfidf_grid"]),
           "## Настройка BM25 (tune, 10 запросов)\n", md_table(r["bm25_grid"]),
           "## Итоговые метрики (test: 19 запросов с релевантными, Q29 отдельно)\n", md_table(r["summary"]),
           "## По запросам (test, итоговые конфигурации)\n", md_table(r["per_query_test"]),
           "## Запросы с разным порядком top-5\n", md_table(r["order_differences"]),
           "## Объяснение оценки документа\n",
           f"Запрос: «{e['query']}» -> {e['terms']}, документ {e['doc_id']}\n",
           "BM25, поле text:\n", md_table(e["bm25_text"]),
           f"BM25, поле title (вес {e['bm25_title_weight']}):\n", md_table(e["bm25_title"]),
           f"Итого BM25 = {e['bm25_total']:.4f}\n",
           "TF-IDF, поле text:\n", md_table(e["tfidf_text"]),
           f"TF-IDF, поле title (вес {e['tfidf_title_weight']}):\n", md_table(e["tfidf_title"]),
           f"Итого TF-IDF = {e['tfidf_total']:.4f}\n",
           "## Повторение термина\n", md_table(r["repetition"]),
           "## Добавление постороннего текста\n", md_table(r["unrelated_text"]),
           "## Пример D1/D2 из задания\n"]
    for name, value in r["toy_example"].items():
        out.append(f"- {name}: {value if name == 'query_terms' else ranking_str(value)}\n")
    out += ["\n## Особенность варианта: аннотации и большие критические статьи (test top-5)\n",
            md_table(r["annotations"])]
    for p in r["annotation_pairs"]:
        out += [f"Запрос «{p['query']}», длины (терминов): {p['lengths']}; ячейка = позиция (оценка)\n",
                md_table(p["rows"])]
    out.append("## Запрос без релевантных документов (Q29)\n")
    for name, q in r["empty_answer"].items():
        out.append(f"- {name}: термины {q['terms']}, документов с оценкой > 0: {q['docs_with_positive_score']}, "
                   f"top-5: {ranking_str(q['top5'])}; порог (мин. top-1 на tune) = {q['threshold']:.4f}; "
                   f"пустой ответ после порога: {q['empty_answer']}; "
                   f"test-запросы, которые порог обнулил бы: {q['test_queries_emptied_by_threshold']}\n")
    return "\n".join(out)


def search(query, method, top, explain_doc=None):
    params = load_json(PARAMS_PATH) if PARAMS_PATH.exists() else DEFAULT_PARAMS
    docs = load_corpus()
    doc_ids = [d["doc_id"] for d in docs]
    titles = {d["doc_id"]: d["title"] for d in docs}
    text_terms, title_terms = prepare_docs(docs)
    tfidf, bm25 = ex.build_rankers(doc_ids, text_terms, title_terms, params)
    ranker = bm25 if method == "bm25" else tfidf
    terms = analyze(query)
    print(f"Термины запроса: {terms}; метод: {method}; параметры: {params}")
    ranking = ranker.rank(terms, top)
    if not ranking:
        print("Ничего не найдено (все оценки равны 0)")
    for i, (doc_id, score) in enumerate(ranking, start=1):
        print(f"{i:2}. {doc_id}  {score:.4f}  {titles[doc_id]}")
    if explain_doc:
        info = ex.explain_document(tfidf, bm25, query, explain_doc)
        print(f"Разбор {explain_doc}, поле text:")
        print(md_table(info[f"{method}_text"]))
        print(f"Поле title (вес {info[method + '_title_weight']}):")
        print(md_table(info[f"{method}_title"]))
        print(f"Итого: {info[method + '_total']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Задача 05: TF-IDF и BM25")
    parser.add_argument("--query", help="поисковый запрос (без него запускается весь эксперимент)")
    parser.add_argument("--method", choices=["tfidf", "bm25"], default="bm25")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--explain", help="doc_id для разбора оценки по терминам")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.query:
        search(args.query, args.method, args.top, args.explain)
    else:
        run_experiment()


if __name__ == "__main__":
    main()
