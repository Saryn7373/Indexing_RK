from itertools import product

from task05_ranking.preprocess import analyze
from utils.metrics import mean, mrr_at_k, ndcg_at_k

K1_VALUES = [0.8, 1.2, 2.0]
B_VALUES = [0.0, 0.75, 1.0]
TITLE_WEIGHTS = [0, 1, 2]


def evaluate(ranker, queries, k=5):
    """nDCG@k и MRR@k по запросам, у которых есть релевантные документы."""
    per_query = {}
    for q in queries:
        if not q["relevance"]:
            continue
        ranked = [d for d, _ in ranker.rank(analyze(q["text"]), top=k)]
        per_query[q["query_id"]] = {"ndcg": ndcg_at_k(ranked, q["relevance"], k),
                                    "mrr": mrr_at_k(ranked, q["relevance"], k), "top": ranked}
    return {"ndcg": mean(r["ndcg"] for r in per_query.values()),
            "mrr": mean(r["mrr"] for r in per_query.values()),
            "per_query": per_query}


def best_row(rows):
    # максимум nDCG@5, при равенстве - MRR@5, затем первая в порядке перебора
    return max(rows, key=lambda r: (round(r["ndcg"], 6), round(r["mrr"], 6)))


def tune_tfidf(ranker, tune_queries):
    rows = []
    for w in TITLE_WEIGHTS:
        ranker.title_weight = w
        res = evaluate(ranker, tune_queries)
        rows.append({"title_weight": w, "ndcg": res["ndcg"], "mrr": res["mrr"]})
    return rows, best_row(rows)


def tune_bm25(ranker, tune_queries):
    rows = []
    for k1, b, w in product(K1_VALUES, B_VALUES, TITLE_WEIGHTS):
        ranker.set_params(k1, b, w)
        res = evaluate(ranker, tune_queries)
        rows.append({"k1": k1, "b": b, "title_weight": w, "ndcg": res["ndcg"], "mrr": res["mrr"]})
    return rows, best_row(rows)
