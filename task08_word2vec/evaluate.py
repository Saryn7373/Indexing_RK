from task08_word2vec.corpus_prep import content_terms
from task08_word2vec.doc_vectors import coverage
from utils.metrics import mean, mrr_at_k, ndcg_at_k, precision_at_k


def evaluate(searcher, queries, k=5):
    """Precision@5 (основная), nDCG@5 и MRR@5 по запросам с релевантными документами."""
    per_query = {}
    for q in queries:
        if not q["relevance"]:
            continue
        ranked = [d for d, _ in searcher.rank(content_terms(q["text"]), top=k)]
        per_query[q["query_id"]] = {"p5": precision_at_k(ranked, q["relevance"], k),
                                    "ndcg5": ndcg_at_k(ranked, q["relevance"], k),
                                    "mrr5": mrr_at_k(ranked, q["relevance"], k), "top": ranked}
    return {"p5": mean(r["p5"] for r in per_query.values()),
            "ndcg5": mean(r["ndcg5"] for r in per_query.values()),
            "mrr5": mean(r["mrr5"] for r in per_query.values()), "per_query": per_query}


def ideal_p5(queries, k=5):
    """Максимально возможный Precision@5: при 2 релевантных документах он не больше 0.4."""
    return mean(min(k, len(q["relevance"])) / k for q in queries if q["relevance"])


def query_coverage(queries, wv):
    rows = []
    for q in queries:
        terms = content_terms(q["text"])
        rows.append({"query_id": q["query_id"], "terms": terms,
                     "oov": [t for t in terms if t not in wv.key_to_index], "coverage": coverage(terms, wv)})
    return rows


def empty_answer_check(searcher, tune_queries, query):
    """Порог = минимальная оценка top-1 на настроечных запросах; ниже порога ответ считаем пустым."""
    tops = [searcher.rank(content_terms(q["text"]), top=1) for q in tune_queries]
    threshold = min(top[0][1] for top in tops if top)
    ranking = searcher.rank(content_terms(query["text"]), top=5)
    top1 = ranking[0][1] if ranking else None
    return {"top5": ranking, "threshold": threshold,
            "empty_answer": top1 is None or top1 < threshold}
