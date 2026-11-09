from task02_morphology.search import missing_terms, search
from utils.metrics import mean, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k

K = 5


def evaluate_query(query, index):
    hits = search(query["text"], index)
    ranked = [h["doc_id"] for h in hits]
    rel = query["relevance"]
    return {
        "query_id": query["query_id"],
        "split": query["split"],
        "tags": query["tags"],
        "found": len(ranked),
        "top": hits[:K],
        "missing": missing_terms(query["text"], index),
        "p5": precision_at_k(ranked, rel, K),
        "r5": recall_at_k(ranked, rel, K),
        "mrr5": mrr_at_k(ranked, rel, K),
        "ndcg5": ndcg_at_k(ranked, rel, K),
    }


def evaluate_mode(queries, index):
    return [evaluate_query(q, index) for q in queries]


def summarize(rows, split, vocab_size):
    """Средние метрики по запросам с релевантными документами; запрос без ответа - отдельно."""
    part = [r for r in rows if r["split"] == split]
    scored = [r for r in part if "no_relevant" not in r["tags"]]
    return {
        "split": split,
        "queries": len(scored),
        "P@5": mean(r["p5"] for r in scored),
        "R@5": mean(r["r5"] for r in scored),
        "MRR@5": mean(r["mrr5"] for r in scored),
        "nDCG@5": mean(r["ndcg5"] for r in scored),
        "vocab": vocab_size,
        "oov_any": mean(1.0 if r["missing"] else 0.0 for r in part),
        "oov_all": mean(1.0 if r["missing"] and r["found"] == 0 else 0.0 for r in part),
        "avg_found": mean(r["found"] for r in part),
    }


def empty_answer_check(rows):
    """Для запроса без релевантных документов корректна пустая выдача."""
    result = []
    for r in rows:
        if "no_relevant" in r["tags"]:
            result.append({"query_id": r["query_id"], "found": r["found"], "correct_empty": r["found"] == 0,
                           "top": [h["doc_id"] for h in r["top"]]})
    return result


def top_ids(row):
    return [h["doc_id"] for h in row["top"]]


def stopword_effect(rows_plain, rows_stop):
    """Как меняется top-5 при удалении стоп-слов (одни и те же запросы и эталон)."""
    changed = []
    for a, b in zip(rows_plain, rows_stop):
        before, after = set(top_ids(a)), set(top_ids(b))
        if before != after:
            changed.append({"query_id": a["query_id"], "removed": sorted(before - after),
                            "added": sorted(after - before), "p5_before": a["p5"], "p5_after": b["p5"]})
    return changed


def tag_summary(rows, tag):
    part = [r for r in rows if tag in r["tags"]]
    return {"tag": tag, "queries": len(part), "P@5": mean(r["p5"] for r in part), "R@5": mean(r["r5"] for r in part)}


def false_matches(rows, queries, index):
    """Нерелевантные документы в top-5 и исходные словоформы, которые дали совпадение."""
    by_id = {q["query_id"]: q for q in queries}
    result = []
    for r in rows:
        rel = by_id[r["query_id"]]["relevance"]
        for hit in r["top"]:
            if hit["doc_id"] not in rel:
                forms = {t: sorted(index["forms"][t][hit["doc_id"]])[:4] for t in hit["terms"]}
                result.append({"query_id": r["query_id"], "doc_id": hit["doc_id"], "forms": forms})
    return result


def mode_diff(rows_a, rows_b, queries, index_b):
    """Документы, которые появились или исчезли в top-5 при переходе от режима A к режиму B."""
    by_id = {q["query_id"]: q for q in queries}
    result = []
    for a, b in zip(rows_a, rows_b):
        rel = by_id[a["query_id"]]["relevance"]
        ids_a, ids_b = top_ids(a), top_ids(b)
        appeared = []
        for hit in b["top"]:
            if hit["doc_id"] not in ids_a:
                forms = {t: sorted(index_b["forms"][t][hit["doc_id"]])[:4] for t in hit["terms"]}
                appeared.append({"doc_id": hit["doc_id"], "relevant": hit["doc_id"] in rel, "forms": forms})
        disappeared = [{"doc_id": d, "relevant": d in rel} for d in ids_a if d not in ids_b]
        if appeared or disappeared:
            result.append({"query_id": a["query_id"], "appeared": appeared, "disappeared": disappeared})
    return result
