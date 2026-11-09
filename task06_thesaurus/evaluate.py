from itertools import product

from task06_thesaurus.expand import DEFAULT_PARAMS, expand
from task06_thesaurus.search import search
from utils.metrics import mean, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k

K = 5
W_SYN_GRID = [0.2, 0.3, 0.5, 0.7, 1.0]
W_HIER_GRID = [0.1, 0.2, 0.3, 0.5]
DEPTH_GRID = [1, 2, 3]
DIRECTIONS_GRID = [("broader",), ("narrower",), ("broader", "narrower")]


def is_ambiguous(expansion):
    return any(d["alternatives"] for d in expansion["decisions"])


def run_queries(queries, index, thesaurus, lookup, mode, params):
    rows = []
    for q in queries:
        exp = expand(q["text"], thesaurus, lookup, mode, params)
        hits = search(exp, index)
        ranked = [h["doc_id"] for h in hits]
        rel = q["relevance"]
        rows.append({"query_id": q["query_id"], "text": q["text"], "split": q["split"], "tags": q["tags"],
                     "ambiguous": is_ambiguous(exp), "found": len(ranked), "top": hits[:K],
                     "p5": precision_at_k(ranked, rel, K), "r5": recall_at_k(ranked, rel, K),
                     "mrr5": mrr_at_k(ranked, rel, K), "ndcg5": ndcg_at_k(ranked, rel, K), "expansion": exp})
    return rows


def summarize(rows):
    """Средние по запросам с релевантными документами (запрос без ответа оценивается отдельно)."""
    scored = [r for r in rows if "no_relevant" not in r["tags"]]
    return {"queries": len(scored), "P@5": mean(r["p5"] for r in scored), "R@5": mean(r["r5"] for r in scored),
            "MRR@5": mean(r["mrr5"] for r in scored), "nDCG@5": mean(r["ndcg5"] for r in scored)}


def objective(rows):
    s = summarize(rows)
    return s["P@5"] + s["R@5"], s["nDCG@5"]


def tune(queries, index, thesaurus, lookup):
    """Подбор весов только на настроечных запросах: сначала w_syn, затем параметры иерархии."""
    grid = []
    best_syn, best_value = None, None
    for w_syn in W_SYN_GRID:
        params = dict(DEFAULT_PARAMS, w_syn=w_syn)
        value = objective(run_queries(queries, index, thesaurus, lookup, "synonyms", params))
        grid.append({"mode": "synonyms", "w_syn": w_syn, "P+R": value[0], "nDCG": value[1]})
        if best_value is None or value > best_value:
            best_syn, best_value = w_syn, value
    best_params, best_value = None, None
    for w_hier, depth, directions in product(W_HIER_GRID, DEPTH_GRID, DIRECTIONS_GRID):
        params = dict(DEFAULT_PARAMS, w_syn=best_syn, w_hier=w_hier, max_depth=depth, directions=directions)
        value = objective(run_queries(queries, index, thesaurus, lookup, "hierarchy", params))
        grid.append({"mode": "hierarchy", "w_syn": best_syn, "w_hier": w_hier, "max_depth": depth,
                     "directions": "+".join(directions), "P+R": value[0], "nDCG": value[1]})
        if best_value is None or value > best_value:
            best_params, best_value = params, value
    return best_params, grid


def top_ids(row):
    return [h["doc_id"] for h in row["top"]]


def added_documents(base_rows, rows, queries, relations):
    """Документы, вошедшие в top-5 после расширения, и связи тезауруса, которые дали вклад."""
    by_id = {q["query_id"]: q for q in queries}
    result = []
    for base, row in zip(base_rows, rows):
        rel = by_id[row["query_id"]]["relevance"]
        for hit in row["top"]:
            if hit["doc_id"] in top_ids(base):
                continue
            links = [c for c in hit["contrib"] if c["relation"] in relations]
            result.append({"query_id": row["query_id"], "doc_id": hit["doc_id"],
                           "relevant": rel.get(hit["doc_id"], 0), "links": links[:3]})
    return result


def removed_documents(base_rows, rows, queries):
    by_id = {q["query_id"]: q for q in queries}
    result = []
    for base, row in zip(base_rows, rows):
        rel = by_id[row["query_id"]]["relevance"]
        for doc_id in top_ids(base):
            if doc_id not in top_ids(row):
                result.append({"query_id": row["query_id"], "doc_id": doc_id, "relevant": rel.get(doc_id, 0)})
    return result


def empty_answer(rows):
    return [{"query_id": r["query_id"], "found": r["found"], "correct_empty": r["found"] == 0,
             "top": top_ids(r)} for r in rows if "no_relevant" in r["tags"]]
