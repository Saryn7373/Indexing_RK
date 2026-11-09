import math

# relevant: dict doc_id -> оценка (0, 1, 2) или множество doc_id
# при нулевом знаменателе метрика равна 0


def _rel_set(relevant):
    if isinstance(relevant, dict):
        return {d for d, r in relevant.items() if r > 0}
    return set(relevant)


def precision_at_k(ranked, relevant, k=5):
    rel = _rel_set(relevant)
    return sum(1 for d in ranked[:k] if d in rel) / k


def recall_at_k(ranked, relevant, k=5):
    rel = _rel_set(relevant)
    if not rel:
        return 0.0
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


def mrr_at_k(ranked, relevant, k=5):
    rel = _rel_set(relevant)
    for i, d in enumerate(ranked[:k], start=1):
        if d in rel:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k=5):
    grades = relevant if isinstance(relevant, dict) else {d: 1 for d in relevant}
    dcg = sum((2 ** grades.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0
