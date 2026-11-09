import statistics
import time

from task07_duplicates.jaccard import containment, jaccard
from task07_duplicates.minhash import MinHasher, estimate_jaccard
from task07_duplicates.shingles import text_shingles

SHINGLE_SIZES = (2, 3, 5)
SIGNATURE_SIZES = (64, 128)
MINHASH_SEED = 1
REPEATS = 5
THRESHOLDS = [i / 100 for i in range(1, 100)]


def method_names():
    return ["exact"] + [f"minhash{n}" for n in SIGNATURE_SIZES]


def score_pairs(pairs, k, hashers):
    """Для каждой пары: статус, точный Jaccard, оценки MinHash, containment."""
    rows = []
    for p in pairs:
        set_a, status_a = text_shingles(p["text_a"], k)
        set_b, status_b = text_shingles(p["text_b"], k)
        row = {"pair_id": p["pair_id"], "type": p["type"], "split": p["split"],
               "duplicate": p["duplicate"], "k": k, "size_a": len(set_a), "size_b": len(set_b)}
        if status_a != "ok" or status_b != "ok":
            row["status"] = "insufficient_tokens"
            rows.append(row)
            continue
        row["status"] = "ok"
        row["exact"] = jaccard(set_a, set_b)
        row["containment"] = containment(set_a, set_b)
        for n, hasher in hashers.items():
            row[f"minhash{n}"] = estimate_jaccard(hasher.signature(set_a), hasher.signature(set_b))
        rows.append(row)
    return rows


def prf(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def choose_threshold(rows, method):
    """Порог с максимальным F1 на части tune; при равенстве - середина лучшего интервала."""
    y_true = [r["duplicate"] for r in rows]
    scores = [r[method] for r in rows]
    f1s = [prf(y_true, [s >= t for s in scores])["f1"] for t in THRESHOLDS]
    best = max(f1s)
    best_thresholds = [t for t, f in zip(THRESHOLDS, f1s) if f == best]
    return statistics.median(best_thresholds), best


def margin(rows, method):
    """Зазор между минимальной оценкой дубликата и максимальной оценкой не дубликата."""
    pos = [r[method] for r in rows if r["duplicate"]]
    neg = [r[method] for r in rows if not r["duplicate"]]
    return min(pos) - max(neg)


def evaluate_config(rows, method):
    tune = [r for r in rows if r["split"] == "tune" and r["status"] == "ok"]
    test = [r for r in rows if r["split"] == "test" and r["status"] == "ok"]
    threshold, tune_f1 = choose_threshold(tune, method)
    test_metrics = prf([r["duplicate"] for r in test], [r[method] >= threshold for r in test])
    return {"method": method, "threshold": threshold, "tune_f1": tune_f1,
            "tune_margin": margin(tune, method), **test_metrics}


def approximation_errors(rows, n):
    errors = [abs(r[f"minhash{n}"] - r["exact"]) for r in rows if r["status"] == "ok"]
    return {"mean_abs_error": statistics.mean(errors), "max_abs_error": max(errors),
            "median_abs_error": statistics.median(errors)}


def median_time(func, repeats=REPEATS):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def timing(pairs, k):
    """Медианное время (сек) на весь набор пар: шинглы, точный Jaccard, подписи, сравнение подписей."""
    texts = [(p["text_a"], p["text_b"]) for p in pairs]
    sets = [(text_shingles(a, k)[0], text_shingles(b, k)[0]) for a, b in texts]
    result = {"k": k, "repeats": REPEATS,
              "shingling": median_time(lambda: [(text_shingles(a, k), text_shingles(b, k)) for a, b in texts]),
              "exact_jaccard": median_time(lambda: [jaccard(a, b) for a, b in sets])}
    for n in SIGNATURE_SIZES:
        hasher = MinHasher(n, MINHASH_SEED)
        sigs = [(hasher.signature(a), hasher.signature(b)) for a, b in sets]
        result[f"signatures{n}"] = median_time(lambda: [(hasher.signature(a), hasher.signature(b)) for a, b in sets])
        result[f"compare{n}"] = median_time(lambda: [estimate_jaccard(a, b) for a, b in sigs])
    return result


def run_all(pairs):
    hashers = {n: MinHasher(n, MINHASH_SEED) for n in SIGNATURE_SIZES}
    all_rows, configs, errors, timings = [], [], [], []
    for k in SHINGLE_SIZES:
        rows = score_pairs(pairs, k, hashers)
        all_rows.extend(rows)
        for method in method_names():
            configs.append({"k": k, **evaluate_config(rows, method)})
        for n in SIGNATURE_SIZES:
            errors.append({"k": k, "num_perm": n, **approximation_errors(rows, n)})
        timings.append(timing(pairs, k))
    return all_rows, configs, errors, timings


def choose_best(configs):
    """Выбор конфигурации только по части tune: F1, затем зазор, затем точный метод."""
    return max(configs, key=lambda c: (c["tune_f1"], round(c["tune_margin"], 3), c["method"] == "exact"))
