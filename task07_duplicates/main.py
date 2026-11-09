import argparse
import platform
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from task07_duplicates import generator
from task07_duplicates.evaluate import SIGNATURE_SIZES, choose_best, method_names, run_all
from task07_duplicates.jaccard import containment, jaccard
from task07_duplicates.minhash import MinHasher, estimate_jaccard
from task07_duplicates.shingles import text_shingles
from utils.corpus import load_json, save_json

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONFIG_PATH = RESULTS_DIR / "best_config.json"


def md_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        cells = [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def type_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[(r["k"], r["type"])].append(r)
    table = []
    for (k, kind), items in sorted(groups.items()):
        table.append({"k": k, "type": kind, "n": len(items),
                      "duplicate": items[0]["duplicate"],
                      "exact_min": min(r["exact"] for r in items),
                      "exact_mean": statistics.mean(r["exact"] for r in items),
                      "exact_max": max(r["exact"] for r in items),
                      "containment_mean": statistics.mean(r["containment"] for r in items)})
    return table


def errors_of(rows, best):
    method, k, t = best["method"], best["k"], best["threshold"]
    wrong = []
    for r in rows:
        if r["k"] == k and r["status"] == "ok" and (r[method] >= t) != r["duplicate"]:
            wrong.append({"pair_id": r["pair_id"], "split": r["split"], "type": r["type"],
                          "duplicate": r["duplicate"], "score": r[method], "exact": r["exact"]})
    return wrong


def all_config_errors(rows, configs):
    """Ошибки на test для каждой конфигурации (порог выбран на tune)."""
    result = []
    for c in configs:
        for r in rows:
            if r["k"] == c["k"] and r["split"] == "test" and r["status"] == "ok":
                if (r[c["method"]] >= c["threshold"]) != r["duplicate"]:
                    result.append({"k": c["k"], "method": c["method"], "pair_id": r["pair_id"],
                                   "type": r["type"], "duplicate": r["duplicate"],
                                   "score": r[c["method"]], "threshold": c["threshold"]})
    return result


def insufficient_report(rows, pairs):
    notes = {p["pair_id"]: p["note"] for p in pairs}
    by_pair = defaultdict(list)
    for r in rows:
        if r["status"] == "insufficient_tokens":
            by_pair[(r["pair_id"], r["split"], r["duplicate"])].append(str(r["k"]))
    return [{"pair_id": pid, "split": split, "duplicate": dup, "note": notes[pid],
             "k": ", ".join(ks), "status": "insufficient_tokens"}
            for (pid, split, dup), ks in sorted(by_pair.items())]


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "machine": platform.machine(),
            "numpy": np.__version__}


def run_experiment():
    RESULTS_DIR.mkdir(exist_ok=True)
    if not generator.PAIRS_PATH.exists():
        generator.main()
    pairs = load_json(generator.PAIRS_PATH)
    rows, configs, errors, timings = run_all(pairs)
    best = choose_best(configs)
    save_json({"k": best["k"], "method": best["method"], "threshold": best["threshold"]}, CONFIG_PATH)

    types = type_summary(rows)
    wrong = errors_of(rows, best)
    wrong_all = all_config_errors(rows, configs)
    short = insufficient_report(rows, pairs)
    variant = [r for r in rows if r["type"] in ("quote", "fragment", "same_work_fragment") and r["status"] == "ok"]
    variant_cols = ["k", "pair_id", "split", "type", "exact", "minhash128", "containment"]

    tables = {
        "Качество на test (порог выбран на tune)": md_table(configs, ["k", "method", "threshold", "tune_f1", "tune_margin", "precision", "recall", "f1", "tp", "fp", "fn"]),
        "Ошибка оценки MinHash относительно точного Jaccard": md_table(errors, ["k", "num_perm", "mean_abs_error", "median_abs_error", "max_abs_error"]),
        "Время на весь набор пар, сек (медиана)": md_table(timings, ["k", "repeats", "shingling", "exact_jaccard"] + [f"{p}{n}" for n in SIGNATURE_SIZES for p in ("signatures", "compare")]),
        "Точный Jaccard по типам пар": md_table(types, ["k", "type", "n", "duplicate", "exact_min", "exact_mean", "exact_max", "containment_mean"]),
        "Особенность варианта: цитаты и фрагменты": md_table(variant, variant_cols),
        "Ошибки выбранной конфигурации": md_table(wrong, ["pair_id", "split", "type", "duplicate", "score", "exact"]) if wrong else "нет ошибок",
        "Ошибки всех конфигураций на test": md_table(wrong_all, ["k", "method", "pair_id", "type", "duplicate", "score", "threshold"]) if wrong_all else "нет ошибок",
        "Недостаточно токенов (оцениваются отдельно)": md_table(short, ["pair_id", "split", "duplicate", "note", "k", "status"]),
    }
    n_sources = len({d for p in pairs for d in p["source_ids"]})
    header = (f"Пар: {len(pairs)} (tune {sum(p['split'] == 'tune' for p in pairs)}, "
              f"test {sum(p['split'] == 'test' for p in pairs)}), исходных документов: {n_sources}\n"
              f"Выбранная конфигурация: k={best['k']}, {best['method']}, порог {best['threshold']:.2f}")
    report = header + "\n\n" + "\n\n".join(f"## {name}\n\n{table}" for name, table in tables.items())
    print(report)
    save_json({"best": best, "configs": configs, "errors": errors, "timings": timings,
               "type_summary": types, "insufficient": short, "misclassified": wrong,
               "misclassified_all_configs": wrong_all, "environment": environment()},
              RESULTS_DIR / "results.json")
    save_json(rows, RESULTS_DIR / "pair_scores.json")


def compare(text_a, text_b, k=None, threshold=None):
    config = load_json(CONFIG_PATH) if CONFIG_PATH.exists() else {"k": 3, "threshold": 0.5}
    k = k or config["k"]
    threshold = threshold if threshold is not None else config["threshold"]
    set_a, status_a = text_shingles(text_a, k)
    set_b, status_b = text_shingles(text_b, k)
    if "insufficient_tokens" in (status_a, status_b):
        print(f"k={k}: недостаточно токенов (A: {status_a}, B: {status_b}), решение не принимается")
        return
    if len(set_a) + len(set_b) <= 20:
        print("A:", sorted(" ".join(s) for s in set_a))
        print("B:", sorted(" ".join(s) for s in set_b))
        print("общие:", sorted(" ".join(s) for s in set_a & set_b))
    exact = jaccard(set_a, set_b)
    print(f"k={k}, |A|={len(set_a)}, |B|={len(set_b)}, Jaccard={exact:.3f}, containment={containment(set_a, set_b):.3f}")
    for n in SIGNATURE_SIZES:
        h = MinHasher(n, 1)
        print(f"MinHash{n}: {estimate_jaccard(h.signature(set_a), h.signature(set_b)):.3f}")
    print(f"порог {threshold:.2f}: {'ДУБЛИКАТ' if exact >= threshold else 'не дубликат'}")


def read_arg(value):
    path = Path(value)
    return path.read_text(encoding="utf-8") if path.is_file() else value


def main():
    parser = argparse.ArgumentParser(description="Поиск дубликатов: шинглы, Jaccard, MinHash")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("experiment", help="полный эксперимент (по умолчанию)")
    cmp = sub.add_parser("compare", help="сравнить два текста (строки или пути к файлам)")
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.add_argument("--k", type=int)
    cmp.add_argument("--threshold", type=float)
    args = parser.parse_args()
    if args.command == "compare":
        compare(read_arg(args.a), read_arg(args.b), args.k, args.threshold)
    else:
        run_experiment()


if __name__ == "__main__":
    main()
