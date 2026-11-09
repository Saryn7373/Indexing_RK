import argparse
import platform
import statistics
from pathlib import Path

import numpy as np

from task12_keyphrases_clustering import clustering
from task12_keyphrases_clustering.candidates import analyze, strip_header
from task12_keyphrases_clustering.evaluate import (ari, crosstab, gold_sets, precision_at_10,
                                                   work_article_pairs)
from task12_keyphrases_clustering.keyphrases_tfidf import (idf_table, score_by_phrase, score_by_words,
                                                           tfidf_weights, top_phrases)
from task12_keyphrases_clustering.textrank import textrank_scores
from utils.corpus import load_corpus, load_json, save_json
from utils.paths import DATA_DIR

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GOLD_PATH = DATA_DIR / "keyphrases_gold.json"
METHODS = ["tfidf", "tfidf_phrase", "textrank_w2", "textrank_w4", "textrank_w8"]
GENRES = ["проза", "поэзия", "публицистика", "критика"]


def md_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        cells = [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def prepare(docs):
    """Тексты без заголовков, кандидаты и idf по корпусу (метки жанров не используются)."""
    texts = [strip_header(d["text"], d["title"], d["author"]) for d in docs]
    analyses = [analyze(t) for t in texts]
    word_lists = [[w for s in sentences for w in s] for sentences, _ in analyses]
    phrase_lists = [list(c) for _, c in analyses]
    word_idf, n = idf_table(word_lists)
    phrase_idf, _ = idf_table(phrase_lists)
    return {"texts": texts, "analyses": analyses, "word_idf": word_idf, "phrase_idf": phrase_idf, "n": n}


def extract(sentences, candidates, method, model, words_only=False, k=10):
    if method == "tfidf":
        words = [w for s in sentences for w in s]
        scores = score_by_words(candidates, tfidf_weights(words, model["word_idf"], model["n"]))
    elif method == "tfidf_phrase":
        scores = score_by_phrase(candidates, model["phrase_idf"], model["n"])
    else:
        window = int(method.split("_w")[1])
        scores = score_by_words(candidates, textrank_scores(sentences, window))
    return top_phrases(scores, candidates, k, words_only)


def keyphrase_experiment(docs, model):
    gold = gold_sets(load_json(GOLD_PATH))
    index = {d["doc_id"]: i for i, d in enumerate(docs)}
    per_doc, outputs, summary = [], {}, []
    for doc_id, gold_phrases in gold.items():
        sentences, candidates = model["analyses"][index[doc_id]]
        row = {"doc_id": doc_id, "title": docs[index[doc_id]]["title"], "gold": len(gold_phrases),
               "in_candidates": sum(1 for g in gold_phrases if g in candidates)}
        outputs[doc_id] = {}
        for method in METHODS:
            for mode in ("words", "phrases"):
                top = extract(sentences, candidates, method, model, words_only=(mode == "words"))
                phrases = [p for p, _ in top]
                row[f"{method}/{mode}"] = precision_at_10(phrases, gold_phrases)
                outputs[doc_id][f"{method}/{mode}"] = [(p, round(s, 4), p in gold_phrases) for p, s in top]
        per_doc.append(row)
    for method in METHODS:
        for mode in ("words", "phrases"):
            values = [r[f"{method}/{mode}"] for r in per_doc]
            summary.append({"method": method, "mode": mode, "P@10": statistics.mean(values),
                            "min": min(values), "max": max(values)})
    return per_doc, summary, outputs, gold


def clustering_experiment(docs, model):
    vectorizer, matrix = clustering.build_matrix(model["texts"])
    names = vectorizer.get_feature_names_out()
    titles = [f"{d['doc_id']} {d['title']}" for d in docs]
    runs, rows = {}, []
    for method, func in (("kmeans", clustering.run_kmeans), ("agglomerative", clustering.run_agglomerative)):
        for k in clustering.K_VALUES:
            _, labels = func(matrix, k)
            runs[(method, k)] = labels
            sizes = sorted(np.bincount(labels).tolist(), reverse=True)
            rows.append({"method": method, "k": k, "silhouette": clustering.silhouette(matrix, labels),
                         "sizes": str(sizes), "min_size": sizes[-1]})
    chosen = {m: clustering.choose_k([r for r in rows if r["method"] == m]) for m in ("kmeans", "agglomerative")}
    descriptions = {m: clustering.describe_clusters(matrix, runs[(m, k)], names, titles) for m, k in chosen.items()}
    return matrix, names, runs, rows, chosen, descriptions


def reveal_genres(docs, matrix, runs, rows, chosen):
    """Скрытые жанры используются только здесь: ARI и сравнение после выбора k."""
    genres = [d["genre"] for d in docs]
    for r in rows:
        r["ARI_genre"] = ari(genres, runs[(r["method"], r["k"])])
        r["chosen"] = "да" if chosen[r["method"]] == r["k"] else ""
    tables = {}
    for method, k in chosen.items():
        tab = crosstab(runs[(method, k)], genres)
        tables[method] = [{"cluster": c, **{g: tab[c][g] for g in GENRES}} for c in sorted(tab)]
    pairs = work_article_pairs(docs)
    same = {m: sum(runs[(m, k)][a] == runs[(m, k)][b] for a, b in pairs) for m, k in chosen.items()}
    labels = runs[("kmeans", chosen["kmeans"])]
    sil = clustering.sample_silhouettes(matrix, labels)
    boundary = [{"doc": f"{docs[i]['doc_id']} {docs[i]['title']}", "genre": genres[i],
                 "cluster": int(labels[i]), "silhouette": float(sil[i])} for i in np.argsort(sil)[:6]]
    pair_rows = [{"work": f"{docs[a]['doc_id']} {docs[a]['title']} ({genres[a]})",
                  "article": f"{docs[b]['doc_id']} {docs[b]['title']}",
                  **{m: f"{runs[(m, k)][a]} / {runs[(m, k)][b]}" for m, k in chosen.items()}}
                 for a, b in pairs]
    return tables, {"pairs": len(pairs), "same_cluster": {m: int(v) for m, v in same.items()}}, pair_rows, boundary


def plot_2d(matrix, labels, docs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD

    points = TruncatedSVD(n_components=2, random_state=clustering.SEED).fit_transform(matrix)
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
    markers = {"проза": "o", "поэзия": "^", "публицистика": "s", "критика": "D"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for genre, marker in markers.items():
        for c in sorted(set(labels)):
            idx = [i for i, d in enumerate(docs) if d["genre"] == genre and labels[i] == c]
            if idx:
                ax.scatter(points[idx, 0], points[idx, 1], c=colors[c % len(colors)], marker=marker, s=40,
                           edgecolors="white", linewidths=0.8,
                           label=f"кластер {c}, {genre}")
    ax.set_title("K-means по TF-IDF, проекция TruncatedSVD (цвет - кластер, форма - жанр)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def environment():
    import matplotlib
    import pymorphy3
    import sklearn
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "numpy": np.__version__, "sklearn": sklearn.__version__,
            "pymorphy3": pymorphy3.__version__, "matplotlib": matplotlib.__version__}


def run_experiment():
    RESULTS_DIR.mkdir(exist_ok=True)
    docs = load_corpus()
    model = prepare(docs)
    per_doc, summary, outputs, gold = keyphrase_experiment(docs, model)
    matrix, names, runs, rows, chosen, descriptions = clustering_experiment(docs, model)
    tables, variant, pair_rows, boundary = reveal_genres(docs, matrix, runs, rows, chosen)
    plot_2d(matrix, runs[("kmeans", chosen["kmeans"])], docs, RESULTS_DIR / "clusters_2d.png")
    save_json({"kmeans_k": chosen["kmeans"], "agglomerative_k": chosen["agglomerative"]}, RESULTS_DIR / "chosen_k.json")

    parts = [f"Документов: {len(docs)}, терминов в TF-IDF для кластеризации: {len(names)}"]
    parts.append("## Precision@10 ключевых фраз (среднее по 10 документам)\n\n"
                 + md_table(summary, ["method", "mode", "P@10", "min", "max"]))
    parts.append("## Precision@10 по документам (in_candidates - сколько эталонных фраз есть среди кандидатов)\n\n"
                 + md_table(per_doc, ["doc_id", "title", "gold", "in_candidates"] + [f"{m}/phrases" for m in METHODS]))
    for doc_id in gold:
        lines = [f"### {doc_id}"]
        for method in ("tfidf", "textrank_w4"):
            items = ", ".join(f"{p}{'*' if hit else ''} ({s})" for p, s, hit in outputs[doc_id][f"{method}/phrases"])
            lines.append(f"- {method}: {items}")
        parts.append("\n".join(lines))
    parts.append("## Кластеризация: silhouette (cosine) и выбор k без меток\n\n"
                 + md_table(rows, ["method", "k", "silhouette", "sizes", "min_size"]))
    for method, clusters in descriptions.items():
        lines = [f"## Каталог тем: {method}, k={chosen[method]}"]
        for c in clusters:
            lines.append(f"- кластер {c['cluster']} ({c['size']} док.): {', '.join(c['top_terms'])}; "
                         f"примеры: {'; '.join(c['examples'])}")
        parts.append("\n".join(lines))
    parts.append("## Итоговая проверка: ARI относительно скрытых жанров (k выбран до этого шага)\n\n"
                 + md_table(rows, ["method", "k", "silhouette", "ARI_genre", "chosen"]))
    for method, tab in tables.items():
        parts.append(f"## Состав кластеров по жанрам: {method}, k={chosen[method]}\n\n"
                     + md_table(tab, ["cluster"] + GENRES))
    parts.append(f"## Особенность варианта: жанр и тема\n\nПар «произведение - статья о нём»: {variant['pairs']}, "
                 f"в одном кластере: {variant['same_cluster']}\n\n"
                 + md_table(pair_rows, ["work", "article"] + list(chosen)))
    parts.append("## Пограничные документы (минимальный silhouette, K-means)\n\n"
                 + md_table(boundary, ["doc", "genre", "cluster", "silhouette"]))
    report = "\n\n".join(parts)
    print(report)
    save_json({"keyphrases_summary": summary, "keyphrases_per_doc": per_doc, "keyphrases_top10": outputs,
               "clustering": rows, "chosen_k": chosen, "catalog": descriptions, "genre_crosstab": tables,
               "variant_case": variant, "work_article_pairs": pair_rows, "boundary_docs": boundary,
               "environment": environment()}, RESULTS_DIR / "results.json")


def keyphrases_cli(text=None, doc_id=None, method="textrank_w4"):
    docs = load_corpus()
    model = prepare(docs)
    if doc_id:
        i = next(i for i, d in enumerate(docs) if d["doc_id"] == doc_id)
        text = model["texts"][i]
    sentences, candidates = analyze(text)
    print([(p, round(s, 4)) for p, s in extract(sentences, candidates, method, model)])
    chosen_path = RESULTS_DIR / "chosen_k.json"
    k = load_json(chosen_path)["kmeans_k"] if chosen_path.exists() else 4
    vectorizer, matrix = clustering.build_matrix(model["texts"])
    kmeans, _ = clustering.run_kmeans(matrix, k)
    vector = vectorizer.transform([clustering.clustering_tokens(text)])
    print("cluster_id:", int(kmeans.predict(vector)[0]), f"(K-means, k={k}; номер кластера не является меткой темы)")


def main():
    parser = argparse.ArgumentParser(description="Ключевые фразы и кластеризация")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("experiment", help="полный эксперимент (по умолчанию)")
    kp = sub.add_parser("keyphrases", help="ключевые фразы и кластер для текста или документа")
    kp.add_argument("--doc", help="doc_id из корпуса")
    kp.add_argument("--text", help="текст или путь к файлу")
    kp.add_argument("--method", default="textrank_w4", choices=METHODS)
    args = parser.parse_args()
    if args.command == "keyphrases":
        text = args.text
        if text and Path(text).is_file():
            text = Path(text).read_text(encoding="utf-8")
        keyphrases_cli(text, args.doc, args.method)
    else:
        run_experiment()


if __name__ == "__main__":
    main()
