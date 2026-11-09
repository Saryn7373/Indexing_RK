from task05_ranking.bm25 import BM25Ranker
from task05_ranking.preprocess import analyze
from task05_ranking.tfidf_ranker import TfidfRanker


def build_rankers(doc_ids, text_terms, title_terms, params):
    tfidf = TfidfRanker(doc_ids, text_terms, title_terms, params["tfidf_title_weight"])
    bm25 = BM25Ranker(doc_ids, text_terms, title_terms, params["k1"], params["b"], params["bm25_title_weight"])
    return tfidf, bm25


def position(ranking, doc_id):
    for i, (d, _) in enumerate(ranking, start=1):
        if d == doc_id:
            return i
    return None


def score_of(ranking, doc_id):
    return dict(ranking).get(doc_id, 0.0)


def explain_document(tfidf, bm25, query_text, doc_id):
    terms = analyze(query_text)
    i = tfidf.doc_ids.index(doc_id)
    return {"query": query_text, "terms": terms, "doc_id": doc_id,
            "bm25_text": bm25.text.explain(terms, i), "bm25_title": bm25.title.explain(terms, i),
            "bm25_title_weight": bm25.title_weight,
            "bm25_total": float(bm25.scores(terms)[i]),
            "tfidf_text": tfidf.text.explain(terms, i), "tfidf_title": tfidf.title.explain(terms, i),
            "tfidf_title_weight": tfidf.title_weight,
            "tfidf_total": float(tfidf.scores(terms)[i])}


def toy_example(k1, b):
    """Пример из задания: D1 = 'литературное произведение', D2 = то же дважды."""
    docs = {"D1": "литературное произведение", "D2": "литературное произведение литературное произведение"}
    query = analyze("литературное произведение")
    ids = list(docs)
    terms = [analyze(t) for t in docs.values()]
    no_titles = [[] for _ in ids]
    result = {"query_terms": query, "tfidf": TfidfRanker(ids, terms, no_titles).rank(query)}
    for b_value in sorted({0.0, 0.75, b, 1.0}):
        bm25 = BM25Ranker(ids, terms, no_titles, k1, b_value)
        result[f"bm25 k1={k1}, b={b_value}"] = bm25.rank(query)
    return result


def _edited_rankings(doc_ids, text_terms, title_terms, params, doc_index, new_terms, query):
    edited = list(text_terms)
    edited[doc_index] = new_terms
    tfidf, bm25 = build_rankers(doc_ids, edited, title_terms, params)
    return tfidf.rank(query), bm25.rank(query)


def repetition_experiment(doc_ids, text_terms, title_terms, params, query_text, doc_id, term, repeats):
    """Дописываем в документ термин запроса r раз; статистика корпуса пересчитывается."""
    query = analyze(query_text)
    i = doc_ids.index(doc_id)
    rows = []
    for r in repeats:
        new_terms = text_terms[i] + [term] * r
        tfidf_rank, bm25_rank = _edited_rankings(doc_ids, text_terms, title_terms, params, i, new_terms, query)
        rows.append({"repeats": r, "tf": new_terms.count(term), "dl": len(new_terms),
                     "tfidf_score": score_of(tfidf_rank, doc_id), "tfidf_pos": position(tfidf_rank, doc_id),
                     "bm25_score": score_of(bm25_rank, doc_id), "bm25_pos": position(bm25_rank, doc_id)})
    return rows


def unrelated_text_experiment(doc_ids, text_terms, title_terms, params, query_text, doc_id, donor_id, sizes):
    """Дописываем посторонний текст (без терминов запроса): tf не меняется, растёт только длина."""
    query = analyze(query_text)
    i = doc_ids.index(doc_id)
    donor = [t for t in text_terms[doc_ids.index(donor_id)] if t not in set(query)]
    rows = []
    for size in sizes:
        added = (donor * (size // max(len(donor), 1) + 1))[:size]
        new_terms = text_terms[i] + added
        tfidf_rank, bm25_rank = _edited_rankings(doc_ids, text_terms, title_terms, params, i, new_terms, query)
        rows.append({"added_tokens": size, "dl": len(new_terms),
                     "tfidf_score": score_of(tfidf_rank, doc_id), "tfidf_pos": position(tfidf_rank, doc_id),
                     "bm25_score": score_of(bm25_rank, doc_id), "bm25_pos": position(bm25_rank, doc_id)})
    return rows


def order_differences(tfidf, bm25, queries, k=5):
    rows = []
    for q in queries:
        terms = analyze(q["text"])
        a = [d for d, _ in tfidf.rank(terms, top=k)]
        b = [d for d, _ in bm25.rank(terms, top=k)]
        if a != b:
            rows.append({"query_id": q["query_id"], "query": q["text"], "tfidf": a, "bm25": b,
                         "common": len(set(a) & set(b))})
    return rows


def annotation_case(docs, text_terms, title_terms, params, queries, k=5):
    """Аннотации и большие критические статьи (жанр 'критика'): доля в top-k и средняя длина выдачи."""
    doc_ids = [d["doc_id"] for d in docs]
    genre = {d["doc_id"]: d["genre"] for d in docs}
    length = dict(zip(doc_ids, map(len, text_terms)))
    tfidf, bm25 = build_rankers(doc_ids, text_terms, title_terms, params)
    variants = [("TF-IDF", tfidf, None)] + [(f"BM25 b={b}", bm25, b) for b in (0.0, 0.75, 1.0)]
    judged = [q for q in queries if q["relevance"]]
    rows = []
    for name, ranker, b in variants:
        if b is not None:
            ranker.set_params(params["k1"], b, params["bm25_title_weight"])
        found, crit, rel_crit = [], [], 0
        for q in judged:
            for d, _ in ranker.rank(analyze(q["text"]), top=k):
                found.append(d)
                if genre[d] == "критика":
                    crit.append(d)
                    rel_crit += q["relevance"].get(d, 0) > 0
        rows.append({"method": name, "critic_share": len(crit) / len(found) if found else 0.0,
                     "critic_relevant_share": rel_crit / len(crit) if crit else 0.0,
                     "mean_len_top": sum(length[d] for d in found) / len(found) if found else 0.0})
    bm25.set_params(params["k1"], params["b"], params["bm25_title_weight"])
    return rows


def annotation_pair(docs, text_terms, title_terms, params, query_text, doc_ids_pair):
    """Позиции короткого оригинала и длинной статьи для одного запроса при разных b."""
    doc_ids = [d["doc_id"] for d in docs]
    query = analyze(query_text)
    tfidf, bm25 = build_rankers(doc_ids, text_terms, title_terms, params)

    def row(name, ranking):
        cells = {d: f"{position(ranking, d)} ({score_of(ranking, d):.4f})" for d in doc_ids_pair}
        return {"method": name, **cells}

    rows = [row("TF-IDF", tfidf.rank(query))]
    for b in (0.0, 0.75, 1.0):
        bm25.set_params(params["k1"], b, params["bm25_title_weight"])
        rows.append(row(f"BM25 b={b}", bm25.rank(query)))
    lengths = {d: len(text_terms[doc_ids.index(d)]) for d in doc_ids_pair}
    return {"query": query_text, "lengths": lengths, "rows": rows}


def empty_answer_check(ranker, tune_queries, test_queries, query):
    """Порог отсечения = минимальная оценка top-1 на настроечных запросах (у всех есть релевантные)."""
    threshold = min(ranker.rank(analyze(q["text"]), top=1)[0][1] for q in tune_queries)
    terms = analyze(query["text"])
    ranking = ranker.rank(terms)
    above = [(d, s) for d, s in ranking if s >= threshold]
    lost = [q["query_id"] for q in test_queries if q["relevance"]
            and not [1 for _, s in ranker.rank(analyze(q["text"]), top=1) if s >= threshold]]
    return {"query_id": query["query_id"], "query": query["text"], "terms": terms,
            "docs_with_positive_score": len(ranking), "top5": ranking[:5],
            "threshold": threshold, "answer_after_threshold": above[:5], "empty_answer": not above,
            "test_queries_emptied_by_threshold": lost}
