import numpy as np


def mean_vector(terms, wv):
    """Среднее только доступных векторов; если ни одного слова нет в словаре - None (пустой вектор)."""
    vectors = [wv[t] for t in terms if t in wv.key_to_index]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def weighted_vector(term_weights, wv):
    """sum(w_t * v_t) / sum(w_t) только по словам из словаря модели."""
    total, weight_sum = None, 0.0
    for term, weight in term_weights.items():
        if term in wv.key_to_index and weight > 0:
            total = wv[term] * weight if total is None else total + wv[term] * weight
            weight_sum += weight
    if total is None:
        return None
    return total / weight_sum


def tfidf_weights(vectorizer, matrix_row):
    """Веса TF-IDF одной строки разреженной матрицы: {термин: вес}."""
    terms = vectorizer.get_feature_names_out()
    row = matrix_row.tocoo()
    return {terms[j]: float(v) for j, v in zip(row.col, row.data)}


def document_vectors(doc_terms, wv, vectorizer, matrix, mode):
    if mode == "mean":
        return [mean_vector(terms, wv) for terms in doc_terms]
    return [weighted_vector(tfidf_weights(vectorizer, matrix[i]), wv) for i in range(len(doc_terms))]


def query_vector(terms, wv, vectorizer, mode):
    if mode == "mean":
        return mean_vector(terms, wv)
    known = [t for t in terms if t in vectorizer.vocabulary_]
    if not known:
        return None
    return weighted_vector(tfidf_weights(vectorizer, vectorizer.transform([known])[0]), wv)


def coverage(terms, wv):
    """Доля слов запроса, для которых есть вектор."""
    if not terms:
        return 0.0
    return sum(1 for t in terms if t in wv.key_to_index) / len(terms)
