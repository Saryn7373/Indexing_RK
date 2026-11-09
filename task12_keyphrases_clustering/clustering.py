import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from task12_keyphrases_clustering.keyphrases_tfidf import document_frequencies
from utils.morph import lemmatize
from utils.text import load_stopwords, tokenize
from utils.tfidf import build_tfidf

SEED = 42
K_VALUES = (3, 4, 5)
MIN_DF = 2
MAX_DF_RATIO = 0.5
MIN_CLUSTER_SIZE = 5


def clustering_tokens(text):
    stop = load_stopwords()
    lemmas = [lemmatize(t) for t in tokenize(text) if t not in stop and t.isalpha() and len(t) > 2]
    return [lemma for lemma in lemmas if lemma not in stop]


def build_matrix(texts):
    """TF-IDF (sublinear tf, L2-нормализация строк) по леммам.

    Отброшены редкие термины (df < 2) и слишком частые (df > 50% документов).
    """
    token_lists = [clustering_tokens(t) for t in texts]
    df = document_frequencies(token_lists)
    max_df = MAX_DF_RATIO * len(texts)
    token_lists = [[t for t in tokens if MIN_DF <= df[t] <= max_df] for tokens in token_lists]
    vectorizer, matrix = build_tfidf(token_lists, sublinear_tf=True)
    return vectorizer, matrix


def run_kmeans(matrix, k):
    model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    return model, model.fit_predict(matrix)


def run_agglomerative(matrix, k):
    model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    return model, model.fit_predict(matrix.toarray())


def choose_k(rows):
    """Максимальный silhouette среди разбиений без мелких кластеров (< 5 документов);
    если таких нет - просто максимальный silhouette."""
    readable = [r for r in rows if r["min_size"] >= MIN_CLUSTER_SIZE]
    return max(readable or rows, key=lambda r: r["silhouette"])["k"]


def silhouette(matrix, labels):
    if len(set(labels)) < 2:
        return 0.0
    return float(silhouette_score(matrix, labels, metric="cosine"))


def sample_silhouettes(matrix, labels):
    return silhouette_samples(matrix, labels, metric="cosine")


def top_terms(matrix, labels, cluster, feature_names, n=10):
    """Характерные термины: наибольший средний TF-IDF в кластере."""
    rows = matrix[np.where(labels == cluster)[0]]
    centroid = np.asarray(rows.mean(axis=0)).ravel()
    return [feature_names[i] for i in centroid.argsort()[::-1][:n]]


def central_docs(matrix, labels, cluster, n=3):
    """Индексы документов, ближайших к центроиду кластера по косинусу."""
    idx = np.where(labels == cluster)[0]
    rows = matrix[idx]
    centroid = np.asarray(rows.mean(axis=0)).ravel()
    sims = rows @ centroid / (np.linalg.norm(centroid) or 1.0)
    return [int(idx[i]) for i in np.argsort(-np.asarray(sims).ravel())[:n]]


def describe_clusters(matrix, labels, feature_names, titles):
    clusters = []
    for c in sorted(set(labels)):
        clusters.append({"cluster": int(c), "size": int((labels == c).sum()),
                         "top_terms": top_terms(matrix, labels, c, feature_names),
                         "examples": [titles[i] for i in central_docs(matrix, labels, c)]})
    return clusters
