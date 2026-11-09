from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(token_lists, sublinear_tf=False, smooth_idf=True):
    """TF-IDF по уже токенизированным текстам.

    Схема sklearn: tf - сырая частота (или 1 + ln(tf) при sublinear_tf),
    idf = ln((1 + N) / (1 + df)) + 1 при smooth_idf, строки нормализуются по L2.
    """
    vectorizer = TfidfVectorizer(analyzer=lambda tokens: tokens, lowercase=False,
                                 sublinear_tf=sublinear_tf, smooth_idf=smooth_idf, norm="l2")
    matrix = vectorizer.fit_transform(token_lists)
    return vectorizer, matrix
