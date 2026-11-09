from collections import defaultdict

DAMPING = 0.85
MAX_ITER = 100
TOL = 1e-6


def build_graph(sentences, window):
    """Вершины - леммы, вес ребра - число пар в окне внутри предложения (окно 2 = соседние слова)."""
    graph = defaultdict(lambda: defaultdict(float))
    for lemmas in sentences:
        for i, a in enumerate(lemmas):
            graph[a].update()  # вершина нужна даже без рёбер
            for b in lemmas[i + 1:i + window]:
                if a != b:
                    graph[a][b] += 1
                    graph[b][a] += 1
    return graph


def pagerank(graph, damping=DAMPING, max_iter=MAX_ITER, tol=TOL):
    """Взвешенный PageRank степенным методом; вес вершины без рёбер раздаётся всем поровну."""
    nodes = list(graph)
    n = len(nodes)
    if n == 0:
        return {}
    rank = {v: 1 / n for v in nodes}
    out_weight = {v: sum(graph[v].values()) for v in nodes}
    for _ in range(max_iter):
        dangling = sum(rank[v] for v in nodes if out_weight[v] == 0)
        new_rank = {}
        for v in nodes:
            incoming = sum(rank[u] * w / out_weight[u] for u, w in graph[v].items())
            new_rank[v] = (1 - damping) / n + damping * (incoming + dangling / n)
        diff = sum(abs(new_rank[v] - rank[v]) for v in nodes)
        rank = new_rank
        if diff < tol:
            break
    return rank


def textrank_scores(sentences, window):
    return pagerank(build_graph(sentences, window))
