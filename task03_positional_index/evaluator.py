from bisect import bisect_left, bisect_right

from task03_positional_index.parser import And, Empty, Near, Not, Or, Phrase, Term, parse_query


def intersect(a, b):
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def union(a, b):
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    return result + a[i:] + b[j:]


def difference(a, b):
    result, j = [], 0
    for x in a:
        while j < len(b) and b[j] < x:
            j += 1
        if j == len(b) or b[j] != x:
            result.append(x)
    return result


def near_spans(left, right, k):
    """left, right: списки (start, end). Порядок: 0 < start_right - end_left <= k."""
    right_starts = [s for s, _ in right]
    found = set()
    for s1, e1 in left:
        lo = bisect_right(right_starts, e1)
        hi = bisect_right(right_starts, e1 + k)
        for s2, e2 in right[lo:hi]:
            found.add((s1, e2))
    return sorted(found)


class Evaluator:
    def __init__(self, index):
        self.index = index

    def search(self, query):
        return self.evaluate(parse_query(query))

    def evaluate(self, node):
        if isinstance(node, Empty):
            return []
        if isinstance(node, (Term, Phrase, Near)):
            return sorted(self.spans(node))
        if isinstance(node, And):
            return intersect(self.evaluate(node.left), self.evaluate(node.right))
        if isinstance(node, Or):
            return union(self.evaluate(node.left), self.evaluate(node.right))
        if isinstance(node, Not):
            return difference(self.index.universe, self.evaluate(node.child))
        raise TypeError(f"неизвестный узел {node!r}")

    def spans(self, node):
        """Позиционный узел -> {doc_id: отсортированный список (start, end)}."""
        if isinstance(node, Term):
            return {d: [(p, p) for p in self.index.positions_of(node.word, d)]
                    for d in self.index.docs_with(node.word)}
        if isinstance(node, Phrase):
            return self.phrase_spans(node.words)
        left, right = self.spans(node.left), self.spans(node.right)
        result = {}
        for d in intersect(sorted(left), sorted(right)):
            matches = near_spans(left[d], right[d], node.k)
            if matches:
                result[d] = matches
        return result

    def phrase_spans(self, words):
        candidates = self.index.docs_with(words[0])
        for word in words[1:]:
            candidates = intersect(candidates, self.index.docs_with(word))
        result = {}
        for d in candidates:
            position_sets = [set(self.index.positions_of(w, d)) for w in words]
            starts = [p for p in self.index.positions_of(words[0], d)
                      if all(p + i in position_sets[i] for i in range(1, len(words)))]
            if starts:
                result[d] = [(p, p + len(words) - 1) for p in starts]
        return result
