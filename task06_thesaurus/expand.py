from task06_thesaurus.thesaurus import concept_terms, context_lemmas
from utils.morph import lemmatize
from utils.text import load_stopwords, tokenize

MODES = ("none", "synonyms", "hierarchy")
MODE_NAMES = {"none": "без расширения", "synonyms": "синонимы", "hierarchy": "синонимы + иерархия"}
DEFAULT_PARAMS = {"w_syn": 0.5, "w_hier": 0.3, "max_depth": 2, "directions": ("broader", "narrower"),
                  "link_bonus": 2, "skip_ambiguous": True}
RELATION_NAMES = {"query": "исходный", "synonym": "синоним", "broader": "шире", "narrower": "уже"}


def query_lemmas(query):
    return [lemmatize(t) for t in tokenize(query)]


def match_terms(lemmas, lookup):
    """Жадный поиск самых длинных терминов тезауруса слева направо."""
    max_len = max(len(k) for k in lookup)
    matches, i = [], 0
    while i < len(lemmas):
        for n in range(min(max_len, len(lemmas) - i), 0, -1):
            key = tuple(lemmas[i:i + n])
            if key in lookup:
                matches.append({"key": key, "start": i, "end": i + n, "candidates": list(lookup[key])})
                i += n
                break
        else:
            i += 1
    return matches


def neighbours(concept):
    return set(concept["broader"]) | set(concept["narrower"])


def score_candidate(cid, match, lemmas, thesaurus, fixed, link_bonus):
    """Контекстное правило: слова запроса из примеров контекста + бонус за связь с другим понятием запроса."""
    around = set(lemmas[:match["start"]] + lemmas[match["end"]:])
    hits = sorted(around & context_lemmas(thesaurus[cid]))
    links = sorted(c for c in fixed if c in neighbours(thesaurus[cid]))
    return {"concept_id": cid, "label": thesaurus[cid]["label"], "note": thesaurus[cid].get("note", ""),
            "score": len(hits) + link_bonus * len(links), "context_hits": hits, "links": links}


def resolve(matches, lemmas, thesaurus, link_bonus):
    """Однозначные термины принимаем сразу; для неоднозначных выбираем значение только при явном перевесе."""
    fixed = [m["candidates"][0] for m in matches if len(m["candidates"]) == 1]
    decisions = []
    for m in matches:
        term = " ".join(m["key"])
        if len(m["candidates"]) == 1:
            decisions.append({"term": term, "key": m["key"], "status": "однозначный",
                              "chosen": m["candidates"][0], "alternatives": []})
            continue
        scored = [score_candidate(c, m, lemmas, thesaurus, fixed, link_bonus) for c in m["candidates"]]
        scored.sort(key=lambda s: -s["score"])
        confident = scored[0]["score"] > 0 and scored[0]["score"] > scored[1]["score"]
        decisions.append({"term": term, "key": m["key"],
                          "status": "выбрано по контексту" if confident else "неоднозначно: показаны альтернативы",
                          "chosen": scored[0]["concept_id"] if confident else None, "alternatives": scored})
    return decisions


def walk_hierarchy(thesaurus, start, max_depth, directions):
    """Обход в ширину по связям шире/уже с ограничением глубины; visited защищает от циклов."""
    visited, frontier, result = {start}, [(start, 0)], []
    while frontier:
        cid, depth = frontier.pop(0)
        if depth == max_depth:
            continue
        for relation in directions:
            for other in thesaurus[cid][relation]:
                if other in visited or other not in thesaurus:
                    continue
                visited.add(other)
                result.append((other, relation, depth + 1))
                frontier.append((other, depth + 1))
    return result


def add_term(terms, log, key, weight, origin, concept, via, ambiguous=False):
    entry = {"term": " ".join(key), "key": key, "weight": round(weight, 4), "origin": origin,
             "relation": RELATION_NAMES[origin], "concept": concept, "via": via}
    if ambiguous:
        log.append(dict(entry, action="пропущен: форма неоднозначна в тезаурусе"))
        return
    old = terms.get(key)
    if old and old["weight"] >= weight:
        log.append(dict(entry, action=f"пропущен: уже есть как «{old['relation']}» с весом {old['weight']}"))
        return
    terms[key] = entry
    if origin != "query":
        log.append(dict(entry, action="добавлен"))


def expand(query, thesaurus, lookup, mode="hierarchy", params=None):
    params = dict(DEFAULT_PARAMS, **(params or {}))
    lemmas = query_lemmas(query)
    stop = load_stopwords()
    terms, log = {}, []
    for lemma in lemmas:
        if lemma not in stop:
            add_term(terms, log, (lemma,), 1.0, "query", None, None)
    decisions = resolve(match_terms(lemmas, lookup), lemmas, thesaurus, params["link_bonus"])
    if mode != "none":
        for d in decisions:
            if d["chosen"] is None:
                continue
            concept = thesaurus[d["chosen"]]
            for key, _ in concept_terms(concept):
                if key != d["key"]:
                    add_term(terms, log, key, params["w_syn"], "synonym", d["chosen"], d["term"],
                             params["skip_ambiguous"] and len(lookup[key]) > 1)
            if mode == "hierarchy":
                for other, relation, depth in walk_hierarchy(thesaurus, d["chosen"], params["max_depth"],
                                                             params["directions"]):
                    for key, _ in concept_terms(thesaurus[other]):
                        add_term(terms, log, key, params["w_hier"] ** depth, relation, other,
                                 f"{d['term']} ({d['chosen']}) глубина {depth}",
                                 params["skip_ambiguous"] and len(lookup[key]) > 1)
    return {"query": query, "mode": mode, "lemmas": lemmas, "terms": list(terms.values()),
            "decisions": decisions, "log": log}
