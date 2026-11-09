from collections import defaultdict

from utils.corpus import load_json
from utils.morph import lemmatize
from utils.paths import DATA_DIR
from utils.text import tokenize

THESAURUS_PATH = DATA_DIR / "thesaurus.json"
REQUIRED = ("concept_id", "label", "aliases", "broader", "narrower", "context")


def term_key(text):
    """Термин тезауруса или запроса -> кортеж лемм (так же нормализуется корпус)."""
    return tuple(lemmatize(t) for t in tokenize(text))


def load_thesaurus(path=THESAURUS_PATH):
    concepts = load_json(path)["concepts"]
    return {c["concept_id"]: c for c in concepts}


def concept_terms(concept):
    """Основное название и варианты названий без повторов: [(key, исходная запись)]."""
    seen = {}
    for name in [concept["label"]] + concept["aliases"]:
        seen.setdefault(term_key(name), name)
    return list(seen.items())


def context_lemmas(concept):
    return {lemmatize(w) for w in concept["context"]}


def build_lookup(thesaurus):
    """key термина -> список concept_id. Несколько id = неоднозначный термин."""
    lookup = defaultdict(list)
    for cid, concept in thesaurus.items():
        for key, _ in concept_terms(concept):
            if cid not in lookup[key]:
                lookup[key].append(cid)
    return dict(lookup)


def find_cycles(thesaurus):
    """Циклы по связи broader (поиск в глубину с тремя цветами)."""
    color, cycles = {}, []

    def visit(cid, path):
        color[cid] = "grey"
        for parent in thesaurus[cid]["broader"]:
            if parent not in thesaurus:
                continue
            if color.get(parent) == "grey":
                cycles.append(path[path.index(parent):] + [parent])
            elif parent not in color:
                visit(parent, path + [parent])
        color[cid] = "black"

    for cid in thesaurus:
        if cid not in color:
            visit(cid, [cid])
    return cycles


def validate(thesaurus):
    """Список ошибок: поля, ссылки на несуществующие понятия, несимметричные связи, циклы."""
    errors = []
    for cid, c in thesaurus.items():
        for field in REQUIRED:
            if field not in c:
                errors.append(f"{cid}: нет поля {field}")
        for relation, inverse in (("broader", "narrower"), ("narrower", "broader")):
            for other in c.get(relation, []):
                if other == cid:
                    errors.append(f"{cid}: ссылка на себя в {relation}")
                elif other not in thesaurus:
                    errors.append(f"{cid}: {relation} -> несуществующее понятие {other}")
                elif cid not in thesaurus[other].get(inverse, []):
                    errors.append(f"{cid}: {relation} {other}, но у {other} нет обратной связи {inverse}")
    for cycle in find_cycles(thesaurus):
        errors.append("цикл broader: " + " -> ".join(cycle))
    return errors


def stats(thesaurus):
    lookup = build_lookup(thesaurus)
    ambiguous = {" ".join(k): v for k, v in lookup.items() if len(v) > 1}
    return {
        "concepts": len(thesaurus),
        "synonym_links": sum(len(concept_terms(c)) - 1 for c in thesaurus.values()),
        "hierarchical_links": sum(len(c["broader"]) for c in thesaurus.values()),
        "terms": len(lookup),
        "ambiguous_terms": ambiguous,
    }
