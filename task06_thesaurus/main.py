from task06_thesaurus import evaluate as ev
from task06_thesaurus.expand import MODE_NAMES, MODES, expand, walk_hierarchy
from task06_thesaurus.index import build_index
from task06_thesaurus.search import search
from task06_thesaurus.thesaurus import build_lookup, load_thesaurus, stats, validate
from utils.corpus import load_corpus, load_queries, save_json
from utils.paths import ROOT_DIR

RESULTS_DIR = ROOT_DIR / "task06_thesaurus" / "results"
TEMP_DIR = ROOT_DIR / "task06_thesaurus" / "temp"
DEMO_QUERIES = ["художественный текст", "стихотворный роман", "новелла о собаке", "поэма Блока",
                "белые ночи", "мир"]
MIN_REQUIREMENTS = {"concepts": 40, "synonym_links": 10, "hierarchical_links": 10, "ambiguous_terms": 5}


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(x):
    return f"{x:.3f}"


def check_thesaurus(thesaurus):
    errors = validate(thesaurus)
    info = stats(thesaurus)
    counts = {k: (len(v) if isinstance(v, dict) else v) for k, v in info.items()}
    for name, minimum in MIN_REQUIREMENTS.items():
        if counts[name] < minimum:
            errors.append(f"{name}: {counts[name]} < {minimum}")
    return errors, info, counts


def cycle_guard_demo():
    """Искусственный тезаурус с циклом A -> B -> A: проверка находит цикл, обход завершается."""
    toy = {"A": {"concept_id": "A", "label": "а", "aliases": [], "broader": ["B"], "narrower": ["B"], "context": []},
           "B": {"concept_id": "B", "label": "б", "aliases": [], "broader": ["A"], "narrower": ["A"], "context": []}}
    return validate(toy), walk_hierarchy(toy, "A", 10, ("broader", "narrower"))


def contrib_text(links):
    return "; ".join(f"«{c['term']}» [{c['relation']}, {c['concept']}, w={c['weight']}] +{c['value']:.2f}"
                     for c in links)


def decisions_rows(rows, thesaurus):
    result = []
    for r in rows:
        for d in r["expansion"]["decisions"]:
            if not d["alternatives"]:
                continue
            alts = "; ".join(f"{a['concept_id']}={a['score']} (контекст {a['context_hits']}, связи {a['links']})"
                             for a in d["alternatives"])
            chosen = f"{d['chosen']} ({thesaurus[d['chosen']].get('note', '')})" if d["chosen"] else "-"
            result.append([r["query_id"], r["text"], d["term"], d["status"], chosen, alts])
    return result


def run():
    RESULTS_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    thesaurus = load_thesaurus()
    errors, info, counts = check_thesaurus(thesaurus)
    print("Тезаурус:", counts)
    if errors:
        raise SystemExit("Ошибки тезауруса:\n" + "\n".join(errors))
    lookup = build_lookup(thesaurus)
    cycle_errors, cycle_walk = cycle_guard_demo()

    docs = load_corpus()
    queries = load_queries()
    tune_q = [q for q in queries if q["split"] == "tune"]
    test_q = [q for q in queries if q["split"] == "test"]
    index = build_index(docs)
    print(f"Документов: {len(docs)}, лемм в индексе: {len(index['postings'])}")

    best, grid = ev.tune(tune_q, index, thesaurus, lookup)
    print("Лучшие параметры на tune:", best)

    rows = {mode: ev.run_queries(queries, index, thesaurus, lookup, mode, best) for mode in MODES}
    table = []
    for split in ("tune", "test"):
        for group, select in (("все", lambda r: True), ("однозначные", lambda r: not r["ambiguous"]),
                              ("неоднозначные", lambda r: r["ambiguous"])):
            for mode in MODES:
                part = [r for r in rows[mode] if r["split"] == split and select(r)]
                s = ev.summarize(part)
                table.append([split, group, MODE_NAMES[mode], s["queries"], fmt(s["P@5"]), fmt(s["R@5"]),
                              fmt(s["MRR@5"]), fmt(s["nDCG@5"])])
    header = ["выборка", "группа", "режим", "запросов", "P@5", "R@5", "MRR@5", "nDCG@5"]

    test_rows = {mode: [r for r in rows[mode] if r["split"] == "test"] for mode in MODES}
    added_syn = ev.added_documents(test_rows["none"], test_rows["synonyms"], test_q, {"синоним"})
    added_hier = ev.added_documents(test_rows["synonyms"], test_rows["hierarchy"], test_q, {"шире", "уже"})
    removed_hier = ev.removed_documents(test_rows["synonyms"], test_rows["hierarchy"], test_q)
    wrong = [a for a in added_hier if not a["relevant"]]
    empty = {mode: ev.empty_answer(rows[mode]) for mode in MODES}

    md = ["# Задача 06. Результаты", "",
          "## Тезаурус", "", md_table(["показатель", "значение", "минимум"],
                                      [[k, counts[k], MIN_REQUIREMENTS.get(k, "-")] for k in counts]), "",
          "Неоднозначные термины: " + "; ".join(f"«{k}» -> {', '.join(v)}" for k, v in info["ambiguous_terms"].items()),
          "", "Ошибок проверки ссылок, симметрии broader/narrower и циклов: 0.", "",
          f"Проверка защиты от циклов на искусственном тезаурусе A<->B: validate -> {cycle_errors}; "
          f"обход из A посетил {[c for c, _, _ in cycle_walk]} и завершился.", "",
          "## Подбор параметров (tune, Q01-Q10)", "",
          md_table(["режим", "w_syn", "w_hier", "глубина", "направления", "P@5+R@5", "nDCG@5"],
                   [[g["mode"], g["w_syn"], g.get("w_hier", "-"), g.get("max_depth", "-"), g.get("directions", "-"),
                     fmt(g["P+R"]), fmt(g["nDCG"])] for g in grid]), "",
          f"Выбрано: w_syn={best['w_syn']}, w_hier={best['w_hier']}, глубина={best['max_depth']}, "
          f"направления={'+'.join(best['directions'])}, link_bonus={best['link_bonus']}", "",
          "## Сравнение режимов", "", md_table(header, table), "",
          "## Запрос без релевантных документов (Q29)", "",
          md_table(["режим", "найдено", "пустой ответ?", "top-5"],
                   [[MODE_NAMES[m], e[0]["found"], e[0]["correct_empty"], e[0]["top"]] for m, e in empty.items()]), "",
          "## Разрешение неоднозначности (все 30 запросов)", "",
          md_table(["id", "запрос", "термин", "решение", "выбрано", "варианты: score"],
                   decisions_rows(rows["synonyms"], thesaurus)), "",
          "## Документы, добавленные синонимами (test, относительно режима без расширения)", "",
          md_table(["запрос", "документ", "релевантность", "вклад связей тезауруса"],
                   [[a["query_id"], a["doc_id"], a["relevant"], contrib_text(a["links"])] for a in added_syn]), "",
          "## Документы, добавленные иерархией (test, относительно режима синонимов)", "",
          md_table(["запрос", "документ", "релевантность", "вклад связей тезауруса"],
                   [[a["query_id"], a["doc_id"], a["relevant"], contrib_text(a["links"])] for a in added_hier]), "",
          "## Ошибочные обобщения (иерархия добавила нерелевантный документ)", "",
          md_table(["запрос", "документ", "связь"], [[a["query_id"], a["doc_id"], contrib_text(a["links"])]
                                                     for a in wrong]), "",
          "Вытесненные иерархией документы: " + ", ".join(
              f"{r['query_id']}:{r['doc_id']}(rel={r['relevant']})" for r in removed_hier), ""]

    per_query = []
    for q in queries:
        row = [q["query_id"], q["split"], q["text"]]
        for mode in MODES:
            r = next(x for x in rows[mode] if x["query_id"] == q["query_id"])
            row.append(f"{r['p5']:.1f}/{r['r5']:.2f}")
        row.append("да" if rows["none"][queries.index(q)]["ambiguous"] else "")
        per_query.append(row)
    md += ["## P@5/R@5 по запросам", "",
           md_table(["id", "выборка", "запрос"] + [MODE_NAMES[m] for m in MODES] + ["неоднозначный"], per_query), ""]

    md += ["## Демонстрация: жанровые термины и синонимы", ""]
    titles = {d["doc_id"]: d["title"] for d in docs}
    demos = {}
    for text in DEMO_QUERIES:
        md += [f"### «{text}»", ""]
        for mode in ("synonyms", "hierarchy"):
            exp = expand(text, thesaurus, lookup, mode, best)
            hits = search(exp, index)[:5]
            demos[f"{text} / {mode}"] = {"expansion": exp, "top": hits}
            added = [e for e in exp["log"] if e["action"] == "добавлен"]
            status = "; ".join(f"«{d['term']}»: {d['status']} -> "
                               f"{d['chosen'] or [a['concept_id'] for a in d['alternatives']]}"
                               for d in exp["decisions"])
            md += [f"- {MODE_NAMES[mode]}: {status}",
                   f"  - добавлено терминов: {len(added)}; первые: "
                   + ", ".join(f"{e['term']} [{e['relation']}, w={e['weight']}]" for e in added[:8]),
                   "  - top-5: " + ", ".join(f"{h['doc_id']} {titles[h['doc_id']]}" for h in hits)]
        md.append("")
    print("\n".join(md))

    save_json({"thesaurus": counts, "best_params": best, "table": [dict(zip(header, t)) for t in table],
               "empty_answer": empty}, RESULTS_DIR / "metrics.json")
    save_json(grid, RESULTS_DIR / "tuning.json")
    save_json(rows, RESULTS_DIR / "per_query.json")
    unresolved = [{"query_id": r["query_id"], "query": r["text"], "term": d["term"], "alternatives": d["alternatives"]}
                  for r in rows["synonyms"] for d in r["expansion"]["decisions"]
                  if d["alternatives"] and not d["chosen"]]
    save_json({"unresolved_ambiguity": unresolved, "added_by_synonyms": added_syn, "added_by_hierarchy": added_hier,
               "wrong_generalization": wrong, "removed_by_hierarchy": removed_hier, "demos": demos},
              RESULTS_DIR / "analysis.json")
    save_json({lemma: index["postings"][lemma] for lemma in sorted(index["postings"])},
              TEMP_DIR / "index_postings.json")


if __name__ == "__main__":
    run()
