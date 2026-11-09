from task02_morphology import evaluate as ev
from task02_morphology.index import build_all, build_index, postings_for_save, vocabulary_size
from task02_morphology.normalizers import MODE_NAMES, MODES, normalize_word
from task02_morphology.search import search
from utils.corpus import load_corpus, load_queries, save_json
from utils.paths import ROOT_DIR
from utils.text import normalize

RESULTS_DIR = ROOT_DIR / "task02_morphology" / "results"
TEMP_DIR = ROOT_DIR / "task02_morphology" / "temp"
SERIES = {"без удаления стоп-слов": False, "с удалением стоп-слов": True}
SPECIAL_WORDS = ["Базаров", "Базарова", "базар", "базаре", "Сечи", "Сечь", "сеча", "очи", "лекаря",
                 "сей", "сия", "Мышкина", "Чичикова", "Муму", "роман", "романа", "романе", "мир", "мать"]
TAGS = ["character_name", "homonym", "negation", "genre"]


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def toy_example():
    docs = [{"doc_id": "D1", "title": "", "text": "роман"}, {"doc_id": "D2", "title": "", "text": "романа"}]
    rows = []
    for mode in MODES:
        found = [h["doc_id"] for h in search("роман", build_index(docs, mode))]
        rows.append([MODE_NAMES[mode], found])
    return md_table(["режим", "выдача по запросу «роман»"], rows)


def special_cases(indexes):
    rows = []
    for word in SPECIAL_WORDS:
        w = normalize(word)
        row = [word]
        for mode in MODES:
            term = normalize_word(w, mode)
            docs = indexes[mode]["postings"].get(term, [])
            shown = ", ".join(docs[:6]) + (" ..." if len(docs) > 6 else "")
            row.append(f"`{term}` df={len(docs)} [{shown}]")
        rows.append(row)
    return md_table(["слово"] + [MODE_NAMES[m] for m in MODES], rows)


def summary_rows(summaries):
    rows = []
    for (series, mode), s in summaries.items():
        rows.append([series, MODE_NAMES[mode], s["queries"], fmt(s["P@5"]), fmt(s["R@5"]), fmt(s["MRR@5"]),
                     fmt(s["nDCG@5"]), s["vocab"], fmt(s["oov_any"]), fmt(s["oov_all"]), fmt(s["avg_found"])])
    return rows


SUMMARY_HEADER = ["серия", "режим", "запросов", "P@5", "R@5", "MRR@5", "nDCG@5", "словарь",
                  "доля запросов с OOV-термином", "доля запросов без совпадений", "ср. размер выдачи"]


def forms_text(forms):
    return "; ".join(f"{t}<-{','.join(f)}" for t, f in forms.items())


def run():
    RESULTS_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    docs = load_corpus()
    queries = load_queries()
    print(f"Документов: {len(docs)}, запросов: {len(queries)}")

    indexes, rows = {}, {}
    for series, drop in SERIES.items():
        indexes[series] = build_all(docs, MODES, drop)
        for mode in MODES:
            rows[(series, mode)] = ev.evaluate_mode(queries, indexes[series][mode])

    summaries = {split: {} for split in ("tune", "test")}
    for (series, mode), r in rows.items():
        vocab = vocabulary_size(indexes[series][mode])
        for split in summaries:
            summaries[split][(series, mode)] = ev.summarize(r, split, vocab)

    best_key = max(summaries["tune"], key=lambda k: (summaries["tune"][k]["R@5"] + summaries["tune"][k]["P@5"]))
    best_test = summaries["test"][best_key]

    main_series = "с удалением стоп-слов"
    stop_effect = {mode: ev.stopword_effect(rows[(list(SERIES)[0], mode)], rows[(main_series, mode)])
                   for mode in MODES}
    empty = {f"{s} / {MODE_NAMES[m]}": ev.empty_answer_check(r) for (s, m), r in rows.items()}
    tags = []
    for (series, mode), r in rows.items():
        for tag in TAGS:
            t = ev.tag_summary(r, tag)
            tags.append([series, MODE_NAMES[mode], tag, t["queries"], fmt(t["P@5"]), fmt(t["R@5"])])

    test_queries = [q for q in queries if q["split"] == "test"]
    test_rows = {key: [x for x in r if x["split"] == "test"] for key, r in rows.items()}
    diffs = {}
    for target in ("stem", "lemma"):
        diffs[target] = ev.mode_diff(test_rows[(main_series, "form")], test_rows[(main_series, target)],
                                     test_queries, indexes[main_series][target])
    false = {mode: ev.false_matches(test_rows[(main_series, mode)], test_queries, indexes[main_series][mode])
             for mode in MODES}

    # ---------- отчёт ----------
    md = ["# Задача 02. Результаты", "", "## Пример из задания", "", toy_example(), ""]
    for split, title in (("tune", "Настроечные запросы Q01-Q10"), ("test", "Проверочные запросы Q11-Q30 (без Q29)")):
        md += [f"## {title}", "", md_table(SUMMARY_HEADER, summary_rows(summaries[split])), ""]
    md += ["## Выбор конфигурации на tune", "",
           f"Лучшая по P@5+R@5 на tune: {best_key[0]} / {MODE_NAMES[best_key[1]]}; "
           f"на test: P@5={best_test['P@5']:.3f}, R@5={best_test['R@5']:.3f}", ""]
    md += ["## Метрики по тегам запросов (все 30 запросов)", "",
           md_table(["серия", "режим", "тег", "запросов", "P@5", "R@5"], tags), ""]
    md += ["## Запрос без релевантных документов (Q29)", "",
           md_table(["конфигурация", "найдено", "пустой ответ?", "top-5"],
                    [[k, v[0]["found"], v[0]["correct_empty"], v[0]["top"]] for k, v in empty.items()]), ""]
    md += ["## Изменение top-5 при удалении стоп-слов", ""]
    for mode in MODES:
        md += [f"### {MODE_NAMES[mode]}: изменилось {len(stop_effect[mode])} из {len(queries)} запросов", "",
               md_table(["запрос", "ушли", "пришли", "P@5 до", "P@5 после"],
                        [[c["query_id"], c["removed"], c["added"], fmt(c["p5_before"]), fmt(c["p5_after"])]
                         for c in stop_effect[mode]]), ""]
    md += ["## Особенность варианта: архаизмы и имена персонажей", "", special_cases(indexes[main_series]), ""]
    for target in ("stem", "lemma"):
        md += [f"## Появившиеся/исчезнувшие документы: словоформы -> {MODE_NAMES[target]} (test, {main_series})",
               "", md_table(["запрос", "появились (релевантен?, совпавшие формы)", "исчезли (релевантен?)"],
                            [[d["query_id"],
                              "<br>".join(f"{a['doc_id']} ({'+' if a['relevant'] else '-'}) {forms_text(a['forms'])}"
                                          for a in d["appeared"]),
                              ", ".join(f"{x['doc_id']} ({'+' if x['relevant'] else '-'})" for x in d["disappeared"])]
                             for d in diffs[target]]), ""]
    for mode in MODES:
        md += [f"## Ложные совпадения в top-5: {MODE_NAMES[mode]} (test, {main_series})", "",
               md_table(["запрос", "документ", "совпавшие исходные формы"],
                        [[f["query_id"], f["doc_id"], forms_text(f["forms"])] for f in false[mode]]), ""]
    per_query = []
    for q in test_queries:
        row = [q["query_id"], q["text"]]
        for series in SERIES:
            for mode in MODES:
                r = next(x for x in test_rows[(series, mode)] if x["query_id"] == q["query_id"])
                row.append(f"{r['p5']:.1f}/{r['r5']:.2f}")
        per_query.append(row)
    md += ["## P@5/R@5 по проверочным запросам", "",
           md_table(["id", "запрос"] + [f"{'стоп' if d else 'все'}-{m}" for d in SERIES.values() for m in MODES],
                    per_query), ""]
    print("\n".join(md))

    save_json({split: {f"{s} / {m}": v for (s, m), v in table.items()} for split, table in summaries.items()},
              RESULTS_DIR / "metrics.json")
    save_json({f"{s} / {m}": r for (s, m), r in rows.items()}, RESULTS_DIR / "per_query.json")
    save_json({"stopword_effect": stop_effect, "mode_diff_test": diffs, "false_matches_test": false,
               "empty_answer": empty}, RESULTS_DIR / "analysis.json")
    for mode in MODES:
        save_json(postings_for_save(indexes[list(SERIES)[0]][mode]), TEMP_DIR / f"index_{mode}.json")

    print(f"\nЛучшая на tune: {best_key}, test P@5={best_test['P@5']:.3f} R@5={best_test['R@5']:.3f}")


if __name__ == "__main__":
    run()
