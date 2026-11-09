import argparse
import platform
import statistics
import time
from pathlib import Path

from task03_positional_index.evaluator import Evaluator
from task03_positional_index.index import PositionalIndex
from task03_positional_index.lexer import QuerySyntaxError
from task03_positional_index.parser import parse_query, show_tree
from task03_positional_index.querygen import generate_queries
from task03_positional_index.scan import scan_search
from utils.corpus import load_corpus, load_json, save_json
from utils.paths import DATA_DIR
from utils.text import normalize

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TEMP_DIR = Path(__file__).resolve().parent / "temp"
REPEATS = 5
SEED = 42

VARIANT_QUERIES = [
    '"мертвые души"',
    "мертвые AND души",
    "мертвые NEAR/3 души",
    '"война и мир"',
    "война AND мир",
    '"преступление и наказание"',
    "преступление AND наказание",
    '"герой нашего времени"',
    '"все счастливые семьи похожи друг на друга"',
    "счастливые AND семьи AND похожи",
    '"мне отмщение и аз воздам"',
]


def median_time(func, repeats=REPEATS):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def print_table(headers, rows):
    widths = [max(len(str(x)) for x in [h] + [r[i] for r in rows]) for i, h in enumerate(headers)]
    print(" | ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(x).ljust(w) for x, w in zip(row, widths)))
    print()


def run_query(evaluator, query):
    try:
        return evaluator.search(query)
    except QuerySyntaxError as e:
        return f"ошибка: {e}"


def small_corpus_checks():
    small = load_json(DATA_DIR / "small_corpus.json")
    docs = small["documents"]
    evaluator = Evaluator(PositionalIndex.build(docs))
    rows, report = [], []
    for check in small["queries"]:
        got_index = run_query(evaluator, check["query"])
        try:
            got_scan = scan_search(docs, check["query"])
        except QuerySyntaxError as e:
            got_scan = f"ошибка: {e}"
        if check.get("error"):
            ok = isinstance(got_index, str) and isinstance(got_scan, str)
        else:
            ok = got_index == check["expected"] == got_scan
        rows.append([check["id"], check["query"] or "<пусто>", str(got_index)[:60], "OK" if ok else "FAIL"])
        report.append({**check, "index": got_index, "scan": got_scan, "ok": ok})
    print(f"1. Маленький корпус: {len(rows)} проверок (индекс и просмотр)")
    print_table(["#", "запрос", "выдача индекса", "итог"], rows)
    return report


def random_comparison(docs, index):
    evaluator = Evaluator(index)
    queries = generate_queries(docs, 300, seed=SEED)
    mismatches = [q for q in queries if evaluator.search(q) != scan_search(docs, q)]
    non_empty = sum(1 for q in queries if evaluator.search(q))
    timing_queries = queries[:30]
    t_index = median_time(lambda: [evaluator.search(q) for q in timing_queries])
    t_scan = median_time(lambda: [scan_search(docs, q) for q in timing_queries])
    result = {"queries": len(queries), "non_empty": non_empty, "mismatches": mismatches,
              "timing_queries": len(timing_queries), "repeats": REPEATS,
              "index_ms_per_query": round(t_index / len(timing_queries) * 1000, 3),
              "scan_ms_per_query": round(t_scan / len(timing_queries) * 1000, 3)}
    print("3. Случайные запросы: индекс против последовательного просмотра")
    print_table(["запросов", "непустых", "расхождений", "индекс, мс/запрос", "просмотр, мс/запрос"],
                [[result["queries"], non_empty, len(mismatches),
                  result["index_ms_per_query"], result["scan_ms_per_query"]]])
    print("Примеры запросов:", *queries[:3], sep="\n  ")
    print()
    return result


def variant_case(docs, index):
    titles = {d["doc_id"]: d["title"] for d in load_corpus()}
    evaluator = Evaluator(index)
    rows, report = [], []
    for query in VARIANT_QUERIES:
        found = evaluator.search(query)
        assert found == scan_search(docs, query)
        names = ", ".join(f"{d} {titles[d]}" for d in found[:4]) + (" ..." if len(found) > 4 else "")
        rows.append([query, len(found), names])
        report.append({"query": query, "found": found})
    print("4. Особенность варианта: фразы из названий и цитат (весь корпус)")
    print_table(["запрос", "док.", "первые документы"], rows)
    return report


def update_demo(docs, index):
    docs = dict(docs)
    term = "мертвые"
    before = index.positions_of(term, "D001")
    new_text = docs["D001"] + " Мёртвые души — поэма Гоголя."
    start = time.perf_counter()
    index.update_document("D001", new_text)
    index.remove_document("D160")
    index.add_document("D161", "Новый документ: «Мёртвые души» в фонде библиотеки.")
    incremental = time.perf_counter() - start
    docs["D001"] = new_text
    del docs["D160"]
    docs["D161"] = "Новый документ: «Мёртвые души» в фонде библиотеки."
    rebuild = median_time(lambda: PositionalIndex.build(docs), repeats=3)
    rebuilt = PositionalIndex.build(docs)
    evaluator = Evaluator(index)
    queries = generate_queries(docs, 100, seed=SEED + 1) + ['"мертвые души"', "NOT души"]
    same_results = all(evaluator.search(q) == scan_search(docs, q) for q in queries)
    result = {
        "positions_before": {term: before}, "positions_after": {term: index.positions_of(term, "D001")},
        "universe_size": len(index.universe), "D160_in_universe": "D160" in index.universe,
        "equal_to_rebuild": index.to_dict() == rebuilt.to_dict(), "sorted": index.is_sorted(),
        "queries_equal_to_scan": same_results,
        "incremental_ms": round(incremental * 1000, 2), "rebuild_ms": round(rebuild * 1000, 2),
    }
    print("5. Изменение D001, удаление D160, добавление D161 без полной перестройки")
    print(f"  позиции '{term}' в D001 до: {before}, после: {result['positions_after'][term]}")
    print(f"  универсум: {result['universe_size']} док., D160 в универсуме: {result['D160_in_universe']}")
    print(f"  совпадает с полной перестройкой: {result['equal_to_rebuild']}, списки упорядочены: {result['sorted']}")
    print(f"  102 запроса совпали с просмотром: {same_results}")
    print(f"  время: инкрементально {result['incremental_ms']} мс, полная перестройка {result['rebuild_ms']} мс")
    print()
    return result


def run_experiments():
    RESULTS_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    report = {"hardware": {"platform": platform.platform(), "processor": platform.processor(),
                           "python": platform.python_version()}, "seed": SEED, "repeats": REPEATS}
    report["small_corpus"] = small_corpus_checks()

    docs = {d["doc_id"]: d["text"] for d in load_corpus()}
    build_time = median_time(lambda: PositionalIndex.build(docs), repeats=3)
    index = PositionalIndex.build(docs)
    index.save(TEMP_DIR / "index_corpus.json")
    stats = index.stats()
    report["corpus_index"] = {**stats, "build_ms": round(build_time * 1000, 1)}
    print("2. Индекс корпуса (сохранён в temp/index_corpus.json)")
    print_table(["документов", "терминов", "пар термин-документ", "позиций", "построение, мс"],
                [[stats["documents"], stats["terms"], stats["postings"], stats["positions"],
                  report["corpus_index"]["build_ms"]]])

    report["random_queries"] = random_comparison(docs, index)
    report["variant_case"] = variant_case(docs, index)
    report["update"] = update_demo(docs, index)
    save_json(report, RESULTS_DIR / "results.json")
    print("Результаты сохранены в", RESULTS_DIR / "results.json")


def query_mode(args):
    if args.small:
        docs = load_json(DATA_DIR / "small_corpus.json")["documents"]
    else:
        docs = {d["doc_id"]: d["text"] for d in load_corpus()}
    index = PositionalIndex.build(docs)
    if args.positions:
        term = normalize(args.positions)
        for d in index.docs_with(term):
            print(f"{term} -> {d}: {index.positions_of(term, d)}")
    if args.query is None:
        return
    try:
        tree = parse_query(args.query)
    except QuerySyntaxError as e:
        print("Синтаксическая ошибка:", e)
        return
    if args.tree:
        print(show_tree(tree))
    result = Evaluator(index).evaluate(tree)
    print(f"Найдено: {len(result)}")
    print(result)


def main():
    parser = argparse.ArgumentParser(description="Позиционный индекс и язык запросов")
    parser.add_argument("--query", help='запрос, например: "мёртвые души" AND NOT гоголь')
    parser.add_argument("--tree", action="store_true", help="напечатать дерево выражения")
    parser.add_argument("--small", action="store_true", help="искать в маленьком корпусе")
    parser.add_argument("--positions", help="показать списки позиций термина")
    args = parser.parse_args()
    if args.query is not None or args.positions:
        query_mode(args)
    else:
        run_experiments()


if __name__ == "__main__":
    main()
