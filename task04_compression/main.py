import os
import platform
import sys
import tempfile
from pathlib import Path

from task04_compression.benchmark import REPEATS, decode_report, pair_report, size_report
from task04_compression.postings import CHUNK_SIZE, MIN_TAIL, build_chunks, build_postings
from task04_compression.skips import SKIP_MIN_LEN
from task04_compression.storage import ENCODERS, write_lists
from task04_compression.synthetic import SCENARIOS, UNIVERSE, make_pair
from task04_compression.vbyte import decode_ids, delta_encode, encode_ids
from utils.corpus import save_json

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TEMP_DIR = Path(__file__).resolve().parent / "temp"
SEED = 42
FREQUENT_WORDS = ["и", "в", "не", "на", "что", "он"]
CORPUS_PAIRS = [("и", "не"), ("и", "в"), ("не", "на"), ("и", "чичиков"), ("в", "каренина"),
                ("не", "гоголь"), ("душа", "любовь"), ("чичиков", "гоголь")]
TIMING_NOTE = f"медиана из {REPEATS} замеров"


def hardware():
    return {"platform": platform.platform(), "processor": platform.processor(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0]}


def print_table(title, headers, rows):
    print(title)
    widths = [max(len(str(x)) for x in [h] + [r[i] for r in rows]) for i, h in enumerate(headers)]
    print(" | ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(x).ljust(w) for x, w in zip(row, widths)))
    print()


def example():
    ids = [3, 10, 11, 25]
    data = encode_ids(ids)
    print("Пример: ID", ids, "-> delta", delta_encode(ids),
          "-> байты", " ".join(f"{b:08b}" for b in data), "->", decode_ids(data))
    big = encode_ids([128, 300 + 128])
    print("Пример: ID [128, 428] -> delta [128, 300] -> байты", " ".join(f"0x{b:02X}" for b in big))
    print()
    return {"ids": ids, "deltas": delta_encode(ids), "bytes_hex": data.hex(" ")}


def open_files(paths):
    return {kind: open(path, "rb") for kind, path in paths.items()}


def write_all(lists, folder, prefix):
    paths = {kind: Path(folder) / f"{prefix}_{kind.replace('+', '_')}.bin" for kind in ENCODERS}
    directories = {kind: write_lists(paths[kind], lists, kind) for kind in ENCODERS}
    return paths, directories


def pair_rows(reports):
    rows = []
    for name, r in reports.items():
        rows.append([name, r["len_a"], r["len_b"], r["common"], f"{r['blocks_decoded']}/{r['blocks_total']}",
                     r["decoded_two_pointers_us"], r["decoded_skips_us"],
                     r["full_uint32_us"], r["full_vbyte_us"], r["full_vbyte+skips_us"]])
    return rows


PAIR_HEADERS = ["пара", "|A|", "|B|", "общих", "блоков декод.", "2 указателя", "с пропусками",
                "полный uint32", "полный vbyte", "полный vbyte+skips"]


def corpus_experiment():
    chunks = build_chunks()
    postings = build_postings(chunks)
    n_docs = len({doc_id for _, doc_id, _ in chunks})
    print(f"Корпус: {n_docs} исходных документов -> {len(chunks)} фрагментов по {CHUNK_SIZE} токенов, "
          f"{len(postings)} терминов\n")
    for word in ["чичиков", "гоголь"]:
        ids = postings[word]
        print(f"Список '{word}': ID {ids} -> delta {delta_encode(ids)} -> байты {encode_ids(ids).hex(' ')}")
    print()
    sizes = size_report(postings)
    print_table("Размер всего индекса (payload, байт)",
                ["списков", "ID", "uint32", "vbyte", "таблица пропусков", "vbyte+skips",
                 "сжатие vbyte", "сжатие vbyte+skips"],
                [[sizes["lists"], sizes["ids"], sizes["uint32_bytes"], sizes["vbyte_bytes"],
                  sizes["skip_table_bytes"], sizes["vbyte_skips_bytes"],
                  sizes["ratio_vbyte"], sizes["ratio_vbyte_skips"]]])

    paths, directories = write_all(postings, TEMP_DIR, "corpus")
    save_json(list(postings), TEMP_DIR / "corpus_terms.json")   # порядок записей в .bin-файлах
    file_sizes = {kind: path.stat().st_size for kind, path in paths.items()}
    print_table("Размер файлов в одинаковом контейнере (байт)", list(file_sizes),
                [list(file_sizes.values())])

    decoding = decode_report(postings)
    print_table(f"Декодирование всех списков, мкс ({TIMING_NOTE})", list(decoding),
                [list(decoding.values())])

    frequent = []
    for word in FREQUENT_WORDS:
        ids = postings[word]
        s = size_report({word: ids})
        deltas = delta_encode(ids)
        frequent.append({"word": word, "df": len(ids), "share": round(len(ids) / len(chunks), 3),
                         "delta_1_share": round(sum(1 for d in deltas[1:] if d == 1) / max(1, len(deltas) - 1), 3),
                         **s})
    print_table("Особенность варианта: очень частые слова",
                ["слово", "df", "доля фрагментов", "доля delta=1", "uint32", "vbyte", "пропуски", "сжатие",
                 "сжатие с пропусками"],
                [[f["word"], f["df"], f["share"], f["delta_1_share"], f["uint32_bytes"], f["vbyte_bytes"],
                  f["skip_table_bytes"], f["ratio_vbyte"], f["ratio_vbyte_skips"]]
                 for f in frequent])

    files = open_files(paths)
    pairs = {}
    try:
        for a, b in CORPUS_PAIRS:
            pairs[f"{a} & {b}"] = pair_report(postings[a], postings[b], files, directories, a, b)
    finally:
        for f in files.values():
            f.close()
    print_table(f"Пересечение на корпусе, мкс ({TIMING_NOTE})", PAIR_HEADERS, pair_rows(pairs))
    return {"documents": n_docs, "chunks": len(chunks), "terms": len(postings),
            "sizes": sizes, "file_sizes": file_sizes, "decoding_us": decoding,
            "frequent_words": frequent, "pairs": pairs}


def synthetic_experiment():
    reports, size_rows, sizes_all = {}, [], {}
    with tempfile.TemporaryDirectory() as folder:
        for i, (name, size_a, size_b, density) in enumerate(SCENARIOS):
            a, b, common = make_pair(size_a, size_b, density, seed=SEED + i)
            assert len(set(a) & set(b)) == common
            lists = {"A": a, "B": b}
            paths, directories = write_all(lists, folder, f"syn{i}")
            files = open_files(paths)
            try:
                reports[name] = pair_report(a, b, files, directories, "A", "B")
            finally:
                for f in files.values():
                    f.close()
            reports[name]["density"] = density
            sizes = size_report(lists)
            sizes["decoding_us"] = decode_report(lists)
            sizes_all[name] = sizes
            size_rows.append([name, sizes["uint32_bytes"], sizes["vbyte_bytes"],
                              sizes["skip_table_bytes"],
                              sizes["ratio_vbyte"], sizes["ratio_vbyte_skips"],
                              sizes["decoding_us"]["uint32_us"], sizes["decoding_us"]["vbyte_us"]])
    print(f"Синтетика: ID из [0, {UNIVERSE}), seed={SEED}+номер сценария\n")
    print_table("Синтетика: размер (A+B, байт) и декодирование A+B (мкс)",
                ["сценарий", "uint32", "vbyte", "пропуски", "сжатие", "сжатие с пропусками",
                 "декод. uint32", "декод. vbyte"], size_rows)
    print_table(f"Синтетика: пересечение, мкс ({TIMING_NOTE})", PAIR_HEADERS, pair_rows(reports))
    return {"universe": UNIVERSE, "pairs": reports, "sizes": sizes_all}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    info = hardware()
    print("Оборудование:", info)
    print(f"Параметры: CHUNK_SIZE={CHUNK_SIZE}, MIN_TAIL={MIN_TAIL}, шаг пропусков floor(sqrt(n)), "
          f"SKIP_MIN_LEN={SKIP_MIN_LEN}, повторов={REPEATS}\n")
    report = {"hardware": info, "seed": SEED, "repeats": REPEATS,
              "params": {"chunk_size": CHUNK_SIZE, "min_tail": MIN_TAIL, "skip_min_len": SKIP_MIN_LEN,
                         "skip_step": "floor(sqrt(n))", "uint32": "little-endian, 4 bytes"},
              "example": example()}
    report["corpus"] = corpus_experiment()
    report["synthetic"] = synthetic_experiment()
    save_json(report, RESULTS_DIR / "results.json")
    print("Результаты сохранены в", RESULTS_DIR / "results.json")


if __name__ == "__main__":
    main()
