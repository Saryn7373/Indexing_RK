import gc
import statistics
import time

from task04_compression.intersect import intersect_compressed, intersect_two_pointers, intersect_with_skips
from task04_compression.skips import SkipList
from task04_compression.storage import decode_payload, read_record
from task04_compression.vbyte import decode_ids, decode_uint32, encode_ids, encode_uint32

REPEATS = 7
MIN_RUN_SECONDS = 0.05


def median_time(func, repeats=REPEATS):
    """Медиана времени одного вызова, мкс. Внутри замера вызов повторяется number раз, GC выключен."""
    start = time.perf_counter()
    func()
    single = time.perf_counter() - start
    number = max(1, int(MIN_RUN_SECONDS / max(single, 1e-9)))
    times = []
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(number):
                func()
            times.append((time.perf_counter() - start) / number)
    finally:
        gc.enable()
    return round(statistics.median(times) * 1e6, 2)


def size_report(lists):
    n_ids = sum(len(ids) for ids in lists.values())
    raw = 4 * n_ids
    vbyte = sum(len(encode_ids(ids)) for ids in lists.values())
    skip_lists = [SkipList.from_ids(ids) for ids in lists.values()]
    skip_table = sum(s.skip_bytes() for s in skip_lists)
    return {
        "lists": len(lists), "ids": n_ids,
        "uint32_bytes": raw, "vbyte_bytes": vbyte,
        "skip_table_bytes": skip_table, "vbyte_skips_bytes": vbyte + skip_table,
        "ratio_vbyte": round(raw / vbyte, 3) if vbyte else 0,
        "ratio_vbyte_skips": round(raw / (vbyte + skip_table), 3) if vbyte else 0,
    }


def decode_report(lists):
    encoded_u = [encode_uint32(ids) for ids in lists.values()]
    encoded_v = [encode_ids(ids) for ids in lists.values()]
    skip_lists = [SkipList.from_ids(ids) for ids in lists.values()]
    return {
        "uint32_us": median_time(lambda: [decode_uint32(x) for x in encoded_u]),
        "vbyte_us": median_time(lambda: [decode_ids(x) for x in encoded_v]),
        "vbyte_skips_us": median_time(lambda: [s.decode_all() for s in skip_lists]),
    }


def full_path(files, directories, kind, key_a, key_b):
    f = files[kind]
    count_a, payload_a = read_record(f, *directories[kind][key_a])
    count_b, payload_b = read_record(f, *directories[kind][key_b])
    a, b = decode_payload(kind, payload_a, count_a), decode_payload(kind, payload_b, count_b)
    if kind != "vbyte+skips":
        return intersect_two_pointers(a, b)
    short, long = (a, b) if a.count <= b.count else (b, a)
    return intersect_compressed(short.decode_all(), long)[0]


def pair_report(a, b, files, directories, key_a, key_b):
    # "уже декодированные" списки берём после декодирования uint32, как в полном пути
    a, b = decode_uint32(encode_uint32(a)), decode_uint32(encode_uint32(b))
    expected = sorted(set(a) & set(b))
    long_skips = SkipList.from_ids(a if len(a) >= len(b) else b)
    short = b if len(a) >= len(b) else a
    result, blocks = intersect_compressed(short, long_skips)
    checks = [intersect_two_pointers(a, b), intersect_with_skips(a, b), result]
    checks += [full_path(files, directories, kind, key_a, key_b) for kind in files]
    assert all(c == expected for c in checks), "результаты пересечения расходятся"
    report = {
        "len_a": len(a), "len_b": len(b), "common": len(expected),
        "blocks_decoded": blocks, "blocks_total": long_skips.n_blocks(),
        "decoded_two_pointers_us": median_time(lambda: intersect_two_pointers(a, b)),
        "decoded_skips_us": median_time(lambda: intersect_with_skips(a, b)),
    }
    for kind in files:
        report[f"full_{kind}_us"] = median_time(lambda: full_path(files, directories, kind, key_a, key_b))
    return report
