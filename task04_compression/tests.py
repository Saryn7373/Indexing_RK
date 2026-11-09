import random
import tempfile
import unittest
from pathlib import Path

from task04_compression.intersect import intersect_compressed, intersect_two_pointers, intersect_with_skips
from task04_compression.postings import build_chunks, build_postings, sorted_unique, split_tokens
from task04_compression.skips import SkipList
from task04_compression.storage import decode_payload, read_record, write_lists
from task04_compression.synthetic import make_pair
from task04_compression.vbyte import (TruncatedError, decode_ids, decode_numbers, decode_uint32, delta_encode,
                                      encode_ids, encode_number, encode_uint32)


class VByteTests(unittest.TestCase):
    def test_example_from_task(self):
        ids = [3, 10, 11, 25]
        self.assertEqual(delta_encode(ids), [3, 7, 1, 14])
        self.assertEqual(encode_ids(ids), bytes([0x83, 0x87, 0x81, 0x8E]))
        self.assertEqual(decode_ids(encode_ids(ids)), ids)

    def test_boundary_numbers(self):
        self.assertEqual(encode_number(0), bytes([0x80]))
        self.assertEqual(encode_number(127), bytes([0xFF]))
        self.assertEqual(encode_number(128), bytes([0x01, 0x80]))
        self.assertEqual(encode_number(16383), bytes([0x7F, 0xFF]))
        self.assertEqual(encode_number(16384), bytes([0x01, 0x00, 0x80]))
        self.assertEqual(len(encode_number(2 ** 32 - 1)), 5)
        for n in [0, 1, 127, 128, 129, 16383, 16384, 2 ** 32 - 1, 2 ** 40 + 5]:
            self.assertEqual(decode_numbers(encode_number(n))[0], [n])

    def test_round_trip(self):
        cases = [[], [0], [127], [128], [0, 127, 128], [5, 2 ** 31, 2 ** 32 - 1], [10 ** 12, 10 ** 12 + 1]]
        rng = random.Random(1)
        cases.append(sorted(rng.sample(range(10 ** 7), 5000)))
        for ids in cases:
            self.assertEqual(decode_ids(encode_ids(ids)), ids)
            skip_list = SkipList.from_ids(ids)
            self.assertEqual(skip_list.decode_all(), ids)
            self.assertEqual(SkipList.from_bytes(skip_list.to_bytes(), len(ids)).decode_all(), ids)
            if not ids or ids[-1] < 2 ** 32:
                self.assertEqual(decode_uint32(encode_uint32(ids)), ids)

    def test_truncated_input(self):
        data = encode_ids([3, 300, 100000])
        for cut in range(1, len(data)):
            prefix = data[:cut]
            if prefix[-1] & 0x80:
                self.assertLess(len(decode_ids(prefix)), 3)
            else:
                with self.assertRaises(TruncatedError):
                    decode_ids(prefix)
        with self.assertRaises(TruncatedError):
            decode_numbers(data[:-1] + bytes([0x80]), count=4)
        with self.assertRaises(TruncatedError):
            decode_uint32(encode_uint32([1, 2])[:-1])
        ids = list(range(0, 1000, 3))
        raw = SkipList.from_ids(ids).to_bytes()
        with self.assertRaises(TruncatedError):
            SkipList.from_bytes(raw[:20], len(ids))
        with self.assertRaises(TruncatedError):
            SkipList.from_bytes(raw[:-1], len(ids)).decode_all()

    def test_unsorted_rejected(self):
        for bad in [[3, 2], [1, 1], [-1]]:
            with self.assertRaises(ValueError):
                encode_ids(bad)
        self.assertEqual(sorted_unique([5, 3, 5, 1]), [1, 3, 5])


class SkipTests(unittest.TestCase):
    def test_block_layout(self):
        ids = list(range(0, 300, 3))           # 100 ID -> шаг 10, 9 записей пропусков
        s = SkipList.from_ids(ids)
        self.assertEqual(s.step, 10)
        self.assertEqual(len(s.skips), 9)
        self.assertEqual(s.skip_bytes(), 72)
        self.assertEqual(s.decode_block(3), ids[30:40])
        self.assertEqual(SkipList.from_ids(list(range(10))).skips, [])

    def test_intersections_agree(self):
        rng = random.Random(3)
        for _ in range(200):
            a = sorted(rng.sample(range(2000), rng.randint(0, 400)))
            b = sorted(rng.sample(range(2000), rng.randint(0, 400)))
            expected = sorted(set(a) & set(b))
            self.assertEqual(intersect_two_pointers(a, b), expected)
            self.assertEqual(intersect_with_skips(a, b), expected)
            short, long = (a, b) if len(a) <= len(b) else (b, a)
            self.assertEqual(intersect_compressed(short, SkipList.from_ids(long))[0], expected)

    def test_synthetic_density(self):
        a, b, common = make_pair(1000, 5000, 0.3, seed=1, universe=100_000)
        self.assertEqual((len(a), len(b), common), (1000, 5000, 300))
        self.assertEqual(len(set(a) & set(b)), 300)
        self.assertEqual(a, sorted(set(a)))


class StorageAndCorpusTests(unittest.TestCase):
    def test_storage_round_trip(self):
        lists = {"x": [1, 2, 3], "empty": [], "big": [0, 128, 2 ** 32 - 1]}
        with tempfile.TemporaryDirectory() as folder:
            for kind in ["uint32", "vbyte", "vbyte+skips"]:
                path = Path(folder) / f"{kind}.bin"
                directory = write_lists(path, lists, kind)
                with open(path, "rb") as f:
                    for key, ids in lists.items():
                        count, payload = read_record(f, *directory[key])
                        decoded = decode_payload(kind, payload, count)
                        if kind == "vbyte+skips":
                            decoded = decoded.decode_all()
                        self.assertEqual((count, decoded), (len(ids), ids))

    def test_chunks(self):
        self.assertEqual([len(c) for c in split_tokens(list(range(650)))], [300, 350])
        self.assertEqual([len(c) for c in split_tokens(list(range(50)))], [50])
        chunks = build_chunks()
        self.assertGreaterEqual(len(chunks), 200)
        postings = build_postings(chunks)
        for ids in postings.values():
            self.assertEqual(decode_ids(encode_ids(ids)), ids)


if __name__ == "__main__":
    unittest.main()
