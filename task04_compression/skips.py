"""Сжатый список с пропусками (skip pointers).

Правило: список из n ID делится на блоки по step = floor(sqrt(n)) элементов.
Для каждого блока, кроме нулевого, хранится запись (первый ID блока, смещение блока в байтах)
- два uint32, 8 байт. Это и есть точки возобновления декодирования: с найденного смещения
декодер стартует, зная первый ID блока, поэтому предыдущие блоки читать не нужно.
Списки короче SKIP_MIN_LEN хранятся без пропусков (один блок).
Шаг и число записей не хранятся: они вычисляются из числа ID.
"""
import math
import struct

from task04_compression.vbyte import TruncatedError, decode_numbers, encode_number

SKIP_MIN_LEN = 16
SKIP_ENTRY_BYTES = 8


def skip_step(n):
    return max(1, math.isqrt(n))


def block_step(count):
    """Шаг блоков и число записей пропусков однозначно выводятся из числа ID."""
    return skip_step(count) if count >= SKIP_MIN_LEN else max(1, count)


class SkipList:
    def __init__(self, count, step, skips, data):
        self.count = count
        self.step = step
        self.skips = skips     # [(первый ID блока, смещение)] для блоков 1, 2, ...
        self.data = data       # тот же поток delta+VByte, что и без пропусков

    @classmethod
    def from_ids(cls, ids):
        step = block_step(len(ids))
        parts, skips, offset, prev = [], [], 0, 0
        for i, x in enumerate(ids):
            if i > 0 and i % step == 0:
                skips.append((x, offset))
            encoded = encode_number(x - prev)
            parts.append(encoded)
            offset += len(encoded)
            prev = x
        return cls(len(ids), step, skips, b"".join(parts))

    def n_blocks(self):
        return len(self.skips) + 1 if self.count else 0

    def skip_bytes(self):
        return len(self.skips) * SKIP_ENTRY_BYTES

    def first_id(self, block):
        return self.skips[block - 1][0]

    def decode_block(self, block):
        start = block * self.step
        count = min(self.step, self.count - start)
        if block == 0:
            numbers, _ = decode_numbers(self.data, 0, count)
            total, ids = 0, []
            for d in numbers:
                total += d
                ids.append(total)
            return ids
        first, offset = self.skips[block - 1]
        numbers, _ = decode_numbers(self.data, offset, count)
        ids = [first]
        for d in numbers[1:]:
            ids.append(ids[-1] + d)
        return ids

    def decode_all(self):
        ids = []
        for block in range(self.n_blocks()):
            ids.extend(self.decode_block(block))
        return ids

    def to_bytes(self):
        """[n_skips пар uint32 (первый ID, смещение)][поток delta+VByte]. Число ID хранит контейнер."""
        table = b"".join(struct.pack("<II", first, offset) for first, offset in self.skips)
        return table + self.data

    @classmethod
    def from_bytes(cls, raw, count):
        step = block_step(count)
        n_skips = max(0, -(-count // step) - 1)
        table_end = n_skips * SKIP_ENTRY_BYTES
        if len(raw) < table_end:
            raise TruncatedError("таблица пропусков обрезана")
        skips = [struct.unpack_from("<II", raw, i * SKIP_ENTRY_BYTES) for i in range(n_skips)]
        return cls(count, step, skips, raw[table_end:])
