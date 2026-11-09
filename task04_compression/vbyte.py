"""Delta + Variable Byte.

Число делится на группы по 7 бит. Группы пишутся от старшей к младшей (big-endian).
В каждом байте 7 младших битов - данные. Старший бит = 1 только у последнего байта числа,
у остальных байтов он равен 0. Пример: 128 -> 0x01 0x80, 5 -> 0x85.
"""
import struct


class TruncatedError(ValueError):
    pass


def delta_encode(ids):
    deltas, prev = [], 0
    for i, x in enumerate(ids):
        if x < 0 or (i > 0 and x <= prev):
            raise ValueError("ID должны быть неотрицательными, отсортированными и без повторов")
        deltas.append(x - prev)
        prev = x
    return deltas


def delta_decode(deltas):
    ids, total = [], 0
    for d in deltas:
        total += d
        ids.append(total)
    return ids


def encode_number(n):
    groups = [n & 0x7F]
    n >>= 7
    while n:
        groups.append(n & 0x7F)
        n >>= 7
    groups.reverse()
    groups[-1] |= 0x80
    return bytes(groups)


def encode_numbers(numbers):
    return b"".join(encode_number(n) for n in numbers)


def decode_numbers(data, offset=0, count=None):
    """Читает числа с позиции offset. count=None - до конца данных. -> (числа, новая позиция)."""
    numbers, n, pos = [], 0, offset
    while pos < len(data) and (count is None or len(numbers) < count):
        byte = data[pos]
        pos += 1
        n = (n << 7) | (byte & 0x7F)
        if byte & 0x80:
            numbers.append(n)
            n = 0
        elif pos == len(data):
            raise TruncatedError(f"последовательность обрезана: байт {pos - 1} без признака конца числа")
    if count is not None and len(numbers) < count:
        raise TruncatedError(f"ожидалось {count} чисел, прочитано {len(numbers)}")
    return numbers, pos


def encode_ids(ids):
    return encode_numbers(delta_encode(ids))


def decode_ids(data):
    return delta_decode(decode_numbers(data)[0])


def encode_uint32(ids):
    """Эталон без сжатия: каждый ID - uint32 little-endian, 4 байта."""
    return struct.pack(f"<{len(ids)}I", *ids)


def decode_uint32(data):
    if len(data) % 4:
        raise TruncatedError("длина uint32-данных не кратна 4")
    return list(struct.unpack(f"<{len(data) // 4}I", data))
