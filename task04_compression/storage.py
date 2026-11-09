"""Одинаковый бинарный контейнер для всех представлений.

Запись списка: [uint32 число ID][uint32 длина payload в байтах][payload].
payload: uint32 - 4 байта на ID; vbyte - поток delta+VByte; skips - SkipList.to_bytes().
"""
import struct

from task04_compression.skips import SkipList
from task04_compression.vbyte import TruncatedError, decode_ids, decode_uint32, encode_ids, encode_uint32

RECORD = struct.Struct("<II")

ENCODERS = {
    "uint32": encode_uint32,
    "vbyte": encode_ids,
    "vbyte+skips": lambda ids: SkipList.from_ids(ids).to_bytes(),
}


def write_lists(path, lists, kind):
    """lists: {ключ: ids}. -> {ключ: (смещение записи, длина записи)}."""
    directory, offset = {}, 0
    with open(path, "wb") as f:
        for key, ids in lists.items():
            payload = ENCODERS[kind](ids)
            record = RECORD.pack(len(ids), len(payload)) + payload
            f.write(record)
            directory[key] = (offset, len(record))
            offset += len(record)
    return directory


def read_record(f, offset, length):
    f.seek(offset)
    raw = f.read(length)
    if len(raw) < RECORD.size:
        raise TruncatedError("запись обрезана")
    count, size = RECORD.unpack_from(raw, 0)
    payload = raw[RECORD.size:]
    if len(payload) != size:
        raise TruncatedError("payload обрезан")
    return count, payload


def decode_payload(kind, payload, count):
    if kind == "vbyte+skips":
        return SkipList.from_bytes(payload, count)
    ids = decode_uint32(payload) if kind == "uint32" else decode_ids(payload)
    if len(ids) != count:
        raise TruncatedError(f"ожидалось {count} ID, декодировано {len(ids)}")
    return ids
