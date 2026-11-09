import random
import zlib

import numpy as np

PRIME = 2 ** 31 - 1  # простое число Мерсенна, a * x + b помещается в int64


def shingle_hash(shingle):
    """Устойчивый хеш шингла: crc32 от строки, одинаков при любом запуске."""
    data = " ".join(shingle).encode("utf-8")
    return zlib.crc32(data) % PRIME


class MinHasher:
    """Семейство универсальных хешей h(x) = (a * x + b) mod p с фиксированным seed."""

    def __init__(self, num_perm=128, seed=1):
        rng = random.Random(seed)
        self.num_perm = num_perm
        self.a = np.array([rng.randint(1, PRIME - 1) for _ in range(num_perm)], dtype=np.int64)
        self.b = np.array([rng.randint(0, PRIME - 1) for _ in range(num_perm)], dtype=np.int64)

    def signature(self, shingles):
        """Компонента i подписи - минимум i-й хеш-функции по всем шинглам."""
        if not shingles:
            return None
        x = np.array([shingle_hash(s) for s in shingles], dtype=np.int64)
        values = (self.a[:, None] * x[None, :] + self.b[:, None]) % PRIME
        return values.min(axis=1)


def estimate_jaccard(sig_a, sig_b):
    """Доля совпавших компонентов подписей. Пустая подпись даёт 0."""
    if sig_a is None or sig_b is None:
        return 0.0
    return float(np.mean(sig_a == sig_b))
