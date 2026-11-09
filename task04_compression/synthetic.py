"""Синтетические отсортированные списки с заданной плотностью пересечения.

Плотность = |A ∩ B| / min(|A|, |B|).
"""
import random

UNIVERSE = 1_000_000

SCENARIOS = [
    # (название, |A|, |B|, плотность)
    ("равные, редкое пересечение", 100_000, 100_000, 0.01),
    ("равные, половина общих", 100_000, 100_000, 0.5),
    ("равные, почти совпадают", 100_000, 100_000, 0.9),
    ("10k и 100k", 10_000, 100_000, 0.1),
    ("1k и 100k, редкое", 1_000, 100_000, 0.01),
    ("1k и 100k, половина", 1_000, 100_000, 0.5),
    ("100 и 100k", 100, 100_000, 0.5),
]


def make_pair(size_a, size_b, density, seed, universe=UNIVERSE):
    rng = random.Random(seed)
    common = round(density * min(size_a, size_b))
    pool = rng.sample(range(universe), size_a + size_b - common)
    shared = pool[:common]
    only_a = pool[common:size_a]
    only_b = pool[size_a:]
    return sorted(shared + only_a), sorted(shared + only_b), common
