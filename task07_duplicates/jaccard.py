def jaccard(set_a, set_b):
    """Точный коэффициент Жаккара. Для двух пустых множеств возвращается 0, а не 1."""
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union


def containment(set_a, set_b):
    """Доля шинглов меньшего текста, найденных в большем (для разбора цитат)."""
    smaller = min(len(set_a), len(set_b))
    if smaller == 0:
        return 0.0
    return len(set_a & set_b) / smaller
