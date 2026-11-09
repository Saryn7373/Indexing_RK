from task04_compression.skips import skip_step


def intersect_two_pointers(a, b):
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def intersect_with_skips(a, b):
    """Декодированные списки; пропуск из позиции i (кратной шагу) ведёт в i + step."""
    step_a, step_b = skip_step(len(a)), skip_step(len(b))
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            if i % step_a == 0 and i + step_a < len(a) and a[i + step_a] <= b[j]:
                i += step_a
            else:
                i += 1
        else:
            if j % step_b == 0 and j + step_b < len(b) and b[j + step_b] <= a[i]:
                j += step_b
            else:
                j += 1
    return result


def intersect_compressed(short_ids, long_list):
    """short_ids - декодированный короткий список, long_list - SkipList.

    Блоки длинного списка декодируются только там, куда попадает очередной ID.
    -> (пересечение, число декодированных блоков).
    """
    result, decoded = [], 0
    n_blocks = long_list.n_blocks()
    if n_blocks == 0:
        return result, decoded
    block, current, values, k = 0, -1, [], 0
    for x in short_ids:
        while block + 1 < n_blocks and long_list.first_id(block + 1) <= x:
            block += 1
        if block != current:
            values, k, current = long_list.decode_block(block), 0, block
            decoded += 1
        while k < len(values) and values[k] < x:
            k += 1
        if k < len(values) and values[k] == x:
            result.append(x)
    return result, decoded
