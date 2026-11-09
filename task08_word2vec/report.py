def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt(v) for v in value)
    return str(value)


def md_table(rows, columns=None):
    if not rows:
        return "(пусто)\n"
    columns = columns or list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def ranking_str(ranking):
    return ", ".join(f"{d}:{s:.3f}" for d, s in ranking)
