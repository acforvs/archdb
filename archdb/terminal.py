"""Plain terminal output; machine exports keep the original values."""

from __future__ import annotations

import json
import shutil
import textwrap
import unicodedata


def clean(text) -> str:
    """Replace terminal controls in display text."""
    return "".join(
        c if c in "\n\t" or not unicodedata.category(c).startswith("C") else "�" for c in str(text)
    )


def display(value, spec: dict | None = None, status: str = "unknown") -> str:
    if value is None:
        return "not disclosed" if status == "not-disclosed" else "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (dict, list)):
        return clean(json.dumps(value, ensure_ascii=False))
    if isinstance(value, (int, float)):
        unit = (spec or {}).get("unit")
        if unit == "bytes":
            for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
                if abs(value) < 1024 or suffix == "TiB":
                    return f"{value:.4g} {suffix}"
                value /= 1024
        if unit in {"params", "tokens"}:
            for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
                if abs(value) >= scale:
                    return f"{value / scale:.4g}{suffix}"
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.5g}"
    return clean(value)


def table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("No matches.")
        return
    cells = [[clean(v).replace("\t", " ") for v in row] for row in [headers, *rows]]
    widths = [
        min(46, max(len(line) for row in cells for line in row[i].splitlines() or [""]))
        for i in range(len(headers))
    ]
    available = shutil.get_terminal_size((120, 24)).columns
    while sum(widths) + 2 * (len(widths) - 1) > available and max(widths) > 12:
        index = widths.index(max(widths))
        widths[index] -= 1
    for index, row in enumerate(cells):
        wrapped = [
            [part for line in cell.splitlines() or [""] for part in textwrap.wrap(line, width=width) or [""]]
            for cell, width in zip(row, widths, strict=True)
        ]
        for n in range(max(map(len, wrapped))):
            print(
                "  ".join(
                    (parts[n] if n < len(parts) else "").ljust(width)
                    for parts, width in zip(wrapped, widths, strict=True)
                ).rstrip()
            )
        if index == 0:
            print("  ".join("─" * width for width in widths))
