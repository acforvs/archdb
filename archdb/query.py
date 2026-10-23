"""Read and query the JSON export without loading the authoring tools."""

from __future__ import annotations

import difflib
import json
import math
import operator
import re
from datetime import date
from pathlib import Path

from .derive import derive

ATTRIBUTES = {
    "id": "Model ID",
    "name": "Model",
    "org": "Organization",
    "country": "Country",
    "kind": "Checkpoint",
    "weights": "Weights",
}
STATUSES = {
    "known",
    "unknown",
    "not-disclosed",
    "reported",
    "config",
    "code",
    "inferred",
    "derived",
    "metadata",
}
SUFFIXES = {
    "": 1,
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
    "x": 1,
    "×": 1,
    "%": 0.01,
}
NUMBER = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*([a-z%×]*)", re.IGNORECASE)
RULE = re.compile(
    r"^([\w.]+)\s*(not\s+contains\b|contains\b|is\b|>=|<=|!=|==|=|>|<|~)\s*(.*?)\s*$", re.IGNORECASE
)
COMPARISONS = {
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


class QueryError(ValueError):
    pass


def number(value) -> float | None:
    """Exact numeric literals, SI/binary suffixes, percentages and a:b ratios."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", "")
    if text.count(":") == 1:
        left, right = (number(part) for part in text.split(":"))
        return left / right if left is not None and right not in (None, 0) else None
    match = NUMBER.fullmatch(text)
    if not match or match[2].lower() not in SUFFIXES:
        return None
    result = float(match[1]) * SUFFIXES[match[2].lower()]
    return result if math.isfinite(result) else None


def positive_integer(text: str) -> int:
    value = number(text)
    if value is None or value <= 0 or not value.is_integer():
        raise ValueError(f"expected a positive integer, got {text!r}")
    return int(value)


def load(path: Path | None = None, *, context: int = 131_072, precision: str = "bf16") -> dict:
    if path is None:
        packaged = Path(__file__).with_name("_data") / "archdb.json"
        path = packaged if packaged.exists() else Path(__file__).resolve().parents[1] / "exports/archdb.json"
    try:
        db = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read dataset {path}: {exc}") from exc
    if (
        not isinstance(db, dict)
        or not isinstance(db.get("fields"), dict)
        or not isinstance(db.get("models"), list)
    ):
        raise QueryError(f"{path} is not an ArchDB JSON export")
    if context <= 0 or precision not in {"bf16", "fp8", "fp4", "sourced"}:
        raise ValueError("invalid KV context or precision")
    for model in db["models"]:
        if not isinstance(model, dict) or not all(
            k in model for k in ("id", "name", "facts", "sources", "derived", "not_disclosed")
        ):
            raise ValueError(f"{path} contains an incomplete model record")
        # Use the same Python formulas as the exports, preserving checkpoint-derived counts.
        calculated = derive(model["facts"], context=context, kv_dtype=precision)
        model["derived"].update({k: v for k, v in calculated.items() if k.startswith("kv.")})
    db["_kv"] = {"context": context, "precision": precision}
    return db


def check_field(db: dict, key: str) -> None:
    if key in db["fields"] or key in ATTRIBUTES:
        return
    close = difflib.get_close_matches(key, [*db["fields"], *ATTRIBUTES], n=3)
    hint = f" Did you mean {', '.join(close)}?" if close else " Run `archdb fields` to list fields."
    raise ValueError(f"unknown field {key!r}.{hint}")


def cell(model: dict, key: str) -> dict:
    if key in ATTRIBUTES:
        value = model.get(key)
        return {"value": value, "status": "metadata" if value is not None else "unknown"}
    if key in model["facts"]:
        fact = model["facts"][key]
        source = model["sources"][fact["source"]]
        status = "inferred" if fact.get("basis") == "inferred" else source["kind"]
        if status not in {"config", "code", "inferred"}:
            status = "reported"
        return {**fact, "status": status if fact["value"] is not None else "unknown", "source": source}
    if key in model["derived"]:
        derived = model["derived"][key]
        return {**derived, "status": "derived" if derived["value"] is not None else "unknown"}
    return {"value": None, "status": "not-disclosed" if key in model["not_disclosed"] else "unknown"}


def resolve(db: dict, text: str) -> dict:
    models = db["models"]
    exact = [m for m in models if text.casefold() in (m["id"].casefold(), m["name"].casefold())]
    matches = exact or search(models, text)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"no model matches {text!r}. Run `archdb models` to list model IDs.")
    raise ValueError(
        f"{text!r} matches several models: {', '.join(m['id'] for m in matches)}. Use an exact ID."
    )


def search(models: list[dict], text: str) -> list[dict]:
    terms = text.casefold().split()
    return [
        m
        for m in models
        if all(t in " ".join(str(m.get(k, "")) for k in ATTRIBUTES).casefold() for t in terms)
    ]


def _date(text) -> int | None:
    try:
        parts = str(text).split("-")
        return date(*[int(v) for v in parts], *([1] * (3 - len(parts)))).toordinal()
    except (ValueError, TypeError):
        return None


def _bounded(value: str, op: str, target: float) -> bool:
    match = re.fullmatch(r"([<>≥≤~])\s*(.+)", value)
    boundary = number(match[2]) if match else None
    if boundary is None or match[1] == "~":
        return False
    sign = {"≥": ">=", "≤": "<="}.get(match[1], match[1])
    lower, upper = (boundary, math.inf) if sign.startswith(">") else (-math.inf, boundary)
    if op == ">":
        return lower > target or (lower == target and sign == ">")
    if op == ">=":
        return lower >= target
    if op == "<":
        return upper < target or (upper == target and sign == "<")
    if op == "<=":
        return upper <= target
    if op == "!=":
        return target < lower or target > upper or (target == boundary and sign in {">", "<"})
    return False


def predicate(db: dict, expression: str):
    match = RULE.fullmatch(expression.strip())
    if not match or not match[3]:
        raise ValueError(
            f"invalid filter {expression!r}; use e.g. 'params.total >= 400B' or 'mtp.depth is known'"
        )
    key, op, text = match.groups()
    check_field(db, key)
    op = " ".join(op.lower().split())
    if text[:1] in {"'", '"'} and text[-1:] == text[:1]:
        text = text[1:-1]
    spec = db["fields"].get(key, {"type": "str"})
    if op == "is":
        status = text.lower().replace(" ", "-")
        if status not in STATUSES:
            raise ValueError(f"unknown status {text!r}; choose from {', '.join(sorted(STATUSES))}")
        return lambda m: (
            (cell(m, key)["value"] is not None) if status == "known" else cell(m, key)["status"] == status
        )
    if op in {"contains", "not contains", "~"}:

        def contains(model):
            value = cell(model, key)["value"]
            if value is None:
                return False
            haystack = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            hit = text.casefold() in haystack.casefold()
            return not hit if op == "not contains" else hit

        return contains
    numeric = spec["type"] in {"int", "float"} or spec.get("unit") == "ratio"
    target = number(text) if numeric else _date(text) if spec["type"] == "date" else text.casefold()
    if spec["type"] == "bool":
        if text.lower() not in {"true", "false"} or op not in {"=", "==", "!="}:
            raise ValueError(f"{key} expects true or false with = or !=")
        target = text.lower() == "true"
    elif target is None:
        raise ValueError(f"{key} expects a {'date' if spec['type'] == 'date' else 'number'}, got {text!r}")
    elif not numeric and spec["type"] != "date" and op not in {"=", "==", "!="}:
        raise ValueError(f"{key} is text; use =, != or contains")

    def test(model):
        value = cell(model, key)["value"]
        if value is None:
            return False
        if numeric:
            parsed = number(value)
            return COMPARISONS[op](parsed, target) if parsed is not None else _bounded(str(value), op, target)
        if spec["type"] == "date":
            value = _date(value)
            return value is not None and COMPARISONS[op](value, target)
        return COMPARISONS[op](value if isinstance(value, bool) else str(value).casefold(), target)

    return test


def select(
    db: dict, text: str = "", filters: list[str] = (), sort: str = "name", descending: bool = False
) -> list[dict]:
    check_field(db, sort)
    tests = [predicate(db, f) for f in filters]
    models = [m for m in search(db["models"], text) if all(test(m) for test in tests)]

    def sort_key(model):
        value = cell(model, sort)["value"]
        numeric = number(value)
        if numeric is None and isinstance(value, str):
            numeric = number(re.sub(r"^[~<>≥≤]\s*", "", value))
        return (0, numeric) if numeric is not None else (1, str(value).casefold())

    known = [m for m in models if cell(m, sort)["value"] is not None]
    missing = [m for m in models if cell(m, sort)["value"] is None]
    return sorted(known, key=sort_key, reverse=descending) + missing
