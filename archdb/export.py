"""Write the dataset in formats that are usable without the website."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .data import Model
from .derive import derive

STAGE_COLUMNS = [
    "name",
    "tokens",
    "seq_len",
    "lr",
    "batch_tokens",
    "batch_sequences",
    "steps",
    "note",
    "quote",
]


def dataset(models: list[Model], groups: dict, fields: dict, orgs: dict) -> dict:
    rows = []
    for m in models:
        derived = derive(m.facts, counts=m.counts)
        rows.append(
            {
                "id": m.id,
                "name": m.name,
                "org": m.org,
                "country": orgs[m.org]["country"],
                "kind": m.kind,
                "weights": m.weights,
                "note": m.note,
                "sources": m.sources,
                "facts": m.facts,
                "not_disclosed": m.not_disclosed,
                "derived": {k: v for k, v in derived.items() if not k.startswith("_")},
                "kv": derived.get("_kv"),
            }
        )
    return {"groups": groups, "fields": fields, "orgs": orgs, "models": rows}


def files(data: dict) -> dict[str, str]:
    """Every export as {relative path: content}."""
    fields, models = data["fields"], data["models"]
    out = {"archdb.json": json.dumps(data, ensure_ascii=False, indent=1) + "\n"}

    out["fields.csv"] = _csv(
        ["field", "group", "label", "type", "unit", "derived", "definition"],
        [
            [
                k,
                k.split(".")[0],
                f["label"],
                f["type"],
                f.get("unit", ""),
                bool(f.get("derived")),
                f["definition"],
            ]
            for k, f in fields.items()
        ],
    )

    keys = [k for k in fields if fields[k]["type"] != "list"]
    out["models.csv"] = _csv(
        ["id", "name", "org", "country", "kind", "weights", *keys],
        [
            [m["id"], m["name"], m["org"], m["country"], m["kind"], m["weights"]]
            + [_cell(_value(m, k)) for k in keys]
            for m in models
        ],
    )

    fact_rows = []
    for m in models:
        for k, f in m["facts"].items():
            if k == "train.stages":
                continue
            src = m["sources"][f["source"]]
            fact_rows.append(
                [
                    m["id"],
                    k,
                    _cell(f["value"]),
                    fields[k].get("unit", ""),
                    src["kind"],
                    src["url"],
                    f["at"],
                    f.get("quote", ""),
                    f.get("basis", "stated"),
                    f.get("note", ""),
                ]
            )
        for k in m["not_disclosed"]:
            fact_rows.append([m["id"], k, "", fields[k].get("unit", ""), "", "", "", "", "not disclosed", ""])
        for k, d in m["derived"].items():
            if d["value"] is not None:
                fact_rows.append(
                    [
                        m["id"],
                        k,
                        _cell(d["value"]),
                        fields[k].get("unit", ""),
                        "derived",
                        "",
                        d["formula"],
                        "",
                        "derived",
                        "",
                    ]
                )
    out["facts.csv"] = _csv(
        ["model", "field", "value", "unit", "source_kind", "source_url", "at", "quote", "basis", "note"],
        fact_rows,
    )

    stage_rows = []
    for m in models:
        f = m["facts"].get("train.stages")
        if not f:
            continue
        for i, st in enumerate(f["value"]):
            url = m["sources"][st.get("source", f["source"])]["url"]
            stage_rows.append([m["id"], i, *[_cell(st.get(c)) for c in STAGE_COLUMNS], url])
    out["stages.csv"] = _csv(["model", "order", *STAGE_COLUMNS, "source_url"], stage_rows)
    return out


def write(data: dict, directory: Path, check: bool = False) -> list[str]:
    """Write exports; with check=True, only report files whose content would change."""
    stale = []
    for name, content in files(data).items():
        path = directory / name
        if path.exists() and path.read_text() == content:
            continue
        stale.append(name)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    return stale


def _value(m: dict, k: str):
    if k in m["facts"]:
        return m["facts"][k]["value"]
    if k in m["derived"]:
        return m["derived"][k]["value"]
    return None


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return v


def _csv(header: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()
