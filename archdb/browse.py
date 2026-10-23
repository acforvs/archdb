"""Research commands over the shipped dataset."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from . import query
from .terminal import clean, display, table

DEFAULT_FIELDS = ["params.total", "params.active", "shape.layers", "shape.d_model", "attn.kind"]
COMMANDS = {"explore", "models", "search", "show", "compare", "fields", "evidence"}


def positive_integer(text):
    try:
        return query.positive_integer(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_commands(sub) -> None:
    explore = sub.add_parser("explore", help="open the interactive terminal browser")
    models = sub.add_parser("models", aliases=["search"], help="search, filter and sort models")
    for command in (models, explore):
        command.add_argument("search", nargs="?", default="", help="model name, organization or ID")
        command.add_argument(
            "--where", action="append", default=[], metavar="FILTER", help="repeat to require every condition"
        )
        command.add_argument("--sort", default="name", metavar="FIELD")
        command.add_argument(
            "--desc", action="store_true", help="sort largest or newest first; missing values stay last"
        )
    models.add_argument("--limit", type=positive_integer, help="maximum number of results")
    show = sub.add_parser("show", help="read one model's architecture and training recipe")
    show.add_argument("model", help="model ID or unambiguous name")
    compare = sub.add_parser("compare", help="compare selected models by field")
    compare.add_argument("models", nargs="+", help="two or more model IDs or unambiguous names")
    fields = sub.add_parser("fields", help="find field keys, units and definitions")
    fields.add_argument("search", nargs="?", default="", help="search keys, labels, definitions and aliases")
    evidence = sub.add_parser("evidence", help="show a value's source, quote or calculation")
    evidence.add_argument("model")
    evidence.add_argument("field")
    for command in (explore, models, show, compare, fields, evidence):
        command.add_argument(
            "--data", type=Path, metavar="JSON", help="use a different local ArchDB JSON export"
        )
        if command is not explore:
            command.add_argument("--format", choices=["table", "json", "csv"], default="table")
        if command is not evidence:
            command.add_argument(
                "--group", action="append", default=[], help="field group, e.g. attn, moe, train; repeatable"
            )
        if command in (explore, models, show, compare):
            command.add_argument("--fields", metavar="KEYS", help="comma-separated field keys")
            if command is not explore:
                command.add_argument("--evidence", action="store_true", help="include sources and formulas")
        if command in (explore, models, show, compare, evidence):
            command.add_argument(
                "--context",
                type=positive_integer,
                default=131_072,
                help="KV sequence length (default: 131072; suffix K = 1000)",
            )
            command.add_argument(
                "--precision",
                choices=["bf16", "fp8", "fp4", "sourced"],
                default="bf16",
                help="KV storage recipe (default: bf16)",
            )


def selected_fields(db: dict, args) -> list[str]:
    keys = []
    if getattr(args, "fields", None):
        keys = [k.strip() for k in args.fields.split(",") if k.strip()]
        if not keys:
            raise ValueError("--fields needs at least one field key")
        for key in keys:
            query.check_field(db, key)
    for group in getattr(args, "group", []):
        if group not in db["groups"]:
            raise ValueError(f"unknown group {group!r}; choose from {', '.join(db['groups'])}")
    grouped = [k for k in db["fields"] if k.split(".")[0] in getattr(args, "group", [])]
    if grouped:
        keys = [k for k in keys if k in grouped] if keys else grouped
        if not keys:
            raise ValueError("none of the selected fields belong to these groups")
    return list(dict.fromkeys(keys))


def rows(db: dict, models: list[dict], keys: list[str], evidence: bool = False) -> list[dict]:
    result = []
    for model in models:
        cells = {k: query.cell(model, k) for k in keys}
        row = {"id": model["id"], "name": model["name"], **{k: c["value"] for k, c in cells.items()}}
        row["_status"] = {k: c["status"] for k, c in cells.items()}
        if any(k.startswith("kv.") for k in keys):
            row["_kv"] = db["_kv"]
        if evidence:
            row["_evidence"] = cells
        result.append(row)
    return result


def json_output(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def csv_output(records: list[dict], keys: list[str]) -> None:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(keys)
    for row in records:
        writer.writerow(
            [
                json.dumps(row.get(k), ensure_ascii=False)
                if isinstance(row.get(k), (dict, list))
                else row.get(k)
                for k in keys
            ]
        )
    print(out.getvalue(), end="")


def explain(db: dict, model: dict, key: str) -> None:
    c = query.cell(model, key)
    spec = db["fields"].get(key, {})
    print(f"{clean(key)} — {display(c['value'], spec, c['status'])}")
    print(clean(f"  Status: {c['status']}" + (f"; unit: {spec['unit']}" if spec.get("unit") else "")))
    if source := c.get("source"):
        print(f"  Source: {clean(source['url'])}")
    for label, name in (
        ("Location", "at"),
        ("Quote", "quote"),
        ("Note", "note"),
        ("Formula", "formula"),
        ("Formula version", "formula_version"),
    ):
        if c.get(name):
            print(f"  {label}: {clean(c[name])}")
    if c.get("sequence"):
        print(f"  Ordered layers: {clean(' → '.join(c['sequence']))}")


def run(args) -> None:
    db = query.load(
        args.data, context=getattr(args, "context", 131_072), precision=getattr(args, "precision", "bf16")
    )
    if args.cmd == "explore":
        from .tui import explore

        explore(db, args)
        return
    if args.cmd == "fields":
        keys = selected_fields(db, args) or list(db["fields"])
        terms = args.search.casefold().split()
        keys = [k for k in keys if all(t in f"{k} {db['fields'][k]}".casefold() for t in terms)]
        records = [{"key": k, **db["fields"][k]} for k in keys]
        if args.format == "json":
            json_output(records)
        elif args.format == "csv":
            csv_output(records, ["key", "label", "type", "unit", "definition"])
        else:
            table(
                ["Field", "Label", "Unit", "Definition"],
                [[r["key"], r["label"], r.get("unit", ""), r["definition"]] for r in records],
            )
        return
    if args.cmd == "evidence":
        model = query.resolve(db, args.model)
        query.check_field(db, args.field)
        record = {
            "id": model["id"],
            "field": args.field,
            "unit": db["fields"].get(args.field, {}).get("unit"),
            **query.cell(model, args.field),
        }
        if args.format == "json":
            json_output(record)
        elif args.format == "csv":
            csv_output([record], list(record))
        else:
            print(clean(model["name"]))
            explain(db, model, args.field)
        return
    if args.cmd in {"models", "search"}:
        models = query.select(db, args.search, args.where, args.sort, args.desc)
        if args.limit:
            models = models[: args.limit]
        keys = selected_fields(db, args) or DEFAULT_FIELDS
    else:
        if args.cmd == "compare" and len(args.models) < 2:
            raise ValueError("compare needs at least two models")
        models = [query.resolve(db, name) for name in ([args.model] if args.cmd == "show" else args.models)]
        keys = selected_fields(db, args) or list(db["fields"])
        if not args.fields and not args.group:
            keys = [k for k in keys if any(query.cell(m, k)["value"] is not None for m in models)]
    records = rows(db, models, keys, args.evidence)
    if args.format == "json":
        json_output(records[0] if args.cmd == "show" else records)
    elif args.format == "csv":
        columns = ["id", "name", *keys, "_status"]
        if any(k.startswith("kv.") for k in keys):
            columns.append("_kv")
        if args.evidence:
            columns.append("_evidence")
        csv_output(records, list(dict.fromkeys(columns)))
    else:
        if args.cmd == "show":
            model = models[0]
            print(f"{clean(model['name'])} ({clean(model['id'])})")
            print(
                clean(
                    f"{model.get('org', '')} · {model.get('kind', '')} · {model.get('weights', '')} weights"
                )
            )
            print(clean(f"https://archdb.vladsavinov.com/models/{model['id']}/"))
            table(
                ["Field", "Value", "Status"],
                [
                    [
                        k,
                        display(
                            query.cell(model, k)["value"], db["fields"].get(k), query.cell(model, k)["status"]
                        ),
                        query.cell(model, k)["status"],
                    ]
                    for k in keys
                ],
            )
        elif args.cmd == "compare":
            table(
                ["Field", *[m["name"] for m in models]],
                [
                    [
                        k,
                        *[
                            display(
                                query.cell(m, k)["value"], db["fields"].get(k), query.cell(m, k)["status"]
                            )
                            for m in models
                        ],
                    ]
                    for k in keys
                ],
            )
        else:
            table(
                [
                    "Model ID",
                    *[db["fields"].get(k, {}).get("label", query.ATTRIBUTES.get(k, k)) for k in keys],
                ],
                [
                    [
                        m["id"],
                        *[
                            display(
                                query.cell(m, k)["value"], db["fields"].get(k), query.cell(m, k)["status"]
                            )
                            for k in keys
                        ],
                    ]
                    for m in models
                ],
            )
        if any(k.startswith("kv.") for k in keys):
            print(
                f"\nKV: {args.context:,} tokens; {args.precision.upper()}"
                + (
                    " storage, including sourced scales and padding."
                    if args.precision == "sourced"
                    else " payload; quantization overhead excluded."
                )
            )
        if args.evidence:
            for model in models:
                print(f"\n{clean(model['name'])}")
                for key in keys:
                    explain(db, model, key)
