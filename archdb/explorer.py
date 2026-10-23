"""State and architecture views for the terminal explorer."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from . import query
from .terminal import display

SECTIONS = {
    "All fields": [],
    "Overview": [
        "meta.modality",
        "params.total",
        "params.active",
        "params.counted_non_embedding",
        "params.counted_active_non_embedding",
        "params.counted_max_layer",
        "shape.layers",
        "shape.d_model",
        "attn.kind",
        "attn.kv_heads",
        "attn.local_kv_heads",
        "attn.linear_heads",
        "attn.kv_lora_rank",
        "moe.experts",
        "moe.top_k",
        "moe.expert_dim",
        "norm.placement",
        "mtp.depth",
        "train.accelerators",
        "train.parallelism",
    ],
    "Attention": ["attn", "kv"],
    "MoE": ["moe", "ffn", "mtp"],
    "Recipe": ["train", "ctx", "norm"],
    "KV cache": [
        "kv",
        "attn.kv_heads",
        "attn.local_kv_heads",
        "attn.kv_lora_rank",
        "attn.rope_dim",
        "attn.window",
        "attn.kv_precision",
    ],
}
PLOTS = [
    ("Active parameters / KV", "params.active", "kv.bytes_per_token", True),
    ("Total parameters / KV per active B", "params.total", "kv.bytes_per_token_per_active", True),
    ("Width / expert intermediate size", "shape.d_model", "moe.expert_dim", False),
    ("Active parameters / capability", "params.active", "eval.aa_index", False),
    ("Training tokens / batch tokens", "train.tokens", "train.batch_tokens", True),
]


def links(model: dict) -> dict[str, str]:
    result = {}
    for source in sorted(model["sources"].values(), key=lambda s: s["kind"] != "config"):
        url = source["url"]
        if not safe_url(url):
            continue
        match = re.match(r"https://huggingface\.co/([^/]+/[^/]+)/(resolve|blob)/", url)
        if match:
            result.setdefault("Hugging Face", "https://huggingface.co/" + match[1])
        if source["kind"] == "config":
            result.setdefault("Config", url.replace("/resolve/", "/blob/", 1))
        if source["kind"] == "paper":
            result.setdefault("Report", url)
    return result


def safe_url(url: str) -> bool:
    if not isinstance(url, str) or any(c.isspace() or not c.isprintable() for c in url):
        return False
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
            and parsed.port in (None, 443)
        )
    except ValueError:
        return False


def mixer_kind(name: str) -> str:
    if name == "SWA":
        return "local"
    if name in {"linear", "KDA", "Gated DeltaNet", "Lightning", "Mamba-2"}:
        return "linear"
    if "SWA" in name:
        return "hybrid"
    return "global"


def layer_glyph(name: str) -> str:
    if name.startswith("CSA2"):
        return "2" if "m=2" in name else "1"
    if name.startswith("CSA"):
        return "C"
    if name.startswith("HCA"):
        return "H"
    return {"KDA": "K", "Gated DeltaNet": "D", "Mamba-2": "M", "Lightning": "L"}.get(
        name, {"global": "■", "local": "▰", "hybrid": "▥", "linear": "▪"}[mixer_kind(name)]
    )


def layer_layout(model: dict) -> tuple[list[str], str]:
    fact = model["facts"].get("attn.layout", {})
    if sequence := fact.get("sequence"):
        qualifier = "inferred order" if fact.get("basis") == "inferred" else "sourced order"
        return sequence, qualifier
    counts = {
        kind: query.cell(model, key)["value"]
        for kind, key in (
            ("global", "attn.global_layers"),
            ("local", "attn.local_layers"),
            ("linear", "attn.linear_layers"),
        )
    }
    nonzero = [(kind, n) for kind, n in counts.items() if isinstance(n, int) and n > 0]
    count_covers_stack = (
        len(nonzero) == 1
        and nonzero[0][1] == query.cell(model, "shape.layers")["value"]
        and "SWA" not in str(query.cell(model, "attn.kind")["value"])
    )
    hybrid_branch = "SWA" in str(query.cell(model, "attn.kind")["value"]) and counts["global"]
    if (
        len(nonzero) == 1
        and not hybrid_branch
        and (all(n is not None for n in counts.values()) or count_covers_stack)
    ):
        kind, count = nonzero[0]
        return [{"global": "global", "local": "SWA", "linear": "linear"}[kind]] * count, "uniform layout"
    description = " · ".join(f"{count} {kind}" for kind, count in nonzero)
    return [], (description + "; order not sourced") if description else "Layer order not available"


def plot_points(
    models: list[dict], xkey: str, ykey: str, width: int, height: int, log: bool
) -> tuple[list[dict], tuple]:
    points = []
    for model in models:
        x, y = query.number(query.cell(model, xkey)["value"]), query.number(query.cell(model, ykey)["value"])
        if x is None or y is None or (log and (x <= 0 or y <= 0)):
            continue
        points.append(
            {
                "model": model,
                "x": x,
                "y": y,
                "px": math.log10(x) if log else x,
                "py": math.log10(y) if log else y,
            }
        )
    if not points:
        return [], ()
    xmin, xmax = min(p["px"] for p in points), max(p["px"] for p in points)
    ymin, ymax = min(p["py"] for p in points), max(p["py"] for p in points)
    for point in points:
        point["col"] = (
            round((point["px"] - xmin) / (xmax - xmin) * (width - 1)) if xmax != xmin else width // 2
        )
        point["row"] = (
            height
            - 1
            - (round((point["py"] - ymin) / (ymax - ymin) * (height - 1)) if ymax != ymin else height // 2)
        )
    return points, (xmin, xmax, ymin, ymax)


@dataclass
class Explorer:
    db: dict
    search: str = ""
    filters: list[str] = field(default_factory=list)
    sort: str = "name"
    descending: bool = False
    cursor: int = 0
    selected: set[str] = field(default_factory=set)
    section: str = "All fields"
    field_search: str = ""
    view: str = "Models"
    custom_fields: list[str] | None = None
    models: list[dict] = field(default_factory=list)
    x: str = PLOTS[0][1]
    y: str = PLOTS[0][2]
    log: bool = True
    plot_preset: int = 0

    def refresh(self, preserve: bool = True) -> None:
        previous = self.model["id"] if preserve and self.model else None
        models = query.select(self.db, self.search, self.filters, self.sort, self.descending)
        self.models = models
        self.cursor = next((i for i, m in enumerate(models) if m["id"] == previous), 0)

    @property
    def model(self) -> dict | None:
        return self.models[min(self.cursor, len(self.models) - 1)] if self.models else None

    @property
    def comparison(self) -> list[dict]:
        return [m for m in self.db["models"] if m["id"] in self.selected]

    def move(self, delta: int) -> None:
        self.cursor = max(0, min(len(self.models) - 1, self.cursor + delta))

    def toggle(self) -> None:
        if self.model:
            key = self.model["id"]
            self.selected.symmetric_difference_update({key})

    def field_keys(self) -> list[str]:
        if self.custom_fields is not None:
            keys = self.custom_fields
        else:
            wanted = SECTIONS[self.section]
            keys = (
                [k for k in wanted if k in self.db["fields"]]
                if self.section == "Overview"
                else [k for k in self.db["fields"] if not wanted or k in wanted or k.split(".")[0] in wanted]
            )
        terms = self.field_search.casefold().split()
        return [k for k in keys if all(t in f"{k} {self.db['fields'][k]}".casefold() for t in terms)]

    def value(self, model: dict, key: str, annotate: bool = False) -> str:
        c = query.cell(model, key)
        if isinstance(c["value"], list):
            noun = (
                "stages"
                if key == "train.stages"
                else "cache groups"
                if key == "attn.kv_groups"
                else "components"
            )
            return f"{len(c['value'])} {noun} · Enter for details"
        value = display(c["value"], self.db["fields"].get(key), c["status"])
        if annotate and c["status"] in {"derived", "inferred"}:
            value += " (calculated)" if c["status"] == "derived" else " (inferred)"
        return value

    def set_sort(self, key: str, descending: bool | None = None) -> None:
        query.check_field(self.db, key)
        self.sort = key
        self.descending = (
            descending
            if descending is not None
            else self.db["fields"].get(key, {}).get("type") in {"int", "float", "date", "bool"}
        )
        self.refresh(False)

    def set_filter(self, expression: str) -> None:
        query.predicate(self.db, expression)
        self.filters.append(expression)
        self.refresh()

    def set_kv(self, context: int, precision: str) -> None:
        from .derive import derive

        if context <= 0 or precision not in {"bf16", "fp8", "fp4", "sourced"}:
            raise ValueError("Context must be positive; precision must be bf16, fp8, fp4 or sourced.")
        for model in self.db["models"]:
            values = derive(model["facts"], context=context, kv_dtype=precision)
            model["derived"].update({k: v for k, v in values.items() if k.startswith("kv.")})
        self.db["_kv"] = {"context": context, "precision": precision}
        self.refresh()
