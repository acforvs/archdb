"""Load and validate the field registry and model records."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class Loader(yaml.SafeLoader):
    """SafeLoader that also reads exponent floats without a dot (2e-4), which YAML 1.1 treats as strings."""


Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCE_KINDS = {"config", "api", "code", "paper", "model-card", "blog", "leaderboard"}
MODEL_KINDS = {"base", "instruct", "reasoning"}
WEIGHTS = {"open", "closed"}

# A source must point at content that cannot change: a commit, a versioned arXiv PDF, a Wayback
# snapshot or a Wikipedia revision. Hugging Face API responses (`api`) are the one exception.
PINNED = re.compile(
    r"huggingface\.co/.+/resolve/[0-9a-f]{40}/"
    r"|raw\.githubusercontent\.com/[^/]+/[^/]+/[0-9a-f]{40}/"
    r"|github\.com/.+/(blob|raw)/[0-9a-f]{40}/"
    r"|arxiv\.org/pdf/\d+\.\d+v\d+"
    r"|web\.archive\.org/web/\d{14}(id_)?/"
    r"|[?&]oldid=\d+"
)
FACT_KEYS = {"value", "source", "at", "quote", "note", "basis"}
TYPES = {
    "int": (int,),
    "float": (int, float),
    "str": (str,),
    "bool": (bool,),
    "list": (list,),
}


@dataclass
class Model:
    id: str
    name: str
    org: str
    kind: str
    sources: dict[str, dict]
    facts: dict[str, dict]
    weights: str = "open"
    note: str | None = None
    not_disclosed: list[str] = field(default_factory=list)
    config_unused: dict[str, str] = field(default_factory=dict)
    counts: dict | None = None
    errors: list[str] = field(default_factory=list)


def load_fields(path: Path = DATA / "fields.yaml") -> tuple[dict[str, str], dict[str, dict]]:
    doc = yaml.load(path.read_text(), Loader)
    return doc["groups"], doc["fields"]


def load_models(
    directory: Path = DATA / "models",
    evals: Path = DATA / "evals.yaml",
    counts: Path = DATA / "param_counts.yaml",
) -> list[Model]:
    """Records, with their entries from evals.yaml and param_counts.yaml merged in."""
    scores = yaml.load(evals.read_text(), Loader) or {} if evals.exists() else {}
    tallies = yaml.load(counts.read_text(), Loader) or {} if counts.exists() else {}
    models = []
    for path in sorted(directory.glob("*.yaml")):
        doc = yaml.load(path.read_text(), Loader)
        models.append(
            Model(
                id=path.stem,
                name=doc["name"],
                org=doc["org"],
                kind=doc["kind"],
                weights=doc.get("weights", "open"),
                note=doc.get("note"),
                sources=doc.get("sources") or {},
                facts=doc.get("facts") or {},
                not_disclosed=doc.get("not_disclosed") or [],
                config_unused=doc.get("config_unused") or {},
                counts=tallies.get(path.stem),
            )
        )
        extra = scores.pop(path.stem, {})
        for kind in ("sources", "facts"):
            clash = set(extra.get(kind) or {}) & set(getattr(models[-1], kind))
            if clash:
                models[-1].errors.append(f"evals.yaml {kind} {sorted(clash)} clash with the record")
            getattr(models[-1], kind).update(extra.get(kind) or {})
    for mid in scores:
        raise ValueError(f"evals.yaml: no record named {mid}")
    return models


def load_orgs(path: Path = DATA / "orgs.yaml") -> dict[str, dict]:
    return yaml.load(path.read_text(), Loader)


def _num(facts: dict, key: str) -> float:
    v = facts.get(key, {}).get("value")
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _text(facts: dict, key: str) -> str:
    return str(facts.get(key, {}).get("value") or "")


def _rope(facts: dict) -> bool:
    return re.search(r"(?<!no )RoPE", _text(facts, "attn.position"), re.IGNORECASE) is not None


LINEAR_KINDS = ("DeltaNet", "KDA", "Lightning", "Mamba", "linear")

# When a field marked `required: <condition>` in fields.yaml must be present (as a fact or listed in
# the record's not_disclosed). Conditions read other facts of the same record.
CONDITIONS = {
    "always": lambda f: True,
    "open_weights": lambda f: f.get("_weights") != "closed",
    "gqa": lambda f: "attn.kv_lora_rank" not in f,
    "mla": lambda f: "attn.kv_lora_rank" in f,
    "rope": lambda f: _rope(f) and "NoPE (global" not in _text(f, "attn.position"),
    "swa_rope": lambda f: _num(f, "attn.local_layers") > 0 and _rope(f),
    "swa_kind": lambda f: "SWA" in _text(f, "attn.kind"),
    "linear_kind": lambda f: any(k in _text(f, "attn.kind") for k in LINEAR_KINDS),
    "swa": lambda f: _num(f, "attn.local_layers") > 0,
    "linear": lambda f: _num(f, "attn.linear_layers") > 0,
    "hybrid": lambda f: _num(f, "attn.local_layers") > 0 or _num(f, "attn.linear_layers") > 0,
    "dsa": lambda f: "DSA" in _text(f, "attn.kind"),
    "moe": lambda f: "moe.experts" in f,
    "shared_experts": lambda f: _num(f, "moe.shared") > 0,
    "dense_ffn": lambda f: "moe.experts" not in f or _num(f, "moe.dense_layers") > 0,
    "mtp": lambda f: _num(f, "mtp.depth") > 0,
}

NUMERIC_ITEM_KEYS = {
    "tokens",
    "seq_len",
    "lr",
    "batch_tokens",
    "batch_sequences",
    "steps",
    "layers",
    "elements",
    "tokens_per_entry",
    "window",
}
BOUND = re.compile(r"^[~<>≥≤]\s*[\d.]+\s*[KMBT]?$")
DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


def required(m: Model, fields: dict[str, dict]) -> list[str]:
    """Fields the record must either state or list in not_disclosed."""
    facts = {**m.facts, "_weights": m.weights}
    return [k for k, spec in fields.items() if spec.get("required") and CONDITIONS[spec["required"]](facts)]


def validate(models: list[Model], fields: dict[str, dict], orgs: dict | None = None) -> list[str]:
    """Return a list of human-readable errors; empty means the data is structurally valid."""
    errors = []
    orgs = load_orgs() if orgs is None else orgs
    for m in models:
        where = f"{m.id}:"
        errors += [f"{where} {e}" for e in m.errors]
        if m.weights not in WEIGHTS:
            errors.append(f"{where} weights must be one of {sorted(WEIGHTS)}")
        if m.org not in orgs:
            errors.append(f"{where} org {m.org!r} is not in data/orgs.yaml")
        elif set(orgs[m.org]) != {"country", "url", "quote"}:
            errors.append(f"data/orgs.yaml: {m.org} needs exactly country, url and quote")
        elif not PINNED.search(orgs[m.org]["url"]):
            errors.append(f"data/orgs.yaml: {m.org} source is not pinned")
        for key in m.not_disclosed:
            if key not in fields:
                errors.append(f"{where} not_disclosed lists unknown field {key}")
            elif key in m.facts:
                errors.append(f"{where} {key} is both stated and listed in not_disclosed")
        missing = [k for k in required(m, fields) if k not in m.facts and k not in m.not_disclosed]
        if missing:
            errors.append(
                f"{where} required fields neither stated nor in not_disclosed: {', '.join(missing)}"
            )
        if m.kind not in MODEL_KINDS:
            errors.append(f"{where} kind {m.kind!r} not in {sorted(MODEL_KINDS)}")
        for sid, src in m.sources.items():
            if set(src) != {"url", "kind"}:
                errors.append(f"{where} source {sid} must have exactly url and kind")
            elif src["kind"] not in SOURCE_KINDS:
                errors.append(f"{where} source {sid} has unknown kind {src['kind']!r}")
            elif src["kind"] != "api" and not PINNED.search(src["url"]):
                errors.append(f"{where} source {sid} is not pinned to a commit, version or snapshot")
        used = set()
        for key, fact in m.facts.items():
            spec = fields.get(key)
            if spec is None:
                errors.append(f"{where} unknown field {key}")
                continue
            if spec.get("derived"):
                errors.append(f"{where} {key} is derived and must not be authored")
            if not isinstance(fact, dict) or "value" not in fact:
                errors.append(f"{where} {key} must be a mapping with a value")
                continue
            extra = set(fact) - FACT_KEYS
            if extra:
                errors.append(f"{where} {key} has unknown keys {sorted(extra)}")
            if fact.get("basis") not in (None, "inferred"):
                errors.append(f"{where} {key} basis must be 'inferred' when present")
            if fact.get("basis") == "inferred" and not fact.get("note"):
                errors.append(f"{where} {key} is inferred and needs a note explaining how")
            if isinstance(fact["value"], list):
                for item in fact["value"]:
                    if isinstance(item, dict) and "source" in item:
                        if item["source"] in m.sources:
                            used.add(item["source"])
                        else:
                            errors.append(f"{where} {key} item cites undefined source {item['source']!r}")
            source = fact.get("source")
            if source not in m.sources:
                errors.append(f"{where} {key} cites undefined source {source!r}")
            else:
                used.add(source)
            if not fact.get("at"):
                errors.append(f"{where} {key} needs a location ('at')")
            errors += _check_type(where, key, fact["value"], spec)
        for sid in set(m.sources) - used:
            errors.append(f"{where} source {sid} is never cited")
    return errors


def _check_type(where: str, key: str, value, spec: dict) -> list[str]:
    kind = spec["type"]
    if kind == "date":
        return [] if isinstance(value, str) and DATE.match(value) else [f"{where} {key} must be 'YYYY-MM-DD'"]
    if kind == "float" and isinstance(value, str) and value.startswith((">", "<", "~")):
        return []  # bounds such as ">30T" are kept verbatim rather than guessed
    if isinstance(value, bool) and kind != "bool":
        return [f"{where} {key} must be {kind}, got bool"]
    if not isinstance(value, TYPES[kind]):
        return [f"{where} {key} must be {kind}, got {type(value).__name__}"]
    if "enum" in spec and value not in spec["enum"]:
        return [f"{where} {key} must be one of {spec['enum']}"]
    if kind == "list":
        bad = [s for s in value if not isinstance(s, dict) or set(s) - set(spec["items"])]
        if bad:
            return [f"{where} {key} items may only use {spec['items']}"]
        for item in value:
            for k in NUMERIC_ITEM_KEYS & set(item):
                v = item[k]
                if not (isinstance(v, (int, float)) and not isinstance(v, bool)) and not BOUND.match(str(v)):
                    return [
                        f"{where} {key} item {item.get('name')!r}: {k} must be a number or a bound like '~2B'"
                    ]
    return []
