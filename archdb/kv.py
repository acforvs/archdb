"""Retained attention-cache payload, separate from prefill workspace and prefix storage."""

from __future__ import annotations

from math import ceil, floor

FORMULA_VERSION = "kv-2"
ELEMENT_BITS = {"bf16": 16, "fp8": 8, "fp4": 4}
COMPRESSED_KINDS = {"CSA", "HCA", "CSA2"}


def _value(facts: dict, key: str):
    return facts.get(key, {}).get("value")


def _number(facts: dict, key: str):
    v = _value(facts, key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def groups(facts: dict) -> list[dict] | str:
    """Cache owners and entry shapes; shared caches are counted once."""
    explicit = _value(facts, "attn.kv_groups")
    if explicit:
        return explicit
    kinds = str(_value(facts, "attn.kind") or "").replace(",", " ").split()
    if COMPRESSED_KINDS.intersection(kinds):
        return "compressed attention without explicit cache groups (not modeled)"
    indexer = _number(facts, "attn.indexer_dim") or 0
    latent, rope = _number(facts, "attn.kv_lora_rank"), _number(facts, "attn.rope_dim")
    if latent is not None:
        if rope is None:
            return "MLA without rope_dim"
        global_elements, local_elements = latent + rope + indexer, latent + rope
    else:
        heads, dk = _number(facts, "attn.kv_heads"), _number(facts, "attn.head_dim")
        dv = _number(facts, "attn.v_head_dim") or dk
        if heads is None or dk is None:
            return "missing kv_heads or head_dim"
        global_elements = heads * (dk + dv) + indexer
        local_heads = _number(facts, "attn.local_kv_heads") or heads
        local_dk = _number(facts, "attn.local_head_dim") or dk
        local_dv = _number(facts, "attn.local_v_head_dim")
        if local_dv is None:
            local_dv = _number(facts, "attn.v_head_dim") or local_dk
        local_elements = local_heads * (local_dk + local_dv)
    n_global = _number(facts, "attn.global_layers")
    if n_global is None:
        return "missing attn.global_layers"
    result = [{"name": "Global KV", "layers": n_global, "elements": global_elements}]
    n_local = _number(facts, "attn.local_layers") or 0
    n_long = _number(facts, "attn.long_window_layers") or 0
    if n_local:
        window, long_window = _number(facts, "attn.window"), _number(facts, "attn.long_window")
        if window is None or (n_long and long_window is None):
            return "sliding-window layers without window"
        result.append(
            {"name": "SWA KV", "layers": n_local - n_long, "elements": local_elements, "window": window}
        )
        if n_long:
            result.append(
                {
                    "name": "Long-window KV",
                    "layers": n_long,
                    "elements": local_elements,
                    "window": long_window,
                }
            )
    return [g for g in result if g["layers"]]


def cache(facts: dict, dtype: str = "bf16") -> dict | str:
    """Uniform payload or the complete sourced storage recipe, including scales and padding."""
    topology = groups(facts)
    if isinstance(topology, str):
        return topology
    precision = _value(facts, "attn.kv_precision") or []
    result, parts = [], []
    for g in topology:
        if dtype == "sourced":
            components = [p for p in precision if p["group"] == g["name"]]
            if sum(p["elements"] for p in components) != g["elements"]:
                return f"cache storage recipe not fully sourced: {g['name']}"
            size = sum(
                ceil(p["elements"] * p["bits"] / 8)
                + p.get("scale_count", 0) * p.get("scale_bits", 0) / 8
                + p.get("padding_bytes", 0)
                for p in components
            )
            detail = " + ".join(
                f"ceil({p['elements']} × {p['bits']}/8)"
                + (f" + {p['scale_count']} × {p['scale_bits']}/8 scales" if p.get("scale_count") else "")
                + (f" + {p['padding_bytes']} padding" if p.get("padding_bytes") else "")
                for p in components
            )
        else:
            size = g["elements"] * ELEMENT_BITS[dtype] / 8
            detail = f"{g['elements']} × {ELEMENT_BITS[dtype]}/8"
        b = g["layers"] * size
        result.append(
            {
                "name": g["name"],
                "bytes_per_entry": b,
                "tokens_per_entry": g.get("tokens_per_entry", 1),
                **({"window": g["window"]} if g.get("window") else {}),
            }
        )
        parts.append(f"{g['name']}: {g['layers']} × ({detail}) B/entry")
    return {
        "per_token": sum(g["bytes_per_entry"] / g["tokens_per_entry"] for g in result if "window" not in g),
        "windows": [[g["bytes_per_entry"], g["window"]] for g in result if "window" in g],
        "groups": result,
        "formula": "; ".join(parts),
        "formula_version": FORMULA_VERSION,
    }


def at(c: dict, context: int) -> float:
    """Completed compressed entries and a bounded ring per SWA owner; tail/workspace excluded."""
    if context <= 0:
        raise ValueError("context must be positive")
    return sum(
        g["bytes_per_entry"] * min(floor(context / g["tokens_per_entry"]), g.get("window", context))
        for g in c["groups"]
    )


def context_formula(c: dict, context: int) -> str:
    parts = []
    for g in c["groups"]:
        entries = (
            f"floor({context} / {g['tokens_per_entry']})" if g["tokens_per_entry"] != 1 else str(context)
        )
        if "window" in g:
            entries = f"min({entries}, {g['window']})"
        parts.append(f"{g['bytes_per_entry']:g} B × {entries} ({g['name']})")
    return " + ".join(parts)
