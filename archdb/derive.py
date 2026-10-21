"""Quantities computed from authored facts. Each result records its inputs so it can be audited."""

from __future__ import annotations

BYTES_BF16 = 2

# Attention types whose cache is not a plain per-token K/V or MLA latent. Without an explicit
# attn.kv_groups description their KV size is left unknown rather than approximated.
REQUIRES_EXPLICIT_KV_GROUPS = ("CSA", "HCA", "CSA2")


def value(facts: dict, key: str):
    fact = facts.get(key)
    return None if fact is None else fact["value"]


def number(facts: dict, key: str):
    v = value(facts, key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def kv_elements(facts: dict) -> tuple[int | None, int | None, str]:
    """K/V scalars cached per token per layer for (global, sliding-window) layers, plus the formula used."""
    kinds = str(value(facts, "attn.kind") or "")
    if any(k in kinds.replace(",", " ").split() for k in REQUIRES_EXPLICIT_KV_GROUPS):
        return None, None, "compressed attention (not modeled)"
    indexer = number(facts, "attn.indexer_dim") or 0
    indexer_note = " + indexer_dim" if indexer else ""
    latent, rope = number(facts, "attn.kv_lora_rank"), number(facts, "attn.rope_dim")
    if latent is not None:
        if rope is None:
            return None, None, "MLA without rope_dim"
        # Sliding-window layers of MLA models run the same attention and cache the same latent.
        return latent + rope + indexer, latent + rope, "MLA: kv_lora_rank + rope_dim" + indexer_note
    heads, dk = number(facts, "attn.kv_heads"), number(facts, "attn.head_dim")
    dv = number(facts, "attn.v_head_dim") or dk
    if heads is None or dk is None:
        return None, None, "missing kv_heads or head_dim"
    global_elems = heads * (dk + dv) + indexer
    local_heads = number(facts, "attn.local_kv_heads") or heads
    local_dim = number(facts, "attn.local_head_dim")
    local_elems = local_heads * (2 * local_dim if local_dim else dk + dv)
    return global_elems, local_elems, "GQA: kv_heads × (head_dim + v_head_dim)" + indexer_note


TOLERANCE = 0.05


def counted(facts: dict, counts: dict | None) -> dict[str, dict]:
    """Parameter counts from checkpoint headers, shown only when they reproduce the publisher's totals.

    The publisher's total may or may not include MTP modules, vision/audio towers and n-gram lookup
    tables, and the active count may or may not include the embedding and output head; a count is
    accepted when some combination of those scopes is within TOLERANCE of both stated numbers."""
    if not counts:
        return {}
    c = counts
    experts, top_k = number(facts, "moe.experts"), number(facts, "moe.top_k")
    share = top_k / experts if experts and top_k else 1.0
    backbone = c["embedding"] + c["head"] + c["routed"] + c["dense"]
    active_ne = c["dense"] + c["routed"] * share
    total, active = number(facts, "params.total"), number(facts, "params.active")
    totals = {
        "+".join(name for name, on in zip(("lookup", "MTP", "other modalities"), combo) if on)
        or "backbone": backbone
        + combo[0] * c["lookup"]
        + combo[1] * c["mtp"]
        + combo[2] * c["other_modality"]
        for combo in [(a, b, d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
    }
    actives = {
        "non-embedding": active_ne,
        "with output head": active_ne + c["head"],
        "with embedding": active_ne + c["embedding"],
        "with embedding and output head": active_ne + c["embedding"] + c["head"],
    }

    def match(options: dict, stated):
        if not stated:
            return None
        # A number stated as "3B" covers 2.5B to 3.5B: allow half a unit of its last digit.
        digits = str(int(stated)).rstrip("0")
        slack = max(TOLERANCE * stated, 0.5 * 10 ** (len(str(int(stated))) - len(digits)))
        best = min(options, key=lambda k: abs(options[k] - stated))
        return best if abs(options[best] - stated) <= slack else None

    t_scope, a_scope = match(totals, total), match(actives, active)
    source = f"{c['repo']}@{c['commit'][:7]} safetensors headers"
    if not (t_scope and a_scope):
        reason = f"checkpoint count ({backbone / 1e9:.1f}B backbone) does not reproduce the stated total and active counts"
        return {k: {"value": None, "formula": reason} for k in COUNTED}
    check = f"; reproduces stated total ({t_scope}) and active ({a_scope})"
    return {
        "params.counted_total": {
            "value": backbone,
            "formula": f"embedding + output head + all experts + other weights, {source}{check}",
        },
        "params.counted_embedding": {
            "value": c["embedding"] + c["head"],
            "formula": f"input embedding + output head, {source}",
        },
        "params.counted_active_non_embedding": {
            "value": active_ne,
            "formula": f"non-expert weights + routed experts × {top_k}/{experts}, excluding embedding and output head, {source}{check}"
            if experts and top_k
            else f"all weights except embedding and output head, {source}{check}",
        },
        **(
            {
                "params.counted_lookup": {
                    "value": c["lookup"],
                    "formula": f"n-gram / per-layer embedding tables, {source}",
                }
            }
            if c["lookup"]
            else {}
        ),
    }


COUNTED = ("params.counted_total", "params.counted_embedding", "params.counted_active_non_embedding")


def derive(facts: dict, context: int = 131_072, counts: dict | None = None) -> dict[str, dict]:
    out = counted(facts, counts)
    layers, d_model = number(facts, "shape.layers"), number(facts, "shape.d_model")
    if layers and d_model:
        out["shape.aspect"] = {"value": d_model / layers, "formula": "d_model / layers"}
    heads, kv_heads, head_dim = (
        number(facts, "attn.heads"),
        number(facts, "attn.kv_heads"),
        number(facts, "attn.head_dim"),
    )
    if heads and head_dim:
        out["attn.qk_width"] = {"value": heads * head_dim, "formula": "heads × head_dim"}
    if heads and kv_heads:
        out["attn.group_size"] = {"value": heads / kv_heads, "formula": "heads / kv_heads"}
    ffn, expert = number(facts, "ffn.dim"), number(facts, "moe.expert_dim")
    if ffn and d_model:
        out["ffn.mult"] = {"value": ffn / d_model, "formula": "ffn.dim / d_model"}
    if expert and d_model:
        out["moe.granularity"] = {"value": expert / d_model, "formula": "moe.expert_dim / d_model"}
    latent = number(facts, "moe.latent_dim")
    if latent and d_model:
        out["moe.latent_ratio"] = {"value": d_model / latent, "formula": "d_model / moe.latent_dim"}
    n_global = number(facts, "attn.global_layers")
    for key, src in (("attn.swa_ratio", "attn.local_layers"), ("attn.linear_ratio", "attn.linear_layers")):
        n = number(facts, src)
        if n and n_global:
            out[key] = {"value": n / n_global, "formula": f"{src} / attn.global_layers"}
    experts, top_k = number(facts, "moe.experts"), number(facts, "moe.top_k")
    if experts and top_k:
        out["moe.sparsity"] = {"value": experts / top_k, "formula": "experts / top_k"}
    total, active = number(facts, "params.total"), number(facts, "params.active")
    if experts and total and active:
        out["moe.active_fraction"] = {"value": active / total, "formula": "params.active / params.total"}

    tokens, batch = number(facts, "train.tokens"), number(facts, "train.batch_tokens")
    if tokens and total:
        out["scaling.tpp"] = {"value": tokens / total, "formula": "train.tokens / params.total"}
    counted_ne = out.get("params.counted_active_non_embedding", {}).get("value")
    active_ne, ne_key = (
        (counted_ne, "params.counted_active_non_embedding")
        if counted_ne
        else (number(facts, "params.active_non_embedding"), "params.active_non_embedding")
    )
    if tokens and active_ne:
        out["scaling.tpp_active"] = {"value": tokens / active_ne, "formula": f"train.tokens / {ne_key}"}
    lr, wd = number(facts, "train.lr_peak"), number(facts, "train.weight_decay")
    if tokens and batch and lr and wd and str(value(facts, "train.optimizer") or "").startswith("AdamW"):
        out["scaling.adamw_tau"] = {
            "value": batch / (lr * wd * tokens),
            "formula": "train.batch_tokens / (train.lr_peak × train.weight_decay × train.tokens)",
        }

    kv = kv_cache(facts)
    if isinstance(kv, str):
        for key in ("kv.growth_per_token", "kv.bytes_per_token", "kv.bytes_at_context"):
            out[key] = {"value": None, "formula": kv}
        return out
    growth, windows, formula = kv
    total = kv_at(growth, windows, context)
    out["_kv"] = {"per_token": growth, "windows": windows}
    out["kv.growth_per_token"] = {"value": growth, "formula": formula}
    out["kv.bytes_at_context"] = {
        "value": total,
        "formula": f"kv.growth_per_token × {context} tokens"
        + "".join(f" + {b} B × min(context, {w})" for b, w in windows),
    }
    out["kv.bytes_per_token"] = {
        "value": total / context,
        "formula": f"kv.bytes_at_context / {context} tokens",
    }
    if active:
        out["kv.bytes_per_token_per_active"] = {
            "value": total / context / (active / 1e9),
            "formula": f"kv.bytes_per_token ({context} tokens) / (params.active / 1e9)",
        }
        out["kv.bytes_at_context_per_active"] = {
            "value": total / (active / 1e9),
            "formula": f"kv.bytes_at_context ({context} tokens) / (params.active / 1e9)",
        }
    return out


def kv_at(per_token: float, windows: list[list[int]], context: int) -> float:
    """BF16 KV bytes for one sequence: growing part plus sliding-window parts capped at their window."""
    return per_token * context + sum(b * min(context, w) for b, w in windows)


def kv_cache(facts: dict):
    """(bytes per token, [[bytes per window entry, window], ...], formula), or a reason string."""
    groups = value(facts, "attn.kv_groups")
    if groups:
        per_token, windows, parts = 0.0, [], []
        for g in groups:
            b = g["layers"] * g["elements"] * BYTES_BF16
            if g.get("window"):
                windows.append([b, g["window"]])
            else:
                per_token += b / g.get("tokens_per_entry", 1)
                parts.append(f"{g['layers']} × {g['elements']} / {g.get('tokens_per_entry', 1)}")
        return per_token, windows, "attn.kv_groups: (" + " + ".join(parts) + ") elements × 2 bytes"
    g_elems, l_elems, formula = kv_elements(facts)
    n_global = number(facts, "attn.global_layers")
    if g_elems is None:
        return formula
    if n_global is None:
        return "missing attn.global_layers"
    per_token = n_global * g_elems * BYTES_BF16
    formula = f"{n_global} global layers × {g_elems} elements × 2 bytes ({formula})"
    n_local, window = number(facts, "attn.local_layers") or 0, number(facts, "attn.window")
    n_long, long_window = number(facts, "attn.long_window_layers") or 0, number(facts, "attn.long_window")
    windows = []
    if n_local:
        if window is None or l_elems is None or (n_long and long_window is None):
            return "sliding-window layers without window"
        windows.append([(n_local - n_long) * l_elems * BYTES_BF16, window])
        if n_long:
            windows.append([n_long * l_elems * BYTES_BF16, long_window])
    return per_token, windows, formula
