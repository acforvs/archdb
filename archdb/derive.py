"""Quantities computed from authored facts. Each result records its inputs so it can be audited."""

from __future__ import annotations

from . import kv


def value(facts: dict, key: str):
    fact = facts.get(key)
    return None if fact is None else fact["value"]


def number(facts: dict, key: str):
    v = value(facts, key)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


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


def derive(
    facts: dict, context: int = 131_072, counts: dict | None = None, kv_dtype: str = "bf16"
) -> dict[str, dict]:
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

    uniform = kv.cache(facts)
    sourced = kv.cache(facts, "sourced")
    selected = (
        uniform if kv_dtype == "bf16" else sourced if kv_dtype == "sourced" else kv.cache(facts, kv_dtype)
    )
    if not isinstance(uniform, str):
        out["_kv"] = uniform
        if not isinstance(sourced, str):
            out["_kv"]["sourced"] = sourced
    if isinstance(selected, str):
        keys = ["kv.growth_per_token", "kv.bytes_per_token", "kv.bytes_at_context"]
        if active:
            keys += ["kv.bytes_per_token_per_active", "kv.bytes_at_context_per_active"]
        for key in keys:
            out[key] = {"value": None, "formula": selected, "formula_version": kv.FORMULA_VERSION}
        return out
    total = kv.at(selected, context)
    out["kv.growth_per_token"] = {
        "value": selected["per_token"],
        "formula": " + ".join(
            f"{g['bytes_per_entry']:g} B / {g['tokens_per_entry']} tokens ({g['name']})"
            for g in selected["groups"]
            if "window" not in g
        )
        or "0 (no global KV)",
    }
    out["kv.bytes_at_context"] = {
        "value": total,
        "formula": kv.context_formula(selected, context),
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
    assumption = (
        "sourced component formats, scales and padding"
        if kv_dtype == "sourced"
        else f"uniform {kv_dtype.upper()}, payload only; quantization overhead excluded"
    )
    for key, result in out.items():
        if key.startswith("kv."):
            result["formula_version"] = kv.FORMULA_VERSION
            result["formula"] += f"; {assumption}. {selected['formula']}"
    return out
