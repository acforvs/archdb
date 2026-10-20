from archdb.derive import derive


def facts(**kv):
    return {k.replace("__", "."): {"value": v} for k, v in kv.items()}


def test_gqa_with_sliding_window():
    f = facts(
        attn__kind="GQA, SWA",
        attn__global_layers=10,
        attn__local_layers=30,
        attn__kv_heads=8,
        attn__head_dim=128,
        attn__window=512,
    )
    d = derive(f, context=4096)
    assert d["kv.growth_per_token"]["value"] == 10 * 8 * 256 * 2
    assert d["kv.bytes_at_context"]["value"] == 10 * 8 * 256 * 2 * 4096 + 30 * 8 * 256 * 2 * 512


def test_mla_caches_latent_plus_rope_key():
    f = facts(attn__kind="MLA", attn__global_layers=61, attn__kv_lora_rank=512, attn__rope_dim=64)
    assert derive(f)["kv.growth_per_token"]["value"] == 61 * 576 * 2


def test_separate_value_dim_and_local_heads():
    f = facts(
        attn__kind="GQA, SWA",
        attn__global_layers=1,
        attn__local_layers=1,
        attn__kv_heads=8,
        attn__local_kv_heads=16,
        attn__head_dim=192,
        attn__v_head_dim=128,
        attn__window=128,
    )
    d = derive(f, context=1000)
    assert d["kv.bytes_at_context"]["value"] == 8 * 320 * 2 * 1000 + 16 * 320 * 2 * 128


def test_compressed_attention_is_not_modeled():
    f = facts(attn__kind="SWA, CSA, HCA", attn__global_layers=61, attn__kv_heads=1, attn__head_dim=512)
    d = derive(f)
    assert d["kv.growth_per_token"]["value"] is None
    assert "not modeled" in d["kv.growth_per_token"]["formula"]


def test_kv_groups_model_compressed_and_shared_caches():
    groups = [
        {"name": "CSA", "layers": 30, "elements": 640, "tokens_per_entry": 4},
        {"name": "HCA", "layers": 31, "elements": 512, "tokens_per_entry": 128},
        {"name": "SWA", "layers": 61, "elements": 512, "window": 128},
    ]
    d = derive(facts(attn__kind="CSA, HCA", attn__kv_groups=groups), context=131_072)
    assert d["kv.growth_per_token"]["value"] == 9848
    assert d["kv.bytes_at_context"]["value"] == 9848 * 131_072 + 61 * 512 * 2 * 128


def test_kv_at_context_per_active_param():
    f = facts(
        attn__kind="MLA",
        attn__global_layers=61,
        attn__kv_lora_rank=512,
        attn__rope_dim=64,
        params__active=37e9,
    )
    d = derive(f, context=4096)
    assert d["kv.bytes_at_context_per_active"]["value"] == 61 * 576 * 2 * 4096 / 37


def test_indexer_keys_count_in_global_cache():
    f = facts(
        attn__kind="MLA, DSA",
        attn__global_layers=61,
        attn__kv_lora_rank=512,
        attn__rope_dim=64,
        attn__indexer_dim=128,
    )
    assert derive(f)["kv.growth_per_token"]["value"] == 61 * 704 * 2


def test_shape_ratios():
    f = facts(
        shape__layers=64,
        shape__d_model=5120,
        attn__heads=40,
        attn__kv_heads=8,
        attn__head_dim=128,
        ffn__dim=27648,
        moe__experts=256,
        moe__top_k=8,
    )
    d = derive(f)
    assert d["shape.aspect"]["value"] == 80
    assert d["attn.group_size"]["value"] == 5
    assert d["attn.qk_width"]["value"] == 5120
    assert d["ffn.mult"]["value"] == 5.4
    assert d["moe.sparsity"]["value"] == 32


def test_layout_and_latent_ratios():
    f = facts(
        shape__d_model=4096,
        moe__latent_dim=1024,
        attn__global_layers=8,
        attn__local_layers=40,
        attn__linear_layers=24,
    )
    d = derive(f)
    assert d["moe.latent_ratio"]["value"] == 4
    assert d["attn.swa_ratio"]["value"] == 5
    assert d["attn.linear_ratio"]["value"] == 3


def test_local_head_dim_differs_from_global():
    f = facts(
        attn__kind="GQA, SWA",
        attn__global_layers=10,
        attn__local_layers=50,
        attn__kv_heads=4,
        attn__local_kv_heads=16,
        attn__head_dim=512,
        attn__local_head_dim=256,
        attn__window=1024,
    )
    d = derive(f, context=2048)
    assert d["kv.bytes_at_context"]["value"] == 10 * 4 * 1024 * 2 * 2048 + 50 * 16 * 512 * 2 * 1024


def test_second_window_group():
    f = facts(
        attn__kind="GQA, SWA",
        attn__global_layers=2,
        attn__local_layers=3,
        attn__long_window_layers=1,
        attn__kv_heads=1,
        attn__head_dim=64,
        attn__window=128,
        attn__long_window=4096,
    )
    d = derive(f, context=8192)
    assert d["kv.bytes_at_context"]["value"] == 2 * 128 * 2 * 8192 + 2 * 128 * 2 * 128 + 1 * 128 * 2 * 4096


def test_exported_kv_components_reproduce_kv_at_context():
    """The website recomputes KV at other contexts from `_kv`; it must agree with derive() for every record."""
    from archdb.data import load_models

    for m in load_models():
        for context in (4096, 131_072, 1_048_576):
            d = derive(m.facts, context=context)
            kv = d.get("_kv")
            if kv is None:
                continue
            site = kv["per_token"] * context + sum(b * min(context, w) for b, w in kv["windows"])
            assert site == d["kv.bytes_at_context"]["value"], (m.id, context)


def test_recipe_ratios():
    f = facts(
        params__total=1e12,
        params__active=5e10,
        params__active_non_embedding=5e10,
        train__tokens=2e13,
        train__batch_tokens=1.6e7,
        train__lr_peak=2e-4,
        train__weight_decay=0.1,
        train__optimizer="AdamW",
    )
    d = derive(f)
    assert d["scaling.tpp"]["value"] == 20
    assert d["scaling.tpp_active"]["value"] == 400
    assert d["scaling.batch_ratio"]["value"] == 1.6e7 / 2e13**0.47
    assert abs(d["scaling.adamw_tau"]["value"] - 1.6e7 / (2e-4 * 0.1 * 2e13)) < 1e-12


def test_adamw_timescale_needs_adamw():
    f = facts(
        train__tokens=2e13,
        train__batch_tokens=1.6e7,
        train__lr_peak=2e-4,
        train__weight_decay=0.1,
        train__optimizer="Muon + AdamW",
    )
    assert "scaling.adamw_tau" not in derive(f)


def test_active_fraction_uses_the_reported_parameter_scope():
    from tests.test_checkpoint import COUNTS

    f = facts(moe__experts=96, moe__top_k=8, params__total=101e9, params__active=11e9)
    d = derive(f, counts=COUNTS)
    assert d["params.counted_total"]["value"] == 100e9
    assert d["moe.active_fraction"]["value"] == 11 / 101
    assert d["moe.active_fraction"]["formula"] == "params.active / params.total"


def test_bytes_per_token_is_total_over_context():
    f = facts(
        attn__kind="GQA, SWA",
        attn__global_layers=1,
        attn__local_layers=5,
        attn__kv_heads=8,
        attn__head_dim=128,
        attn__window=1024,
    )
    d = derive(f, context=4096)
    assert d["kv.bytes_per_token"]["value"] == d["kv.bytes_at_context"]["value"] / 4096
    assert d["kv.bytes_per_token"]["value"] > d["kv.growth_per_token"]["value"]


def test_mla_with_sliding_window_layers_caches_the_latent():
    f = facts(
        attn__kind="MLA, SWA",
        attn__global_layers=2,
        attn__local_layers=6,
        attn__kv_lora_rank=512,
        attn__rope_dim=64,
        attn__window=128,
    )
    d = derive(f, context=4096)
    assert d["kv.bytes_at_context"]["value"] == 2 * 576 * 2 * 4096 + 6 * 576 * 2 * 128
