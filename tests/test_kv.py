import pytest

from archdb import kv
from archdb.data import load_models
from archdb.derive import derive
from tests.test_derive import facts


@pytest.fixture(scope="module")
def records():
    return {m.id: m for m in load_models()}


@pytest.mark.parametrize(
    "mid,expected",
    [
        ("deepseek-v32", 48_068),
        ("deepseek-v4-flash-base", 3_514),
        ("deepseek-v41", 890),
    ],
)
def test_global_storage_matches_deepseek_v41_figure_one(records, mid, expected):
    # arXiv:2609.19969v1, p.1 Figure 1(b), independently reported whole-byte labels.
    d = derive(records[mid].facts, kv_dtype="sourced")
    assert round(d["kv.growth_per_token"]["value"]) == expected
    assert d["kv.growth_per_token"]["formula_version"] == "kv-2"


def test_v41_keeps_swa_in_runtime_cache(records):
    d = derive(records["deepseek-v41"].facts, context=131_072, kv_dtype="sourced")
    # 40 independent 128-entry SWA rings, 528 bytes/entry (FlashMLA's V4.1 FP8 layout).
    assert d["kv.bytes_at_context"]["value"] == 890 * 131_072 + 40 * 128 * 528
    assert d["kv.bytes_per_token"]["value"] == 910.625
    assert d["kv.bytes_per_token_per_active"]["value"] == 910.625 / 16


def test_v41_uniform_fp4_is_not_its_sourced_storage_recipe(records):
    f = records["deepseek-v41"].facts
    assert derive(f)["kv.growth_per_token"]["value"] == 3200
    assert derive(f, kv_dtype="fp4")["kv.growth_per_token"]["value"] == 800
    assert derive(f, kv_dtype="sourced")["kv.growth_per_token"]["value"] == 890


def test_v4_pro_uses_its_own_layer_counts(records):
    d = derive(records["deepseek-v4-pro"].facts, kv_dtype="sourced")
    assert d["kv.growth_per_token"]["value"] == 5_031.4375
    assert d["kv.bytes_at_context"]["value"] == 5_031.4375 * 131_072 + 61 * 128 * 584


@pytest.mark.parametrize(
    "mid,global_bytes,local_owners",
    [
        ("mimo-v2-flash-base", 23_040, 39),
        ("mimo-v2-6-flash", 23_040, 39),
        ("mimo25-pro-base", 51_200, 60),
        ("mimo-v2-6-pro", 51_200, 60),
    ],
)
def test_mimo_swa_has_different_key_and_value_dimensions(records, mid, global_bytes, local_owners):
    # Official configs: swa_num_key_value_heads=8, swa_head_dim=192, swa_v_head_dim=128.
    d = derive(records[mid].facts, context=131_072)
    assert d["kv.growth_per_token"]["value"] == global_bytes
    assert (
        d["kv.bytes_at_context"]["value"] == global_bytes * 131_072 + local_owners * 8 * (192 + 128) * 2 * 128
    )


def test_local_value_dimension_overrides_global():
    f = facts(
        attn__global_layers=1,
        attn__local_layers=1,
        attn__kv_heads=2,
        attn__head_dim=192,
        attn__v_head_dim=128,
        attn__local_head_dim=96,
        attn__local_v_head_dim=64,
        attn__window=4,
    )
    assert (
        derive(f, context=8)["kv.bytes_at_context"]["value"]
        == 2 * (192 + 128) * 2 * 8 + 2 * (96 + 64) * 2 * 4
    )


def test_compressed_entries_are_whole_and_incomplete_tail_is_excluded():
    f = facts(
        attn__kv_groups=[
            {"name": "compressed", "layers": 2, "elements": 512, "tokens_per_entry": 4},
            {"name": "local", "layers": 2, "elements": 512, "window": 4},
        ]
    )
    assert derive(f, context=3)["kv.bytes_at_context"]["value"] == 2 * 512 * 2 * 3
    assert derive(f, context=7)["kv.bytes_at_context"]["value"] == 2 * 512 * 2 * (1 + 4)


def test_partial_storage_recipe_does_not_fall_back_to_uniform_precision():
    f = facts(
        attn__kv_groups=[{"name": "main", "layers": 1, "elements": 512}],
        attn__kv_precision=[{"name": "partial", "group": "main", "elements": 256, "bits": 8}],
    )
    assert derive(f)["kv.growth_per_token"]["value"] == 1024
    assert derive(f, kv_dtype="sourced")["kv.growth_per_token"]["value"] is None
    assert "sourced" not in derive(f)["_kv"]


def test_cache_at_rejects_zero_context():
    with pytest.raises(ValueError, match="positive"):
        kv.at({"groups": []}, 0)


def test_fp4_storage_packs_two_values_per_byte():
    f = facts(
        attn__kv_groups=[{"name": "main", "layers": 1, "elements": 33}],
        attn__kv_precision=[{"name": "main", "group": "main", "elements": 33, "bits": 4}],
    )
    assert derive(f, context=1, kv_dtype="sourced")["kv.bytes_at_context"]["value"] == 17
