import json

import pytest

from archdb.checkpoint import SIDE, TEXT_PREFIX, classify, elements
from archdb.data import Model
from archdb.derive import counted


@pytest.mark.parametrize(
    ("name", "part"),
    [
        ("model.embed_tokens.weight", "embedding"),
        ("tok_embeddings.weight", "embedding"),
        ("embed.weight", "embedding"),
        ("lm_head.weight", "head"),
        ("unembed.weight", "head"),
        ("output.weight", "head"),
        ("model.layers.3.mlp.experts.17.gate_proj.weight", "routed"),
        ("model.layers.3.mlp.experts.gate_up_proj", "routed"),
        ("model.layers.3.mlp.shared_experts.gate_proj.weight", "dense"),
        ("model.layers.3.mlp.gate.weight", "dense"),
        ("model.layers.61.eh_proj.weight", "mtp"),
        ("mtp.layers.0.self_attn.q_proj.weight", "mtp"),
        ("model.visual.blocks.0.attn.qkv.weight", "other_modality"),
        ("model.engram.tables.0.weight", "lookup"),
    ],
)
def test_classify(name, part):
    assert classify(TEXT_PREFIX.sub("", name), layers=61) == part


def test_multimodal_prefix_is_stripped():
    assert TEXT_PREFIX.sub("", "model.language_model.embed_tokens.weight") == "embed_tokens.weight"
    assert TEXT_PREFIX.sub("", "model.llm.unembed.weight") == "unembed.weight"


def test_packed_and_side_tensors():
    assert elements("x.down_proj_blocks", {"dtype": "U8", "shape": [4, 90, 16]}) == 4 * 90 * 16 * 2
    assert elements("x.weight", {"dtype": "I8", "shape": [10, 8]}, mx_scaled=True) == 160
    assert elements("x.weight", {"dtype": "BF16", "shape": [10, 8]}) == 80
    assert SIDE.search("q_a_proj.weight_scale_inv")
    assert SIDE.search("experts.0.w1.scale")
    assert not SIDE.search("q_a_proj.weight")


COUNTS = {
    "repo": "o/m",
    "commit": "0" * 40,
    "embedding": 1e9,
    "head": 1e9,
    "lookup": 0,
    "routed": 96e9,
    "dense": 2e9,
    "mtp": 1e9,
    "other_modality": 0,
}


def facts(**kv):
    return {k.replace("__", "."): {"value": v} for k, v in kv.items()}


def test_counted_active_uses_top_k_share_and_scope_check():
    d = counted(facts(moe__experts=96, moe__top_k=8, params__total=100e9, params__active=12e9), COUNTS)
    assert d["params.counted_active_non_embedding"]["value"] == 2e9 + 96e9 * 8 / 96
    assert d["params.counted_total"]["value"] == 100e9
    assert d["params.counted_non_embedding"]["value"] == 98e9
    assert "params.counted_max_layer" not in d


def test_counted_max_layer_when_counted():
    counts = {**COUNTS, "max_layer": 2.5e9}
    d = counted(facts(moe__experts=96, moe__top_k=8, params__total=100e9, params__active=12e9), counts)
    assert d["params.counted_max_layer"]["value"] == 2.5e9


def test_counted_rejects_counts_that_do_not_reproduce_stated_totals():
    d = counted(facts(moe__experts=96, moe__top_k=8, params__total=150e9, params__active=12e9), COUNTS)
    assert d["params.counted_active_non_embedding"]["value"] is None
    assert d["params.counted_non_embedding"]["value"] is None


def test_max_layer_keeps_encoder_and_decoder_blocks_separate(monkeypatch, tmp_path):
    from archdb import checkpoint

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"num_hidden_layers": 2}))
    monkeypatch.setattr(checkpoint, "fetch", lambda _: config)
    monkeypatch.setattr(checkpoint, "shards", lambda *_: ["model.safetensors"])
    shapes = {
        "model.encoder.layers.0.attn.weight": [2, 3],
        "model.decoder.layers.0.attn.weight": [3, 4],
        "model.decoder.layers.0.engram.weight": [100, 4],
        "model.decoder.layers.2.mlp.weight": [200, 4],
    }
    monkeypatch.setattr(checkpoint, "header", lambda _: {
        name: {"dtype": "BF16", "shape": shape} for name, shape in shapes.items()
    })
    model = Model("test", "test", "test", "base", {
        "config": {"kind": "config", "url": f"https://huggingface.co/o/m/resolve/{'0' * 40}/config.json"},
    }, {})
    result = checkpoint.count(model)
    assert result["max_layer"] == 12
    assert result["dense"] == 18
    assert result["lookup"] == 400
    assert result["mtp"] == 800
