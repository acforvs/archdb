import json

import pytest

from archdb import query


def model(mid, facts=None, undisclosed=()):
    return {
        "id": mid,
        "name": mid,
        "org": "Lab",
        "kind": "base",
        "weights": "open",
        "sources": {"config": {"kind": "config", "url": "https://example.org/config.json"}},
        "facts": {k: {"value": v, "source": "config", "at": "/" + k} for k, v in (facts or {}).items()},
        "derived": {},
        "not_disclosed": list(undisclosed),
    }


@pytest.fixture
def db():
    return {
        "fields": {
            "params.total": {"type": "float"},
            "attn.ratio": {"type": "float", "unit": "ratio"},
            "attn.gate": {"type": "bool"},
            "attn.kind": {"type": "str"},
            "train.tokens": {"type": "float"},
            "meta.released": {"type": "date"},
        },
        "models": [
            model(
                "small",
                {
                    "params.total": "670B",
                    "attn.ratio": "16:2",
                    "attn.gate": False,
                    "attn.kind": "GQA, SWA",
                    "meta.released": "2025-12-01",
                },
            ),
            model(
                "large",
                {
                    "params.total": "1T",
                    "attn.ratio": "16:1",
                    "attn.gate": True,
                    "attn.kind": "MLA",
                    "meta.released": "2026-01-04",
                },
            ),
            model("undisclosed", undisclosed=["attn.gate"]),
            model("unknown"),
        ],
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        ("670B", 670e9),
        ("1T", 1e12),
        ("16:2", 8),
        ("16:1", 16),
        ("2 GiB", 2 * 1024**3),
        ("32x", 32),
        ("10%", 0.1),
        ("2e-4", 0.0002),
        ("1,024", 1024),
    ],
)
def test_numeric_literals(text, expected):
    assert query.number(text) == expected


@pytest.mark.parametrize("value", [True, False, "NaN", "inf", "1e999", "2:0", "one billion", ">35T"])
def test_non_exact_numbers_are_not_silently_coerced(value):
    assert query.number(value) is None


def test_sort_uses_scale_and_ratio_values_and_keeps_missing_last(db):
    assert [m["id"] for m in query.select(db, sort="params.total", descending=True)] == [
        "large",
        "small",
        "undisclosed",
        "unknown",
    ]
    assert [m["id"] for m in query.select(db, sort="attn.ratio")] == [
        "small",
        "large",
        "undisclosed",
        "unknown",
    ]


def test_filters_require_every_condition(db):
    selected = query.select(db, filters=["params.total >= 500B", "attn.ratio > 16:2", "attn.gate = true"])
    assert [m["id"] for m in selected] == ["large"]
    assert query.select(db, filters=["attn.kind contains swa"])[0]["id"] == "small"
    assert query.select(db, filters=["meta.released >= 2026-01"])[0]["id"] == "large"


def test_missing_is_distinct_from_false_zero_and_undisclosed(db):
    assert query.cell(db["models"][0], "attn.gate")["status"] == "config"
    assert [m["id"] for m in query.select(db, filters=["attn.gate = false"])] == ["small"]
    assert [m["id"] for m in query.select(db, filters=["attn.gate is unknown"])] == ["unknown"]
    assert [m["id"] for m in query.select(db, filters=["attn.gate is not-disclosed"])] == ["undisclosed"]
    assert [m["id"] for m in query.select(db, filters=["attn.gate != true"])] == ["small"]
    m = model("zero", {"params.total": 0, "attn.kind": "None"})
    assert query.predicate(db, "params.total is known")(m)
    assert query.predicate(db, "attn.kind = None")(m)


def test_reported_bounds_only_match_conclusive_comparisons(db):
    bounded = model("bound", {"train.tokens": ">35T"})
    assert query.predicate(db, "train.tokens > 30T")(bounded)
    assert query.predicate(db, "train.tokens > 35T")(bounded)
    assert not query.predicate(db, "train.tokens > 40T")(bounded)
    assert not query.predicate(db, "train.tokens = 35T")(bounded)
    approximate = model("approx", {"train.tokens": "~12T"})
    assert not query.predicate(db, "train.tokens = 12T")(approximate)
    assert query.predicate(db, "train.tokens is known")(approximate)


@pytest.mark.parametrize(
    "expression",
    [
        "params.typo >= 100B",
        "params.total >= infinity",
        "attn.gate > true",
        "attn.kind >= MLA",
        "attn.gate is probably",
        "params.total",
    ],
)
def test_bad_filters_fail_instead_of_returning_unfiltered_data(db, expression):
    with pytest.raises(ValueError):
        query.select(db, filters=[expression])


def test_model_resolution_never_silently_chooses_a_variant(db):
    assert query.resolve(db, "large")["id"] == "large"
    with pytest.raises(ValueError, match="several models"):
        query.resolve(db, "Lab")
    with pytest.raises(ValueError, match="no model"):
        query.resolve(db, "missing")


def test_bad_or_missing_export_is_reported(tmp_path):
    path = tmp_path / "dataset.json"
    with pytest.raises(ValueError, match="cannot read"):
        query.load(path)
    path.write_text(json.dumps({"models": [], "fields": []}))
    with pytest.raises(ValueError, match="not an ArchDB"):
        query.load(path)


def test_kv_uses_existing_sourced_formula_and_preserves_checkpoint_counts():
    db = query.load(precision="sourced")
    m = query.resolve(db, "deepseek-v41")
    assert query.cell(m, "kv.bytes_per_token")["value"] == 910.625
    assert query.cell(m, "kv.growth_per_token")["value"] == 890
    assert query.cell(m, "kv.bytes_per_token")["formula_version"] == "kv-2"
    assert query.resolve(db, "deepseek-v3")["derived"]["params.counted_non_embedding"]["value"] > 600e9
