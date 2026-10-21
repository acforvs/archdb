from archdb.data import Model, load_fields, load_models, required, validate

_, FIELDS = load_fields()


def model(**facts):
    return Model(
        id="m",
        name="M",
        org="O",
        kind="base",
        sources={
            "config": {"url": f"https://huggingface.co/o/m/resolve/{'0' * 40}/config.json", "kind": "config"}
        },
        facts=facts,
    )


def test_repository_data_is_valid():
    models = load_models()
    assert models
    assert validate(models, FIELDS) == []


def test_required_conditions_exist():
    from archdb.data import CONDITIONS

    assert {f["required"] for f in FIELDS.values() if "required" in f} <= set(CONDITIONS)


def test_required_field_must_be_stated_or_not_disclosed():
    m = model(**{"moe.experts": {"value": 8, "source": "config", "at": "/n"}})
    assert any("moe.top_k" in e for e in validate([m], FIELDS, orgs={"O": {}}))
    m.not_disclosed = [k for k in required(m, FIELDS) if k not in m.facts]
    assert not any("required" in e for e in validate([m], FIELDS, orgs={"O": {}}))


def test_rejects_unknown_field():
    assert any("unknown field" in e for e in validate([model(**{"shape.depth": {"value": 1}})], FIELDS))


def test_rejects_authored_derived_field():
    fact = {"value": 80.0, "source": "config", "at": "/x"}
    assert any("derived" in e for e in validate([model(**{"shape.aspect": fact})], FIELDS))


def test_rejects_undefined_source_and_missing_location():
    errors = validate([model(**{"shape.layers": {"value": 61, "source": "paper"}})], FIELDS)
    assert any("undefined source" in e for e in errors)
    assert any("location" in e for e in errors)


def test_inferred_requires_note():
    fact = {"value": 61, "source": "config", "at": "/n", "basis": "inferred"}
    assert any("needs a note" in e for e in validate([model(**{"shape.layers": fact})], FIELDS))


def test_type_checks():
    fact = {"value": "61", "source": "config", "at": "/n"}
    assert any("must be int" in e for e in validate([model(**{"shape.layers": fact})], FIELDS))


def test_rejects_unpinned_source():
    m = model(**{"shape.layers": {"value": 61, "source": "blog", "at": "x", "quote": "61"}})
    m.sources["blog"] = {"url": "https://example.org/post", "kind": "blog"}
    assert any("not pinned" in e for e in validate([m], FIELDS, orgs={"O": {}}))


def test_stage_numbers_must_be_numbers_or_bounds():
    stage = {
        "value": [{"name": "main", "tokens": "500B → 200B", "quote": "q"}],
        "source": "config",
        "at": "x",
    }
    assert any(
        "must be a number" in e for e in validate([model(**{"train.stages": stage})], FIELDS, orgs={"O": {}})
    )
    stage["value"][0]["tokens"] = ">35T"
    assert not any(
        "must be a number" in e for e in validate([model(**{"train.stages": stage})], FIELDS, orgs={"O": {}})
    )


def test_cache_geometry_requires_whole_positive_counts():
    from archdb.data import _check_kv

    for key, value in [("layers", -1), ("elements", True), ("tokens_per_entry", 0), ("window", 1.5)]:
        group = {"name": "main", "layers": 1, "elements": 512, key: value}
        assert _check_kv("test:", {"attn.kv_groups": {"value": [group]}})


def test_cache_precision_checks_references_and_component_coverage():
    from archdb.data import _check_kv

    group = {"name": "main", "layers": 1, "elements": 512}
    component = {
        "name": "latent",
        "group": "main",
        "elements": 512,
        "bits": 4,
        "format": "E2M1",
        "scale_count": 32,
        "scale_bits": 8,
    }
    f = {"attn.kv_groups": {"value": [group]}, "attn.kv_precision": {"value": [component]}}
    assert _check_kv("test:", f) == []
    for key, value in [
        ("name", ""),
        ("group", "typo"),
        ("elements", 513),
        ("bits", 5),
        ("scale_bits", 0),
        ("padding_bytes", -1),
    ]:
        f["attn.kv_precision"]["value"] = [{**component, key: value}]
        assert _check_kv("test:", f)
