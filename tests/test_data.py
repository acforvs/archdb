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
