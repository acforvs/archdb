import pytest

from archdb.sources import POINTER, lookup, normalize, same

DOC = {
    "layer_types": ["sliding", "sliding", "full", "sliding", "full"],
    "a": 2,
    "b": 3,
    "pattern": "MEM*E",
    "weight_map": {"model.layers.0.mlp.weight": "a", "lm_head.weight": "b"},
}


@pytest.mark.parametrize(
    ("at", "expected"),
    [
        ("/a", 2),
        ("/a + /b", 5),
        ("/layer_types | count full", 2),
        ("/layer_types | len", 5),
        ("/layer_types/0:3 | count sliding", 2),
        ("/pattern | count M", 2),
        ("/weight_map | has mtp.", False),
        ("/weight_map | has lm_head", True),
    ],
)
def test_pointer_forms(at, expected):
    assert POINTER.match(at)
    assert lookup(DOC, at) == expected


def test_free_text_location_is_not_a_pointer():
    assert not POINTER.match("p.22 §4.2")


def test_normalize_handles_pdf_hyphenation_and_markup():
    assert normalize("pre-\ntraining   on 14.8T") == normalize("pretraining on 14.8T")
    assert normalize('<td style="x">501.3B</td>', html=True) == "501.3b"
    assert normalize("if a<b and c>d", html=True) == "if a<b and c>d"


def test_same_compares_numbers_and_labels():
    assert same(1e-4, 0.0001)
    assert same("SiLU", "silu")
    assert not same(True, 1)


def test_normalize_curly_quotes():
    assert normalize("each layer’s output") == normalize("each layer's output")


def test_config_leaves_skip_ignored_and_empty_values():
    from archdb.sources import config_leaves, ignored_config_keys

    doc = {
        "hidden_size": 4096,
        "bos_token_id": 1,
        "final_logit_softcapping": None,
        "use_qkv_bias": False,
        "vision_config": {"hidden_size": 1024},
        "rope_parameters": {"rope_type": "default", "rope_theta": 1e6},
    }
    assert dict(config_leaves(doc, ignored_config_keys())) == {"hidden_size": 4096, "rope_theta": 1e6}
