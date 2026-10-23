from copy import deepcopy

import pytest

from archdb import query
from archdb.explorer import Explorer, layer_glyph, layer_layout, links, mixer_kind, plot_points, safe_url
from archdb.tui import Application
from tests.test_query import model


@pytest.fixture
def state():
    state = Explorer(query.load())
    state.refresh(False)
    return state


def test_marked_models_survive_search_and_can_be_unmarked(state):
    state.search = "deepseek-v3"
    state.refresh()
    state.toggle()
    marked = state.model["id"]
    state.search = "kimi"
    state.refresh()
    assert marked in [m["id"] for m in state.comparison]
    state.selected.remove(marked)
    assert state.comparison == []


def test_empty_custom_selection_stays_empty(state):
    state.custom_fields = []
    assert state.field_keys() == []


def test_filter_errors_do_not_mutate_current_selection(state):
    previous = list(state.models)
    with pytest.raises(ValueError):
        state.set_filter("made.up >= 10")
    assert state.models == previous and state.filters == []


def test_mixed_counts_are_never_visualized_as_an_ordered_sequence():
    m = model("mixed", {"attn.global_layers": 2, "attn.local_layers": 6, "attn.linear_layers": 0})
    sequence, note = layer_layout(m)
    assert sequence == []
    assert "order not sourced" in note
    sequence = ["SWA", "GQA", "SWA", "SWA", "GQA", "SWA", "SWA", "SWA"]
    m["facts"]["attn.layout"] = {"value": "3:1", "sequence": sequence}
    assert layer_layout(m) == (sequence, "sourced order")


def test_unknown_layer_counts_are_not_assumed_zero():
    m = model("partial", {"attn.global_layers": 10})
    assert layer_layout(m)[0] == []
    m["facts"].update(model("complete", {"attn.local_layers": 0, "attn.linear_layers": 0})["facts"])
    assert layer_layout(m)[0] == ["global"] * 10


def test_hybrid_compressed_attention_keeps_its_swa_branch():
    assert mixer_kind("CSA2 m=2 + SWA") == "hybrid"
    assert mixer_kind("Mamba-2") == "linear"
    assert mixer_kind("SWA") == "local"
    assert layer_glyph("CSA2 m=2 + SWA") != layer_glyph("CSA2 m=1 + SWA")
    assert layer_glyph("CSA + SWA") != layer_glyph("HCA + SWA")
    m = model(
        "compressed",
        {
            "attn.kind": "CSA2 + SWA",
            "shape.layers": 10,
            "attn.global_layers": 10,
            "attn.local_layers": 0,
            "attn.linear_layers": 0,
        },
    )
    assert layer_layout(m)[0] == []


def test_hf_link_targets_checkpoint_and_config_is_pinned(state):
    hrefs = links(query.resolve(state.db, "deepseek-v3"))
    assert hrefs["Hugging Face"] == "https://huggingface.co/deepseek-ai/DeepSeek-V3-Base"
    assert "/blob/" in hrefs["Config"]
    assert hrefs["Report"].startswith("https://arxiv.org/pdf/")
    assert not safe_url("javascript:alert(1)")
    assert not safe_url("file:///etc/passwd")
    assert not safe_url("https://user:password@example.org")


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "https://user:password@example.org",
        "https://example.org/\x1b]52;c;clipboard\x07",
        "https://example.org/\nconfig.json",
        "https://example.org/\u202econfig.json",
        "https://[invalid",
        "https://example.org:invalid/config.json",
        None,
    ],
)
def test_invalid_source_urls_never_reach_browser(state, monkeypatch, url):
    monkeypatch.setattr("webbrowser.open", lambda *_: pytest.fail("invalid source URL opened"))
    assert not safe_url(url)
    app(state).open_url(url)


@pytest.mark.parametrize("symlink", [False, True])
def test_export_does_not_overwrite_files_or_follow_existing_symlinks(state, tmp_path, monkeypatch, symlink):
    original = tmp_path / "original.json"
    original.write_text("keep this file")
    target = tmp_path / "selection.json" if symlink else original
    if symlink:
        target.symlink_to(original)
    ui = app(state)
    monkeypatch.setattr(ui, "prompt", lambda *_: str(target))
    with pytest.raises(FileExistsError):
        ui.key("e")
    assert original.read_text() == "keep this file"


def test_plot_omits_missing_and_nonpositive_log_values():
    models = [
        model("a", {"x": 1, "y": 10}),
        model("b", {"x": 100, "y": 1000}),
        model("zero", {"x": 0, "y": 10}),
        model("missing"),
    ]
    points, bounds = plot_points(models, "x", "y", 21, 11, True)
    assert [(p["model"]["id"], p["col"], p["row"]) for p in points] == [("a", 0, 10), ("b", 20, 0)]
    assert bounds == (0, 2, 1, 3)
    assert len(plot_points(models, "x", "y", 21, 11, False)[0]) == 3


class Screen:
    def __init__(self, keys=(), size=(40, 128)):
        self.keys = iter(keys)
        self.size = size
        self.writes = []

    def getmaxyx(self):
        return self.size

    def addnstr(self, y, x, text, limit, attr):
        assert 0 <= y < self.size[0] and 0 <= x < self.size[1]
        self.writes.append(text[:limit])

    def erase(self):
        pass

    def refresh(self):
        pass

    def get_wch(self):
        import curses

        try:
            return next(self.keys)
        except StopIteration as exc:
            raise curses.error("no input") from exc

    def timeout(self, value):
        pass


def app(state, keys=(), size=(40, 128)):
    instance = Application(state)
    instance.screen = Screen(keys, size)
    return instance


@pytest.mark.parametrize("size", [(22, 78), (30, 100), (40, 128), (14, 60)])
def test_views_render_at_multiple_terminal_sizes(state, size, monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda *a, **kw: pytest.fail("drawing must not open a browser"))
    state.selected = {m["id"] for m in state.models[:3]}
    ui = app(state, size=size)
    for view in ("Models", "Compare", "Plot"):
        state.view = view
        ui.draw()


def test_field_picker_can_clear_every_field(state):
    ui = app(state, ["n", "\n"])
    ui.field_picker()
    assert state.custom_fields == []
    ui.draw()


def test_field_picker_selects_group_then_deselects_one_field(state):
    import curses

    ui = app(state, ["n", " ", curses.KEY_RIGHT, curses.KEY_DOWN, " ", "\n"])
    ui.field_picker()
    assert "meta.released" not in state.custom_fields
    assert {"meta.license", "meta.modality"} <= set(state.custom_fields)


def test_plot_model_inspection_returns_to_plot(state):
    ui = app(state)
    state.view = "Plot"
    previous = state.model["id"]
    ui.enter()
    assert state.view == "Models" and state.model["id"] == previous
    ui.key("\x1b")
    assert state.view == "Plot" and state.model["id"] == previous


def test_kv_controls_recompute_the_same_sourced_recipe(state):
    previous = deepcopy(query.resolve(state.db, "deepseek-v41")["derived"])
    state.set_kv(262144, "sourced")
    m = query.resolve(state.db, "deepseek-v41")
    assert m["derived"]["kv.bytes_at_context"]["value"] == 890 * 262144 + 40 * 128 * 528
    assert previous["shape.aspect"] == m["derived"]["shape.aspect"]


def test_all_fields_and_live_field_search_are_the_default(state):
    assert state.section == "All fields"
    assert len(state.field_keys()) == len(state.db["fields"])
    ui = app(state, [*"KV heads", "\n"])
    ui.key("/")
    assert state.field_search == "KV heads"
    assert {"attn.kv_heads", "attn.local_kv_heads"} <= set(state.field_keys())
    assert "train.optimizer" not in state.field_keys()
    ui.screen = Screen(["\x15", "\n"])
    ui.key("/")
    assert len(state.field_keys()) == len(state.db["fields"])


def test_model_search_does_not_clear_field_search(state):
    state.field_search = "expert"
    ui = app(state, [*"deepseek", "\n"])
    ui.key("m")
    assert state.models and all("deepseek" in m["id"] for m in state.models)
    assert state.field_search == "expert"
    assert "moe.expert_dim" in state.field_keys()


def test_numeric_sort_selects_largest_first_and_shows_sorted_value(state):
    state.cursor = 20
    ui = app(state, [*"params.total", "\n"])
    ui.key("s")
    assert state.sort == "params.total" and state.descending
    assert state.cursor == 0
    values = [query.cell(m, "params.total")["value"] for m in state.models]
    known = [v for v in values if isinstance(v, (int, float))]
    assert known == sorted(known, reverse=True)
    ui.draw()
    assert "TOTAL" in ui.screen.writes
    assert "largest/newest first" in ui.notice
    ui.key("d")
    assert not state.descending and state.cursor == 0


def test_h_opens_hugging_face_in_models_and_compare(state, monkeypatch):
    urls = []
    monkeypatch.setattr("webbrowser.open", urls.append)
    state.cursor = next(i for i, m in enumerate(state.models) if m["id"] == "deepseek-v3")
    ui = app(state)
    ui.key("h")
    assert urls[-1] == "https://huggingface.co/deepseek-ai/DeepSeek-V3-Base"
    state.selected = {"deepseek-v3", "kimi-k2"}
    state.view = "Compare"
    ui.compare_col = 0
    ui.key("h")
    assert urls[-1] == links(state.comparison[0])["Hugging Face"]
    ui.key("c")
    assert urls[-1] == links(state.comparison[0])["Config"]
    assert "/blob/" in urls[-1]


def test_mouse_wheel_immediately_scrolls_fields_without_prior_focus(state):
    from archdb.keys import Mouse

    ui = app(state)
    ui.open_groups = set(state.db["groups"])
    ui.show_missing = True
    ui.focus = "models"
    ui.draw()
    ui.key(Mouse(100, 20, "down"))
    ui.draw()
    assert ui.field_scroll == 3
    assert ui.focus == "detail"
    ui.key(Mouse(100, 20, "up"))
    ui.draw()
    assert ui.field_scroll == 0


def test_derived_values_use_words_and_plot_names_start_visible(state):
    ui = app(state)
    assert ui.labels
    m = query.resolve(state.db, "deepseek-v3")
    text = state.value(m, "params.counted_non_embedding", annotate=True)
    assert "(calculated)" in text
    assert "ƒ" not in text


def test_arrows_focus_panes_and_g_offers_searchable_sections(state):
    import curses

    ui = app(state)
    ui.key(curses.KEY_RIGHT)
    assert ui.focus == "detail"
    ui.key(curses.KEY_LEFT)
    assert ui.focus == "models"
    assert state.section == "All fields"
    ui.screen = Screen([*"attention", "\n"])
    ui.key("g")
    assert state.section == "Attention"


def test_collapsed_groups_expand_and_search_reveals_matching_fields(state):
    ui = app(state)
    model = state.model
    rows = ui.detail_rows(model, 75)
    assert any(row[2] == "@group.attn" for row in rows)
    assert not any(row[2] == "attn.kind" for row in rows)
    ui.inspect(model, "@group.attn")
    assert any(row[2] == "attn.kind" for row in ui.detail_rows(model, 75))
    ui.open_groups.clear()
    state.field_search = "token mixers"
    assert any(row[2] == "attn.kind" for row in ui.detail_rows(model, 75))


def test_missing_values_are_hidden_until_searched_or_requested(state):
    ui = app(state)
    ui.open_groups = set(state.db["groups"])
    model = state.model
    keys = state.field_keys()
    missing = next(k for k in keys if query.cell(model, k)["value"] is None)
    assert not any(row[2] == missing for row in ui.detail_rows(model, 75))
    ui.key("u")
    assert any(row[2] == missing for row in ui.detail_rows(model, 75))
    ui.key("u")
    state.field_search = missing
    assert any(row[2] == missing for row in ui.detail_rows(model, 75))


def test_batch_size_sort_uses_numeric_records_instead_of_batch_notes(state):
    ui = app(state, [*"batch size", "\n"])
    ui.key("s")
    assert state.sort == "train.batch_tokens" and state.descending
    assert state.model["id"] == "trinity-large"
    known = [query.cell(m, state.sort)["value"] for m in state.models]
    count = sum(value is not None for value in known)
    assert count == 18
    assert known[:count] == sorted(known[:count], reverse=True)
    assert all(value is None for value in known[count:])
    ui.draw()
    assert "BATCH" in ui.screen.writes
    assert "18/62 known" in ui.notice


def test_enter_on_model_shows_every_field_without_changing_model(state):
    state.field_search = "expert"
    state.custom_fields = ["moe.expert_dim"]
    ui = app(state)
    selected = state.model["id"]
    assert ui.focus == "models"
    ui.enter()
    assert state.model["id"] == selected and ui.focus == "detail"
    assert state.field_search == "" and state.custom_fields is None
    assert state.section == "All fields" and ui.show_missing
    rows = ui.detail_rows(state.model, 75)
    shown = {row[2] for row in rows if row[2] in state.db["fields"]}
    assert shown == set(state.db["fields"])


def test_right_arrow_keeps_field_selection_and_view_menu_lists_all_views(state):
    import curses

    from archdb.explorer import SECTIONS

    state.field_search = "expert"
    ui = app(state)
    ui.key(curses.KEY_RIGHT)
    assert state.field_search == "expert" and state.section == "All fields"
    ui.screen = Screen([curses.KEY_RIGHT, "\n"])
    ui.key("g")
    assert state.section == "Overview"
    assert all(any(name in text for text in ui.screen.writes) for name in SECTIONS)


def test_common_controls_are_labeled_in_the_main_screen(state):
    ui = app(state)
    ui.draw()
    text = "\n".join(ui.screen.writes)
    assert all(
        label in text for label in ["1 Models", "2 Compare", "f Filter", "a Show all fields", "g Views"]
    )
    assert all(label in text for label in ["All fields", "Overview", "Attention", "MoE", "Recipe", "KV"])
