"""Curses interface for the local dataset."""

from __future__ import annotations

import curses
import json
import locale
import sys
import textwrap
import webbrowser
from pathlib import Path

from . import query
from .explorer import PLOTS, SECTIONS, Explorer, layer_glyph, layer_layout, links, mixer_kind, safe_url
from .keys import Mouse, read_key
from .terminal import clean, display


class Application:
    def __init__(self, state: Explorer):
        self.state = state
        self.focus = "models"
        self.field_cursor = 0
        self.field_scroll = 0
        self.compare_col = 0
        self.compare_start = 0
        self.notice = ""
        self.hits = []
        self.colors = {}
        self.plotted = []
        self.labels = True
        self.return_view = None
        self.open_groups = {"params", "shape"}
        self.search_collapsed = set()
        self.show_missing = False

    def put(self, y, x, text, style="normal", width=None):
        height, columns = self.screen.getmaxyx()
        if y < 0 or y >= height or x < 0 or x >= columns - 1:
            return
        limit = max(0, min(columns - x - 1, width if width is not None else columns))
        try:
            self.screen.addnstr(
                y, x, clean(text).replace("\n", " ").replace("\t", " "), limit, self.colors.get(style, 0)
            )
        except curses.error:
            pass

    def rule(self, y, x, width, title=""):
        self.put(y, x, "─" * max(0, width), "muted", width)
        if title:
            self.put(y, x + 1, f" {title} ", "accent", width - 2)

    def button(self, y, x, text, action, style="normal"):
        self.put(y, x, text, style)
        self.hits.append((y, x, len(text), action))

    def wrap(self, text, width):
        return [
            piece
            for line in clean(text).splitlines() or [""]
            for piece in textwrap.wrap(line, max(8, width)) or [""]
        ]

    def header(self):
        _, width = self.screen.getmaxyx()
        self.put(0, 2, "archdb", "accent")
        x = 12
        views = ("Models", "Compare", "Plot") if self.state.view == "Plot" else ("Models", "Compare")
        for view in views:
            name = (
                f"Compare ({len(self.state.selected)})" if view == "Compare" and self.state.selected else view
            )
            label = f" {('Models', 'Compare', 'Plot').index(view) + 1} {name} "
            self.button(0, x, label, ("view", view), "selected" if self.state.view == view else "muted")
            x += len(label) + 2
        self.button(0, width - 10, "? Help", ("help", None), "muted")
        split = min(51, max(33, width * 2 // 5))
        text = "m models: " + (self.state.search or "search")
        self.button(1, 2, text[: split - 4], ("search", None), "normal" if self.state.search else "muted")
        if self.state.view != "Plot":
            text = "/ fields: " + (self.state.field_search or "search")
            self.button(
                1,
                split + 3,
                text[: width - split - 5],
                ("field-search", None),
                "normal" if self.state.field_search else "muted",
            )
        else:
            precision = self.state.db["_kv"]["precision"]
            self.put(
                1, split + 3, f"KV {self.state.db['_kv']['context']:,} tokens · {precision.upper()}", "muted"
            )
        if self.state.filters:
            self.put(2, 2, " AND ".join(self.state.filters), "gold")

    def draw(self):
        self.screen.erase()
        self.hits = []
        height, width = self.screen.getmaxyx()
        if height < 22 or width < 78:
            self.put(1, 2, "archdb needs at least 78 columns × 22 rows.", "accent")
            self.put(3, 2, "Resize the terminal, or press q to quit.")
            self.screen.refresh()
            return
        self.header()
        if self.state.view == "Models":
            self.models_view()
        elif self.state.view == "Compare":
            self.compare_view()
        else:
            self.plot_view()
        self.rule(height - 3, 1, width - 2)
        hints = "↑↓ fields · ← models · / search · [ ] view · Enter inspect · ? keys"
        if self.state.view == "Plot":
            hints = "↑↓ model · Enter inspect · x/y axes · ? keys · q quit"
        elif self.state.view == "Compare":
            hints = "↑↓ field · ←→ model · / fields · Enter evidence · ? keys · q quit"
        elif self.focus == "models":
            hints = "↑↓ models · → fields · Enter show all · m search · space mark · ? keys"
        self.put(height - 2, 2, hints, "muted")
        if self.notice:
            self.put(height - 1, 2, self.notice, "accent")
        elif self.state.field_search:
            self.put(height - 1, 2, f"{len(self.state.field_keys())} matching fields", "muted")
        self.screen.refresh()

    def models_view(self):
        height, width = self.screen.getmaxyx()
        split = min(51, max(33, width * 2 // 5))
        self.split = split
        self.rule(4, 1, split - 2, f"Models · {len(self.state.models)}")
        sort_label = self.state.db["fields"].get(self.state.sort, {}).get("label", self.state.sort)
        self.button(
            3,
            2,
            f"s Sort: {sort_label} {'↓' if self.state.descending else '↑'}"[: split - 14],
            ("sort", None),
            "muted",
        )
        self.button(3, split - 10, "f Filter", ("filter", None), "muted")
        self.put(5, 3, "MODEL", "muted")
        sort_spec = self.state.db["fields"].get(self.state.sort, {})
        metric = (
            self.state.sort if sort_spec.get("type") in {"int", "float", "date", "bool"} else "params.active"
        )
        self.put(
            5,
            split - 11,
            {
                "params.active": "ACTIVE",
                "params.total": "TOTAL",
                "meta.released": "RELEASED",
                "train.batch_tokens": "BATCH",
            }.get(metric, "VALUE"),
            "muted",
        )
        capacity = height - 10
        start = max(0, min(self.state.cursor - capacity // 2, len(self.state.models) - capacity))
        for row, index in enumerate(range(start, min(len(self.state.models), start + capacity)), 6):
            model = self.state.models[index]
            focused = index == self.state.cursor
            style = "selected" if focused and self.focus == "models" else "accent" if focused else "muted"
            self.put(row, 2, " " * (split - 4), style)
            self.put(
                row, 2, "●" if model["id"] in self.state.selected else " ", "green" if not focused else style
            )
            name = clean(model["name"])
            name_width = split - 17
            name = name if len(name) <= name_width else name[: name_width - 1] + "…"
            self.put(row, 4, name, style, name_width)
            self.put(row, split - 11, self.state.value(model, metric), style, 9)
            self.hits.append((row, 2, split - 4, ("model", index)))
            self.hits.append((row, split - 11, 9, ("model-metric", (index, metric))))
        if not self.state.models:
            self.put(7, 3, "No matching models.", "muted")
            self.put(9, 3, "m searches; F clears filters.", "muted", split - 5)
        for y in range(4, height - 3):
            self.put(y, split, "│", "muted")
        if self.state.model:
            self.model_detail(self.state.model, split + 3, width - split - 6)

    def detail_rows(self, model, width):
        rows = []
        if (
            self.state.section == "Overview"
            and self.state.custom_fields is None
            and not self.state.field_search
        ):
            sequence, note = layer_layout(model)
            rows.append(("LAYERS · " + note, "accent", "attn.layout"))
            if sequence:
                size = max(1, width - 2)
                for start in range(0, len(sequence), size):
                    rows.append((sequence[start : start + size], "layers", "attn.layout"))
                legend = " · ".join(f"{layer_glyph(name)} {name}" for name in dict.fromkeys(sequence))
                rows.extend((line, "muted", "attn.layout") for line in self.wrap(legend, width))
            rows.append(("", "normal", None))
        keys = self.state.field_keys()
        if not self.show_missing and not self.state.field_search:
            keys = [k for k in keys if query.cell(model, k)["value"] is not None]
        if self.state.section != "All fields":
            for key in keys:
                rows.extend(self.field_rows(model, key, width))
            return rows
        priority = [
            "params",
            "shape",
            "attn",
            "ffn",
            "moe",
            "mtp",
            "norm",
            "train",
            "ctx",
            "kv",
            "meta",
            "scaling",
            "eval",
        ]
        groups = dict.fromkeys([*priority, *self.state.db["groups"]])
        for group in groups:
            members = [k for k in keys if k.startswith(group + ".")]
            if not members:
                continue
            opened = (
                group not in self.search_collapsed if self.state.field_search else group in self.open_groups
            )
            label = self.state.db["groups"][group]
            summary = "" if opened else self.group_summary(model, group)
            title = f"{'▾' if opened else '▸'} {label}" + ("  · " + summary if summary else "")
            rows.append((title, "accent" if opened else "muted", "@group." + group))
            if opened:
                if group == "kv":
                    rows.append((self.kv_label(), "muted", None))
                for key in members:
                    rows.extend(self.field_rows(model, key, width))
                rows.append(("", "normal", None))
        return rows

    def group_summary(self, model, group):
        choices = {
            "params": [("params.total", " total"), ("params.active", " active")],
            "shape": [("shape.layers", " layers"), ("shape.d_model", " wide")],
            "attn": [("attn.kind", "")],
            "ffn": [("ffn.dim", " intermediate")],
            "moe": [
                ("moe.experts", " experts"),
                ("moe.top_k", " per token"),
                ("moe.expert_dim", " intermediate"),
            ],
            "mtp": [("mtp.depth", " prediction depth")],
            "norm": [("norm.placement", "")],
            "train": [("train.tokens", " tokens")],
            "ctx": [("ctx.final", " tokens")],
            "meta": [("meta.modality", "")],
        }
        return " · ".join(
            self.state.value(model, key) + suffix
            for key, suffix in choices.get(group, [])
            if query.cell(model, key)["value"] is not None
        )

    def kv_label(self):
        precision = self.state.db["_kv"]["precision"]
        label = "sourced storage" if precision == "sourced" else "uniform " + precision.upper() + " payload"
        return f"{self.state.db['_kv']['context']:,} tokens · {label}"

    def field_rows(self, model, key, width):
        cell = query.cell(model, key)
        label = self.state.db["fields"][key]["label"]
        value = self.state.value(model, key, annotate=True)
        # Counted fields already identify their provenance in the field label.
        if key.startswith("params.counted_"):
            value = self.state.value(model, key)
        label_width = min(29, max(17, width // 2 - 2))
        labels = self.wrap(label, label_width)
        values = self.wrap(value, width - label_width - 2)
        return [
            (
                f"{labels[n] if n < len(labels) else '':<{label_width}}  {values[n] if n < len(values) else ''}",
                "normal" if cell["value"] is not None else "muted",
                key,
            )
            for n in range(max(len(labels), len(values)))
        ]

    def model_detail(self, model, x, width):
        height, _ = self.screen.getmaxyx()
        width = min(width, 100)
        self.button(3, x, "a Show all fields", ("show-all", None), "muted")
        title = self.wrap(model["name"], width)
        self.put(4, x, title[0], "title", width)
        self.put(
            5,
            x,
            f"{model.get('org', '')} · {model.get('kind', '')} · {model.get('weights', '')} weights",
            "muted",
            width,
        )
        position = x
        for label, url in links(model).items():
            shortcut = {"Hugging Face": "h", "Config": "c", "Report": "r"}[label]
            self.button(6, position, shortcut + " " + label + " ↗", ("url", url), "accent")
            position += len(label) + 5
        self.button(7, x, "g Views ▾", ("section-menu", None), "muted")
        if not self.state.field_search:
            label = "u Hide missing" if self.show_missing else "u Show missing"
            self.button(7, x + max(20, width - len(label)), label, ("missing", None), "muted")
        labels = {"KV cache": "KV"}
        required = sum(len(labels.get(name, name)) + 2 for name in SECTIONS)
        if required <= width:
            position = x
            for name in SECTIONS:
                label = " " + labels.get(name, name) + " "
                self.button(
                    8,
                    position,
                    label,
                    ("section", name),
                    "selected" if self.state.section == name else "muted",
                )
                position += len(label)
        else:
            self.button(8, x, self.state.section + " ▾  [ ]", ("section-menu", None), "accent")
        rows = self.detail_rows(model, width - 1)
        self.detail_fields = rows
        self.field_cursor = min(self.field_cursor, max(0, len(rows) - 1))
        if rows and self.field_cursor == 0 and rows[0][2] is None:
            self.field_cursor = next((i for i, row in enumerate(rows) if row[2] is not None), 0)
        if not rows:
            message = (
                "No fields selected. Press v to choose fields."
                if self.state.custom_fields == []
                else "No fields match. Press / to edit the field search."
            )
            self.put(10, x, message, "muted", width)
        capacity = height - 13
        self.field_scroll = min(self.field_scroll, max(0, len(rows) - capacity))
        if self.focus == "detail":
            self.field_scroll = min(self.field_scroll, self.field_cursor)
            if self.field_cursor >= self.field_scroll + capacity:
                self.field_scroll = self.field_cursor - capacity + 1
        for y, (text, style, key) in enumerate(rows[self.field_scroll : self.field_scroll + capacity], 9):
            index = y - 9 + self.field_scroll
            if style == "layers":
                for col, layer in enumerate(text):
                    kind = mixer_kind(layer)
                    self.put(
                        y,
                        x + col,
                        layer_glyph(layer),
                        {"global": "accent", "local": "green", "hybrid": "gold", "linear": "purple"}[kind],
                    )
            else:
                self.put(
                    y,
                    x,
                    text,
                    "selected" if index == self.field_cursor and self.focus == "detail" else style,
                    width,
                )
            if key:
                self.hits.append((y, x, width, ("field", (index, key))))
        if len(rows) > capacity:
            self.put(height - 4, x, "↑↓ / trackpad to scroll", "muted", width)

    def compare_view(self):
        height, width = self.screen.getmaxyx()
        models = self.state.comparison
        if len(models) < 2:
            self.put(9, 3, "Mark two or more models with space in the Models view.", "accent")
            self.put(11, 3, "Their architecture and training fields will line up here.", "muted")
            return
        label_width = min(30, width // 3)
        count = max(1, (width - label_width - 4) // 28)
        self.compare_col = min(self.compare_col, len(models) - 1)
        self.compare_start = min(self.compare_start, self.compare_col)
        if self.compare_col >= self.compare_start + count:
            self.compare_start = self.compare_col - count + 1
        shown = models[self.compare_start : self.compare_start + count]
        column_width = (width - label_width - 4) // len(shown)
        self.button(4, 2, "g " + self.state.section + " ▾", ("section-menu", None), "accent")
        for i, model in enumerate(shown):
            self.put(
                4,
                label_width + 3 + i * column_width,
                model["name"],
                "accent" if self.compare_start + i == self.compare_col else "title",
                column_width - 2,
            )
        self.rule(5, 2, width - 4)
        rows = []
        keys = self.state.field_keys()
        if not self.show_missing and not self.state.field_search:
            keys = [k for k in keys if any(query.cell(m, k)["value"] is not None for m in models)]
        for key in keys:
            values = [
                self.wrap(
                    self.state.value(m, key, annotate=not key.startswith("params.counted_")), column_width - 2
                )
                for m in shown
            ]
            labels = self.wrap(self.state.db["fields"][key]["label"], label_width - 2)
            for n in range(max(len(labels), *map(len, values))):
                rows.append(
                    (key, labels[n] if n < len(labels) else "", [v[n] if n < len(v) else "" for v in values])
                )
            rows.append((None, "", [""] * len(shown)))
        self.compare_rows = rows
        self.field_cursor = min(self.field_cursor, max(0, len(rows) - 1))
        capacity = height - 11
        self.field_scroll = min(self.field_scroll, self.field_cursor)
        if self.field_cursor >= self.field_scroll + capacity:
            self.field_scroll = self.field_cursor - capacity + 1
        for row, (key, label, values) in enumerate(rows[self.field_scroll : self.field_scroll + capacity], 6):
            selected = row - 6 + self.field_scroll == self.field_cursor
            self.put(row, 2, label, "accent" if selected else "muted", label_width - 2)
            for i, value in enumerate(values):
                self.put(
                    row,
                    label_width + 3 + i * column_width,
                    value,
                    "selected" if selected and self.compare_start + i == self.compare_col else "normal",
                    column_width - 2,
                )
                if key:
                    self.hits.append(
                        (
                            row,
                            label_width + 3 + i * column_width,
                            column_width - 2,
                            ("compare-field", (row - 6 + self.field_scroll, self.compare_start + i, key)),
                        )
                    )
        self.put(
            height - 4,
            2,
            f"Models {self.compare_start + 1}–{self.compare_start + len(shown)} of {len(models)} · ←→ moves between models · Enter shows source or calculation",
            "muted",
        )

    def plot_view(self):
        height, width = self.screen.getmaxyx()
        x0, y0, plot_width, plot_height = 14, 6, width - 20, height - 16
        self.put(4, 2, self.state.db["fields"][self.state.y]["label"], "accent")
        points, bounds = self.make_plot(plot_width, plot_height)
        self.plotted = [p["model"]["id"] for p in points]
        if not points:
            self.put(10, 3, "No models have usable values for these axes.", "gold")
            self.put(12, 3, "x / y changes axes; L toggles logarithmic scales.", "muted")
            return
        xmin, xmax, ymin, ymax = bounds
        for i in range(5):
            row = y0 + round(i * (plot_height - 1) / 4)
            value = ymax - i * (ymax - ymin) / 4
            value = 10**value if self.state.log else value
            self.put(row, 1, display(value, self.state.db["fields"][self.state.y]), "muted", 11)
            self.put(row, x0 - 1, "┼" + "┄" * plot_width, "muted", plot_width + 1)
        for i in range(5):
            col = x0 + round(i * (plot_width - 1) / 4)
            value = xmin + i * (xmax - xmin) / 4
            value = 10**value if self.state.log else value
            self.put(
                y0 + plot_height,
                min(width - 13, col),
                display(value, self.state.db["fields"][self.state.x]),
                "muted",
                13,
            )
        focused = self.state.model["id"] if self.state.model else None
        points.sort(key=lambda p: p["model"]["id"] == focused)
        labels = []
        for point in points:
            model = point["model"]
            active = model["id"] == focused
            marked = model["id"] in self.state.selected
            row, col = y0 + point["row"], x0 + point["col"]
            self.put(
                row,
                col,
                "◆" if active else "◎" if marked else "●",
                "gold" if active else "green" if marked else "accent",
            )
            self.hits.append((row, col, 1, ("point", model["id"])))
            if active or marked or self.labels:
                labels.append((active, row, col, model["name"]))
        occupied = {(y0 + p["row"], x0 + p["col"]) for p in points}
        for active, row, col, name in sorted(labels, reverse=True):
            name = name[:28]
            positions = [(row + dy, col + 2) for dy in (0, -1, 1)]
            positions += [(row + dy, col - len(name) - 1) for dy in (0, -1, 1)]
            for label_row, label_col in positions:
                cells = {(label_row, label_col + i) for i in range(len(name))}
                if (
                    y0 <= label_row < y0 + plot_height
                    and x0 <= label_col
                    and label_col + len(name) < width - 2
                    and not occupied.intersection(cells)
                ):
                    self.put(label_row, label_col, name, "gold" if active else "muted", len(name))
                    occupied.update(cells)
                    break
        self.put(
            y0 + plot_height + 2,
            x0,
            self.state.db["fields"][self.state.x]["label"]
            + (" · log scales" if self.state.log else " · linear scales"),
            "accent",
        )
        model = self.state.model
        if model:
            self.put(height - 6, 2, model["name"], "title")
            self.put(
                height - 5,
                2,
                f"{self.state.x}: {self.state.value(model, self.state.x)}    {self.state.y}: {self.state.value(model, self.state.y)}",
                "normal",
            )
        self.put(
            height - 4,
            2,
            f"{len(points)} plotted · {len(self.state.models) - len(points)} missing or outside log domain · ◆ focused · ◎ marked · Enter opens model",
            "muted",
        )

    def make_plot(self, width, height):
        from .explorer import plot_points

        return plot_points(self.state.models, self.state.x, self.state.y, width, height, self.state.log)

    def prompt(self, label, initial="", live=None):
        value = initial
        cursor = len(value)
        while True:
            self.draw()
            height, width = self.screen.getmaxyx()
            self.put(height - 2, 1, " " * (width - 2), "selected")
            self.put(height - 2, 2, label + ": " + value[:cursor] + "▏" + value[cursor:], "selected")
            self.screen.refresh()
            key = read_key(self.screen)
            if key in ("\n", "\r", curses.KEY_ENTER):
                return value
            if key == "\x1b":
                if live:
                    live(initial)
                return None
            if key in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                value = value[: max(0, cursor - 1)] + value[cursor:]
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_LEFT:
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_RIGHT:
                cursor = min(len(value), cursor + 1)
            elif key == curses.KEY_HOME:
                cursor = 0
            elif key == curses.KEY_END:
                cursor = len(value)
            elif key == curses.KEY_DC:
                value = value[:cursor] + value[cursor + 1 :]
            elif key == "\x15":
                value = ""
                cursor = 0
            elif isinstance(key, str) and key.isprintable():
                value = value[:cursor] + key + value[cursor:]
                cursor += len(key)
            if live:
                live(value)

    def choose(self, title, options, initial=None):
        text = ""
        cursor = next((i for i, option in enumerate(options) if option[0] == initial), 0)
        while True:
            self.draw()
            height, width = self.screen.getmaxyx()
            filtered = [
                (key, label)
                for key, label in options
                if all(
                    term
                    in (
                        key
                        + " "
                        + label
                        + " "
                        + " ".join(self.state.db["fields"].get(key, {}).get("aliases", []))
                    ).casefold()
                    for term in text.casefold().split()
                )
            ]
            cursor = min(cursor, max(0, len(filtered) - 1))
            left, top, popup_width = max(2, (width - 88) // 2), 6, min(88, width - 4)
            for row in range(top, height - 4):
                self.put(row, left, " " * popup_width, "panel")
            self.put(top, left + 2, title + "  / " + text + "▏", "panel-title", popup_width - 4)
            count = height - top - 7
            start = max(0, cursor - count + 1)
            for row, (key, label) in enumerate(filtered[start : start + count], top + 2):
                active = row - top - 2 + start == cursor
                if active:
                    self.put(row, left, " " * popup_width, "selected")
                self.put(
                    row,
                    left + 2,
                    f"{'›' if active else ' '} {key}  {label}",
                    "selected" if active else "panel",
                    popup_width - 4,
                )
            self.put(
                height - 5,
                left + 2,
                "Type to search · arrows move · Enter choose · Esc cancel",
                "panel-muted",
                popup_width - 4,
            )
            self.screen.refresh()
            key = read_key(self.screen)
            if key == "\x1b":
                return None
            if key in ("\n", "\r", curses.KEY_ENTER) and filtered:
                return filtered[cursor][0]
            if key == curses.KEY_UP:
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_DOWN:
                cursor = min(len(filtered) - 1, cursor + 1)
            elif key in (curses.KEY_LEFT, curses.KEY_RIGHT) and filtered:
                cursor = (cursor + (-1 if key == curses.KEY_LEFT else 1)) % len(filtered)
            elif key in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                text = text[:-1]
                cursor = 0
            elif isinstance(key, str) and key.isprintable():
                text += key
                cursor = 0

    def field_picker(self):
        chosen, expanded, text, cursor = set(self.state.field_keys()), set(), "", 0
        while True:
            self.draw()
            height, width = self.screen.getmaxyx()
            groups, fields = self.state.db["groups"], self.state.db["fields"]
            entries = []
            for group, label in groups.items():
                keys = [
                    k
                    for k, spec in fields.items()
                    if k.startswith(group + ".")
                    and all(t in f"{k} {spec}".casefold() for t in text.casefold().split())
                ]
                if not keys:
                    continue
                tick = (
                    "x" if all(k in chosen for k in keys) else "−" if any(k in chosen for k in keys) else " "
                )
                entries.append((f"{'▾' if group in expanded or text else '▸'} [{tick}] {label}", keys, group))
                if group in expanded or text:
                    entries.extend(
                        (f"    [{'x' if k in chosen else ' '}] {fields[k]['label']}  {k}", [k], group)
                        for k in keys
                    )
            cursor = min(cursor, max(0, len(entries) - 1))
            left, popup_width = 3, width - 6
            for row in range(6, height - 4):
                self.put(row, left, " " * popup_width, "panel")
            self.put(
                6,
                left + 2,
                f"Fields · {len(chosen)} selected" + (f" · search: {text}" if text else ""),
                "panel-title",
            )
            count = height - 16
            start = max(0, cursor - count + 1)
            for row, (label, _, _) in enumerate(entries[start : start + count], 8):
                active = row - 8 + start == cursor
                if active:
                    self.put(row, left, " " * popup_width, "selected")
                self.put(row, left + 2, label, "selected" if active else "panel", popup_width - 4)
            self.put(
                height - 6,
                left + 2,
                "↑↓ move · → expand · ← collapse · space toggle · a all · n none · / search",
                "panel-muted",
                popup_width - 4,
            )
            self.put(height - 5, left + 2, "Enter applies · Esc cancels", "panel-muted")
            self.screen.refresh()
            key = read_key(self.screen)
            if key == "\x1b":
                return
            if key in ("\n", "\r", curses.KEY_ENTER):
                self.state.custom_fields = [k for k in fields if k in chosen]
                self.field_cursor = self.field_scroll = 0
                return
            if key in (curses.KEY_UP, "k"):
                cursor = max(0, cursor - 1)
            elif key in (curses.KEY_DOWN, "j"):
                cursor = min(len(entries) - 1, cursor + 1)
            elif key in (curses.KEY_RIGHT, "l") and entries:
                expanded.add(entries[cursor][2])
            elif key in (curses.KEY_LEFT, "h") and entries:
                expanded.discard(entries[cursor][2])
            elif key == " " and entries:
                keys = entries[cursor][1]
                chosen.difference_update(keys) if all(k in chosen for k in keys) else chosen.update(keys)
            elif key == "a":
                chosen = set(fields)
            elif key == "n":
                chosen.clear()
            elif key == "/":
                text = self.prompt("Find fields", text) or ""
                cursor = 0

    def document(self, title, lines, url=None):
        scroll = 0
        while True:
            self.screen.erase()
            height, width = self.screen.getmaxyx()
            self.put(1, 3, title, "title")
            self.rule(3, 2, width - 4)
            wrapped = [part for line in lines for part in self.wrap(line, width - 8)]
            scroll = max(0, min(scroll, len(wrapped) - (height - 8)))
            for row, line in enumerate(wrapped[scroll : scroll + height - 8], 5):
                self.put(row, 4, line, "normal")
            self.put(
                height - 2,
                3,
                "↑↓ scroll · o opens source · Esc closes" if url else "↑↓ scroll · Esc closes",
                "muted",
            )
            self.screen.refresh()
            key = read_key(self.screen)
            if key in ("\x1b", "q", "\n", "\r"):
                return
            if key in (curses.KEY_DOWN, "j", curses.KEY_NPAGE):
                scroll += (height - 9) if key == curses.KEY_NPAGE else 1
            elif key in (curses.KEY_UP, "k", curses.KEY_PPAGE):
                scroll -= (height - 9) if key == curses.KEY_PPAGE else 1
            elif key == "o" and url:
                self.open_url(url)

    def evidence(self, model, key):
        cell = query.cell(model, key)
        spec = self.state.db["fields"][key]
        value = (
            json.dumps(cell["value"], ensure_ascii=False, indent=2)
            if isinstance(cell["value"], list)
            else self.state.value(model, key)
        )
        lines = [value, "", spec["definition"], "", "Status: " + cell["status"]]
        if spec.get("unit"):
            lines.append("Unit: " + spec["unit"])
        url = cell.get("source", {}).get("url")
        if url:
            lines += ["", "Source: " + url, "Location: " + cell.get("at", "")]
        for label, item in (
            ("Note", "note"),
            ("Calculation", "formula"),
            ("Formula version", "formula_version"),
        ):
            if cell.get(item):
                lines += ["", label + ": " + str(cell[item])]
        if cell.get("sequence"):
            lines += ["", "Ordered layers: " + " → ".join(cell["sequence"])]
        if cell.get("quote"):
            lines += ["", "Source excerpt:", cell["quote"]]
        self.document(model["name"] + " · " + spec["label"], lines, url)

    def open_url(self, url):
        if safe_url(url):
            webbrowser.open(url)
            self.notice = "Opened " + url
        else:
            self.notice = "Only HTTPS links can be opened."

    def inspect(self, model, key):
        if key.startswith("@group."):
            group = key.split(".", 1)[1]
            target = self.search_collapsed if self.state.field_search else self.open_groups
            target.symmetric_difference_update({group})
        else:
            self.evidence(model, key)

    def choose_section(self):
        selected = self.choose("Field views", [(name, "") for name in SECTIONS], initial=self.state.section)
        if selected:
            self.reset_fields(selected)

    def reset_fields(self, section):
        self.state.section = section
        self.state.custom_fields = None
        self.field_cursor = self.field_scroll = 0

    def show_all_fields(self):
        self.state.view = "Models"
        self.reset_fields("All fields")
        self.state.field_search = ""
        self.open_groups = set(self.state.db["groups"])
        self.search_collapsed.clear()
        self.show_missing = True
        self.focus = "detail"
        self.notice = (
            "All fields for " + self.state.model["name"] if self.state.model else "No model selected"
        )

    def field_options(self):
        fields = self.state.db["fields"]
        priority = [
            "params.total",
            "params.active",
            "train.batch_tokens",
            "train.tokens",
            "shape.layers",
            "shape.d_model",
            "moe.expert_dim",
            "moe.sparsity",
            "meta.released",
        ]
        keys = dict.fromkeys([*priority, *fields])
        return [
            (k, fields[k]["label"] + (" · text" if fields[k]["type"] == "str" else ""))
            for k in keys
            if k in fields
        ] + list(query.ATTRIBUTES.items())

    def scroll(self, delta, mouse_x):
        if self.state.view == "Models" and mouse_x < self.split:
            self.move(delta, mouse_x)
            return
        if self.state.view == "Plot":
            self.move(delta)
            return
        height, _ = self.screen.getmaxyx()
        rows = (
            getattr(self, "detail_fields", [])
            if self.state.view == "Models"
            else getattr(self, "compare_rows", [])
        )
        capacity = height - (13 if self.state.view == "Models" else 11)
        self.field_scroll = max(0, min(max(0, len(rows) - capacity), self.field_scroll + delta))
        self.field_cursor = max(self.field_scroll, min(self.field_cursor, self.field_scroll + capacity - 1))
        self.focus = "detail"

    def mouse(self, event):
        x, y = event.x, event.y
        if event.action in {"up", "down"}:
            self.scroll(-3 if event.action == "up" else 3, x)
        elif event.action in {"left", "right"}:
            self.horizontal(-1 if event.action == "left" else 1)
        elif event.action == "click":
            hit = next(
                (
                    action
                    for row, col, width, action in reversed(self.hits)
                    if row == y and col <= x < col + width
                ),
                None,
            )
            if not hit:
                return
            action, value = hit
            if action == "view":
                self.state.view = value
                self.field_cursor = self.field_scroll = 0
            elif action == "section":
                self.reset_fields(value)
            elif action == "section-menu":
                self.choose_section()
            elif action == "missing":
                self.key("u")
            elif action == "help":
                self.key("?")
            elif action == "sort":
                self.key("s")
            elif action == "filter":
                self.key("f")
            elif action == "show-all":
                self.show_all_fields()
            elif action == "url":
                self.open_url(value)
            elif action == "search":
                self.edit_search()
            elif action == "field-search":
                self.edit_field_search()
            elif action == "model":
                self.state.cursor = value
                self.focus = "models"
                self.field_cursor = self.field_scroll = 0
            elif action == "model-metric":
                self.state.cursor, metric = value
                self.evidence(self.state.model, metric)
            elif action == "point":
                self.state.cursor = next(
                    i for i, model in enumerate(self.state.models) if model["id"] == value
                )
            elif action == "field":
                self.focus = "detail"
                self.field_cursor, key = value
                self.inspect(self.state.model, key)
            elif action == "compare-field":
                self.field_cursor, self.compare_col, key = value
                self.evidence(self.state.comparison[self.compare_col], key)

    def move(self, delta, mouse_x=None):
        if self.state.view == "Models" and (
            mouse_x is not None and mouse_x < self.split or mouse_x is None and self.focus == "models"
        ):
            self.state.move(delta)
            self.field_cursor = self.field_scroll = 0
        elif self.state.view == "Plot" and self.plotted:
            current = self.state.model["id"] if self.state.model else None
            index = self.plotted.index(current) if current in self.plotted else 0
            target = self.plotted[max(0, min(len(self.plotted) - 1, index + delta))]
            self.state.cursor = next(i for i, model in enumerate(self.state.models) if model["id"] == target)
        else:
            self.field_cursor = max(0, self.field_cursor + delta)

    def edit_search(self):
        def update(text):
            self.state.search = text
            self.state.refresh()
            self.field_cursor = self.field_scroll = 0

        self.prompt("Search models", self.state.search, update)

    def edit_field_search(self):
        if self.state.custom_fields is None:
            self.state.section = "All fields"

        def update(text):
            self.state.field_search = text
            self.search_collapsed.clear()
            self.focus = "detail"
            self.field_cursor = self.field_scroll = 0

        self.prompt("Find fields", self.state.field_search, update)

    def horizontal(self, delta):
        if self.state.view == "Compare":
            self.compare_col = max(0, min(max(0, len(self.state.comparison) - 1), self.compare_col + delta))
        elif self.state.view == "Models":
            self.focus = "models" if delta < 0 else "detail"
        else:
            self.move(delta)

    def add_filter(self):
        fields = self.state.db["fields"]
        key = self.choose("Filter field", self.field_options())
        if key is None:
            return
        spec = fields.get(key, {"type": "str"})
        options = [("=", "equals"), ("!=", "does not equal")]
        if spec["type"] in {"int", "float", "date"}:
            options += [
                (op, text)
                for op, text in (
                    (">", "greater than"),
                    (">=", "at least"),
                    ("<", "less than"),
                    ("<=", "at most"),
                )
            ]
        elif spec["type"] in {"str", "list"}:
            options += [("contains", "contains text"), ("not contains", "does not contain text")]
        options.append(("is", "known, unknown, not disclosed, or evidence status"))
        op = self.choose("Filter " + key, options)
        if op is None:
            return
        if op == "is":
            value = self.choose("Value status", [(s, "") for s in sorted(query.STATUSES)])
        elif spec["type"] == "bool":
            value = self.choose("Value", [("true", "yes"), ("false", "no")])
        else:
            value = self.prompt(f"{key} {op}")
        if value:
            self.state.set_filter(f"{key} {op} {value}")

    def enter(self):
        if self.state.view == "Plot":
            self.return_view = "Plot"
            self.state.view = "Models"
            self.focus = "detail"
            self.field_cursor = self.field_scroll = 0
        elif self.state.view == "Models":
            if self.focus == "models":
                self.show_all_fields()
            elif self.state.model and getattr(self, "detail_fields", []):
                key = self.detail_fields[self.field_cursor][2]
                if key:
                    self.inspect(self.state.model, key)
        elif self.state.comparison and getattr(self, "compare_rows", []):
            key = self.compare_rows[self.field_cursor][0]
            if key:
                self.evidence(self.state.comparison[self.compare_col], key)

    def key(self, key):
        self.notice = ""
        if isinstance(key, Mouse):
            self.mouse(key)
            return True
        if key in ("q", "\x03"):
            return False
        if key in ("1", "2", "3"):
            self.state.view = ("Models", "Compare", "Plot")[int(key) - 1]
            self.field_cursor = self.field_scroll = 0
        elif key in (curses.KEY_DOWN, "j", curses.KEY_NPAGE):
            self.move(8 if key == curses.KEY_NPAGE else 1)
        elif key in (curses.KEY_UP, "k", curses.KEY_PPAGE):
            self.move(-8 if key == curses.KEY_PPAGE else -1)
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT):
            self.horizontal(-1 if key == curses.KEY_LEFT else 1)
        elif key in ("\t", curses.KEY_BTAB):
            self.focus = "detail" if self.focus == "models" else "models"
        elif key == "\x1b":
            self.focus = "models"
            if self.return_view:
                self.state.view, self.return_view = self.return_view, None
        elif key in ("\n", "\r", curses.KEY_ENTER):
            self.enter()
        elif key == " ":
            if self.state.view == "Compare" and self.state.comparison:
                self.state.selected.remove(self.state.comparison[self.compare_col]["id"])
                self.compare_col = max(0, self.compare_col - 1)
            else:
                self.state.toggle()
        elif key == "/":
            self.edit_search() if self.state.view == "Plot" else self.edit_field_search()
        elif key == "\x06":
            self.edit_field_search()
        elif key == "m":
            self.edit_search()
        elif key == "f":
            self.add_filter()
        elif key == ":":
            expression = self.prompt("Filter, e.g. moe.sparsity >= 32")
            if expression:
                self.state.set_filter(expression)
        elif key == "F":
            self.state.filters.clear()
            self.state.refresh()
        elif key in (curses.KEY_BACKSPACE, "\x7f", "\b") and self.state.filters:
            self.state.filters.pop()
            self.state.refresh()
        elif key == "s":
            selected = self.choose(
                "Sort by field",
                self.field_options(),
            )
            if selected:
                self.state.set_sort(selected)
                self.field_cursor = self.field_scroll = 0
                self.notice = (
                    "Sorted by "
                    + self.state.db["fields"].get(selected, {}).get("label", selected)
                    + (" · largest/newest first" if self.state.descending else " · ascending")
                )
                known = sum(query.cell(model, selected)["value"] is not None for model in self.state.models)
                self.notice += f" · {known}/{len(self.state.models)} known"
        elif key == "d":
            self.state.set_sort(self.state.sort, not self.state.descending)
            self.field_cursor = self.field_scroll = 0
        elif key in ("[", "]"):
            sections = list(SECTIONS)
            self.reset_fields(
                sections[(sections.index(self.state.section) + (1 if key == "]" else -1)) % len(sections)]
            )
        elif key == "g":
            self.choose_section()
        elif key == "u":
            self.show_missing = not self.show_missing
            self.field_cursor = self.field_scroll = 0
            self.notice = (
                "Showing missing values"
                if self.show_missing
                else "Hiding missing values; field search still finds them"
            )
        elif key == "a" and self.state.view == "Models":
            self.show_all_fields()
        elif key == "A" and self.state.view == "Models":
            self.open_groups.clear()
            self.search_collapsed = set(self.state.db["groups"]) if self.state.field_search else set()
            self.field_cursor = self.field_scroll = 0
        elif key == "v":
            self.field_picker()
        elif key in ("h", "o", "c", "r") and self.state.model:
            model = (
                self.state.comparison[self.compare_col]
                if self.state.view == "Compare" and self.state.comparison
                else self.state.model
            )
            target = {"h": "Hugging Face", "o": "Hugging Face", "c": "Config", "r": "Report"}[key]
            url = links(model).get(target)
            if url:
                self.open_url(url)
            else:
                self.notice = f"No {target} link for this model."
        elif key == "C":
            value = self.prompt("KV context in tokens", str(self.state.db["_kv"]["context"]))
            if value:
                self.state.set_kv(query.positive_integer(value), self.state.db["_kv"]["precision"])
        elif key == "t":
            precisions = ["bf16", "fp8", "fp4", "sourced"]
            self.state.set_kv(
                self.state.db["_kv"]["context"],
                precisions[(precisions.index(self.state.db["_kv"]["precision"]) + 1) % len(precisions)],
            )
        elif key in ("x", "y") and self.state.view == "Plot":
            selected = self.choose(
                "Plot " + key.upper(),
                [
                    (k, v["label"])
                    for k, v in self.state.db["fields"].items()
                    if v["type"] in {"int", "float"}
                ],
            )
            if selected:
                setattr(self.state, key, selected)
        elif key == "p" and self.state.view == "Plot":
            self.state.plot_preset = (self.state.plot_preset + 1) % len(PLOTS)
            label, self.state.x, self.state.y, self.state.log = PLOTS[self.state.plot_preset]
            self.notice = label
        elif key == "L":
            self.state.log = not self.state.log
        elif key == "n":
            self.labels = not self.labels
        elif key == "e":
            target = self.prompt("Export JSON to new file", "archdb-selection.json")
            if target:
                from .browse import rows

                with Path(target).expanduser().open("x", encoding="utf-8") as out:
                    json.dump(
                        rows(
                            self.state.db,
                            self.state.comparison or self.state.models,
                            self.state.field_keys(),
                            True,
                        ),
                        out,
                        ensure_ascii=False,
                        indent=2,
                    )
                    out.write("\n")
                self.notice = "Saved " + target
        elif key == "?":
            self.document(
                "archdb · Keys",
                [
                    "1 Models · 2 Compare · 3 Plot · q quit",
                    "",
                    "←→ switches panes; Tab also works. ↑↓ or j/k moves through the focused pane. Enter on a model shows all its fields; Enter on a field opens its evidence.",
                    "",
                    "m searches models; / searches fields. a shows all fields and clears field selection and search. A collapses groups; Enter expands one. u shows missing values. g chooses a view; [ and ] cycle views.",
                    "",
                    "f builds a filter; : accepts an expression, such as moe.sparsity >= 32. Every filter must match. Backspace removes the last; F clears filters. Use 'is unknown' to filter for missing data. s chooses a sort field; d reverses the order. Missing values stay last.",
                    "",
                    "Space marks a model for Compare. In Compare, ←→ moves between models, ↑↓ scrolls fields, and space removes a model.",
                    "",
                    "v selects fields: → expands a group, ← collapses it, space toggles, a selects all, n clears all, / searches, Enter applies.",
                    "",
                    "h opens Hugging Face, c the pinned config, r the report. e exports marked models (or the filtered list) to a new JSON file.",
                    "",
                    "C sets KV context (K=1000); t cycles uniform BF16 / FP8 / FP4 payloads and sourced storage.",
                    "",
                    "In Plot: x/y chooses axes, p cycles presets, L toggles log scales, n toggles names. Enter opens the focused model; Esc returns. Missing values are omitted.",
                ],
            )
        return True

    def run(self, screen):
        self.screen = screen
        screen.keypad(True)
        curses.set_escdelay(25)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            for index, (name, foreground, background) in enumerate(
                [
                    ("normal", -1, -1),
                    ("muted", 244 if curses.COLORS >= 256 else 7, -1),
                    ("accent", 81 if curses.COLORS >= 256 else 6, -1),
                    ("green", 114 if curses.COLORS >= 256 else 2, -1),
                    ("purple", 177 if curses.COLORS >= 256 else 5, -1),
                    ("gold", 221 if curses.COLORS >= 256 else 3, -1),
                    ("selected", 255 if curses.COLORS >= 256 else 7, 237 if curses.COLORS >= 256 else 4),
                    ("panel", 255 if curses.COLORS >= 256 else 7, 235 if curses.COLORS >= 256 else 4),
                    ("panel-muted", 250 if curses.COLORS >= 256 else 7, 235 if curses.COLORS >= 256 else 4),
                ],
                1,
            ):
                curses.init_pair(index, foreground, background)
                self.colors[name] = curses.color_pair(index)
            self.colors["title"] = self.colors["normal"] | curses.A_BOLD
            self.colors["panel-title"] = self.colors["panel"] | curses.A_BOLD
            # A nonzero background pair overrides per-cell colors on macOS ncurses.
            screen.bkgd(" ", 0)
        else:
            self.colors = {"selected": curses.A_REVERSE, "title": curses.A_BOLD, "accent": curses.A_BOLD}
        curses.mousemask(curses.ALL_MOUSE_EVENTS)
        curses.mouseinterval(0)
        self.state.refresh(False)
        # Explicit SGR reporting covers trackpads and downward scrolling on the
        # older ncurses shipped with macOS, which has no BUTTON5 constant.
        sys.stdout.write("\x1b[?1000h\x1b[?1006h")
        sys.stdout.flush()
        try:
            while True:
                self.draw()
                try:
                    if not self.key(read_key(screen)):
                        return
                except (ValueError, OSError) as exc:
                    self.notice = clean(exc)
        finally:
            sys.stdout.write("\x1b[?1006l\x1b[?1000l")
            sys.stdout.flush()


def explore(db: dict, args) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError(
            "explore needs an interactive terminal; use `archdb models --format json` in scripts"
        )
    locale.setlocale(locale.LC_ALL, "")
    state = Explorer(db, search=args.search, filters=args.where, sort=args.sort, descending=args.desc)
    if args.fields or args.group:
        from .browse import selected_fields

        state.custom_fields = selected_fields(db, args)
        if any(k not in db["fields"] for k in state.custom_fields):
            raise ValueError("explore --fields accepts architecture fields; model identity is already shown")
    state.refresh(False)
    try:
        curses.wrapper(Application(state).run)
    except curses.error as exc:
        raise ValueError(
            f"terminal initialization failed: {exc}; try `archdb models` for plain output"
        ) from exc
