# Terminal usage

Install with [uv](https://docs.astral.sh/uv/):

```sh
uv tool install git+https://github.com/acforvs/archdb
archdb
```

The explorer supports macOS and Linux with Python 3.11+ and curses. After installation, `archdb`
works from any directory. The dataset is bundled; Node.js and a running website are not needed.
If the command isn't found, run `uv tool update-shell` and reopen your terminal.

Update the tool and bundled data:

```sh
uv tool upgrade archdb
```

Remove it with `uv tool uninstall archdb`. From a checkout, use `uv tool install .` to install a
snapshot, `uv tool install --editable .` for development, or `uv run archdb` without installing a
tool. Browsing itself uses only the Python standard library; the package also installs dependencies
used by the database authoring commands.

## Explorer

The opening view shows All fields next to the model list. Parameters and Shape start expanded;
other groups show a short summary. Press Enter on a group to expand it, or `/` and type a field name, such as
`KV heads` or `expert`, to narrow the fields as you type. Press `m` to find a model. The two searches
are independent, so a field search stays in place when you switch models. Matching groups expand
automatically. Empty fields are hidden until you search for them or turn on `u Show missing`.

Use `←→` to move between the model and field panes, and `g` to choose a section by name. Overview keeps the common architecture
lookups together. Hugging Face, pinned config and report links are separate. Evidence opens on request.

| Key | Action |
|---|---|
| `/` or `Ctrl-f` | Search the displayed fields as you type. In Plot, `/` searches models. |
| `m` | Search model names and organizations as you type. |
| `↑↓` or `j/k` | Scroll fields, or move between models when the model pane has focus. |
| Enter on a model / `a` | Show all fields for that model, including missing values; clear the field search and field selection. |
| Enter on a group | Expand or collapse it. `A` collapses all groups. |
| Enter on a field | Open its value, definition and source evidence. |
| `Tab` | An alternative to arrows for switching panes. |
| `u` | Include or hide unavailable values. A field search always includes matching unavailable fields. |
| `1`, `2`, `3` | Models, Compare, Plot. |
| Space | Mark or unmark a model. In Compare, remove the focused model. |
| `←→` | Focus the left/right pane in Models; move between marked models in Compare. Option arrows work too. |
| `g` | Choose a field section by name. `[` / `]` also change sections. |
| `v` | Choose fields. Expand groups with `→`, toggle with space, select all with `a`, clear with `n`, search with `/`. `Enter` applies. |
| `f` | Build a filter with searchable fields and operators. |
| `:` | Enter a filter expression directly. |
| Backspace / `F` | Remove the last filter / clear all filters. |
| `s` / `d` | Choose a sort field / reverse its order. |
| `x` / `y` in Plot | Choose axes. |
| `p` / `L` / `n` in Plot | Cycle plots / toggle log scales / toggle point labels. |
| `C` / `t` | Set KV context / cycle storage recipes. |
| `h` / `c` / `r` | Open Hugging Face / the pinned config / the technical report. `o` is also an alias for Hugging Face. |
| `e` | Export marked models, or the filtered list, to a new JSON file. |
| `Esc` | Close a dialog or leave field navigation. After inspecting a plot point, return to the plot. |
| `?` / `q` | Keys / quit. |

The mouse selects models, opens field evidence and follows the labeled links. A trackpad or mouse
wheel scrolls the pane under it, including All fields. Numeric sorts start largest first and dates
start newest first; changing sort moves to the top and shows the sorted value in the model list.
For example, press `s`, type `batch size`, and press Enter to rank models by their numeric global
batch size in tokens. Batch schedule notes are a separate text field. Missing values remain last.

Layer strips preserve the sourced order. Inferred sequences are labeled; counts without an order
are shown as counts. Calculated and inferred values are labeled. Plots show model names where they
fit; the focused model's full name and axis values appear below the plot.

## Commands and scripts

```sh
archdb models deepseek
archdb fields expert --group moe
archdb show deepseek-v3 --group moe
archdb compare deepseek-v3 kimi-k2 --fields params.active,moe.experts,moe.expert_dim
archdb evidence deepseek-v3 shape.layers
```

Use `uv run archdb` in place of `archdb` when working from the repository. A model ID or an
unambiguous name is accepted; an ambiguous name lists the matching variants.

Repeat `--where` to require every condition:

```sh
archdb models --where 'params.total >= 400B' --where 'params.total <= 700B' \
  --where 'attn.kind contains SWA' --sort shape.aspect --desc

archdb models --where 'moe.sparsity >= 32' --where 'moe.active_fraction < 10%' \
  --fields params.total,params.active,moe.sparsity --format json

archdb models --where 'mtp.depth is not-disclosed' --format csv
archdb show deepseek-v41 --group kv --precision sourced --context 131072 --evidence
```

Operators are `=`, `!=`, `>`, `>=`, `<`, `<=`, `contains`, `not contains`, and `is`. Text matching is
case-insensitive. `is` accepts `known`, `unknown`, `not-disclosed`, `reported`, `config`, `code`,
`inferred`, `derived`, and `metadata`. Missing values do not match ordinary comparisons, including
`!=`. Zero, false, and the string `None` remain actual values.

Numbers accept decimal `K`, `M`, `B`, `T`, binary `KiB`/`MiB`/`GiB`, percentages, and ratios such as
`16:2`. `1T` sorts above `670B`; `16:1` sorts above `16:2`. Numeric bounds match only conclusive
comparisons: a reported `>35T` matches `>30T`, but cannot establish `>40T`. Approximate values such as
`~12T` sort by their stated number but do not satisfy exact numeric comparisons. Dates accept
`YYYY`, `YYYY-MM` or `YYYY-MM-DD`; omitted month/day values are treated as 1.

`--format json` and `--format csv` keep raw values and include `_status`; `--evidence` adds sources,
locations, quotes and formulas. CSV encodes structured metadata as JSON cells. KV output includes
`_kv`, identifying context and storage. Human output abbreviates large numbers and byte sizes.

KV defaults to 131,072 tokens and uniform BF16 payload. `--precision fp8` and `fp4` are uniform
payload comparisons; `sourced` includes documented formats, scales and padding and stays unknown
when the storage recipe is incomplete. Calculations include bounded SWA caches. See
[KV calculations](kv-cache.md).

Commands read the local `exports/archdb.json`, or the snapshot included in an installed wheel.
`--data path/to/archdb.json` selects another export. Browsing makes no network requests; opening a
source link launches your browser. To update a checkout's snapshot after editing records, run
`uv run archdb build`.
