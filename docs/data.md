# Using the data

Everything is in [`exports/`](../exports), regenerated from the records by `uv run archdb build`:

| File | Contents |
|---|---|
| [`models.csv`](../exports/models.csv) | One row per model with org, country, checkpoint kind and weights availability, one column per field (raw units: parameters, tokens, bytes). |
| [`facts.csv`](../exports/facts.csv) | One row per value with its source URL, location, quote and basis (`stated`, `inferred`, `derived`, `not disclosed`). Use this to check a number. |
| [`stages.csv`](../exports/stages.csv) | Pretraining stages: tokens, sequence length, learning rate, batch, steps. |
| [`fields.csv`](../exports/fields.csv) | Field definitions and units. |
| [`archdb.json`](../exports/archdb.json) | All of the above in one file. |

Field keys are `group.name`, for example `attn.kv_heads`, `train.lr_peak`, `moe.top_k`. Definitions
are in [`data/fields.yaml`](../data/fields.yaml). Fields marked `derived` are computed from other
values: ratios (`shape.aspect`, `attn.swa_ratio`, `moe.latent_ratio`, `moe.sparsity`), tokens per
parameter, the AdamW timescale τ = B/(ηλD) and B/D^0.47 from Power Lines (arXiv 2505.13738), and KV
cache sizes. KV sizes assume BF16 (the website also shows FP8) and count global layers, sliding
windows capped at their window, and sparse-attention indexer keys. `kv.bytes_per_token` is the total
cache at a context divided by that context, so sliding-window caches are included;
`kv.growth_per_token` is the long-context limit (global layers only). Context-dependent values in the
exports are evaluated at 131,072 tokens.

Parameter counts by part (`params.counted_*`) are read from the released checkpoints' safetensors
headers (tensor names and shapes, fetched with range requests) by `uv run archdb count-params` and
stored in [`data/param_counts.yaml`](../data/param_counts.yaml). A count is shown only when it reproduces
the publisher's stated total and active parameters under some scope (with or without embeddings,
output head, MTP modules, vision towers and n-gram tables). The non-embedding active count excludes the
input embedding, the output head and n-gram/per-layer lookup tables.

Benchmark scores (`eval.*`) are independent measurements of the released post-trained models, kept in
[`data/evals.yaml`](../data/evals.yaml) apart from the architecture records.

```python
import pandas as pd

m = pd.read_csv("exports/models.csv")
m[["name", "params.active", "attn.global_layers", "kv.bytes_per_token"]].sort_values("kv.bytes_per_token")

f = pd.read_csv("exports/facts.csv")
f[(f.model == "deepseek-v3") & f.field.str.startswith("train.")][["field", "value", "quote"]]
```

`basis` in `facts.csv` is `stated` (pointer or quote), `inferred` (simple derivation by the
curator, explained in `note`), or `derived` (computed by `archdb/derive.py`).

## Scope

Recent models from major labs with public weights and a config, plus a few closed-weight models whose
technical report discloses architecture and recipe in detail (`weights: closed`). One record per
pretraining run: quantized, distilled and post-training-only variants are not separate records. When
only a post-trained checkpoint is public, the record says so in its `note`.

Things the database does not do: estimate undisclosed values, infer training precision from
checkpoint dtypes, or treat config fields the modeling code ignores as evidence.
