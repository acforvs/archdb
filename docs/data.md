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
parameter, the peak-LR AdamW timescale estimate τ = B/(ηλD), and KV cache sizes. For τ, normalization
by the full run follows [Power Lines](https://arxiv.org/html/2505.13738v2#S2.SS2); replacing total steps
S with D/B assumes constant batch size.
Exported KV sizes use uniform BF16 at 131,072 tokens. They count global caches, bounded sliding
windows and sparse-attention indexer keys. `kv.bytes_per_token` includes the windows;
`kv.growth_per_token` is global KV only. The website also offers uniform FP8/FP4 payload comparisons
and **Sourced storage**, which uses component formats, scales and documented padding. A sourced
value is left unknown until the entire cache recipe is covered. See [KV calculations](kv-cache.md)
for formulas, assumptions and checks against the DeepSeek reports.

Parameter counts by part (`params.counted_*`) are read from the released checkpoints' safetensors
headers (tensor names and shapes, fetched with range requests) by `uv run archdb count-params` and
stored in [`data/param_counts.yaml`](../data/param_counts.yaml). A count is shown only when it reproduces
the publisher's stated total and active parameters under some scope (with or without embeddings,
output head, MTP modules, vision towers and n-gram tables). The non-embedding active count excludes the
input embedding, the output head and n-gram/per-layer lookup tables.
`params.counted_non_embedding` gives the total backbone count with those same exclusions.
`params.counted_max_layer` counts the largest group of weights sharing a full `layers.<index>`
tensor path, including all routed experts. Hybrid checkpoints can store mixers and FFNs in separate
layers; the number is not an estimate of all-gather time.

`train.accelerators` and `train.parallelism` keep short hardware and parallelism descriptions apart
from the longer `train.hardware` notes. When a value applies to a particular training stage, its
evidence note identifies that stage. `meta.modality` lists the checkpoint's input modalities.

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
