---
name: add-model
description: Add a new model record to archdb (or bring an existing record up to date) with every value backed by a pinned config pointer or verbatim quote, then audit it. Use when asked to add, update or audit a model in data/models/.
---

# Adding a model to archdb

The reader is a pretraining researcher who will use these values without opening the sources. A
missing value is acceptable; a wrong one is not. Read `CONTRIBUTING.md` and `data/fields.yaml`
before starting, and open one complete record (`data/models/deepseek-v3.yaml`) as a template.

## 1. Decide whether it gets a record

One record per pretraining run. Before writing anything, establish from the publisher's own pages:

- It is a distinct pretraining run, not a post-training refresh, quantization, distillation,
  pruned variant or dated update of a run that already has a record (`grep -h '^name:'
  data/models/*.yaml`). A new post-trained release of a covered base is at most a note.
- It is recent and strong (competitive among open weights on the Artificial Analysis Intelligence
  Index) or has an unusually detailed pretraining report. Closed-weight models qualify only through
  such a report and get `weights: closed`.
- Which checkpoint the record describes. If no base checkpoint is public, the top-level `note` is
  exactly: "No base checkpoint is public; facts describe the released post-trained checkpoint."

## 2. Collect and pin sources

Primary sources only: config.json, modeling code (publisher's or Hugging Face transformers'),
model card, technical report, publisher blog. Pin every URL; `archdb validate` rejects anything else:

| Source | Pinned form |
|---|---|
| Hugging Face file | `https://huggingface.co/<repo>/resolve/<40-char commit>/<file>` (commit from `https://huggingface.co/api/models/<repo>`) |
| GitHub file | `https://raw.githubusercontent.com/<org>/<repo>/<commit>/<path>` |
| arXiv | `https://arxiv.org/pdf/<id>v<N>` |
| Any other web page or PDF | Wayback snapshot `https://web.archive.org/web/<14 digits>/<url>`; use `.../<14 digits>id_/<url>` for PDFs, JS and JSON so the raw file is served |
| Hugging Face API JSON | `kind: api`, e.g. commit history for a release date |

Source kinds: `config`, `code`, `paper`, `model-card`, `blog`, `api`, `leaderboard`.

- Pin the commit whose files describe the checkpoint the record covers. When a repository has
  several commits (re-uploads, renamed checkpoints, edited configs), take the one that matches the
  released weights and say which checkpoint it is in a note if the repository name is ambiguous.
- Every source in `sources` must be cited by at least one fact (or stage item); `validate` rejects
  uncited sources, so do not list sources you only read.
- A publisher's training script or launch config is evidence for the recipe only when it is the
  production configuration (its steps, tokens or batch match what the publisher reports for the
  released run). Example or demo configs are not.

## 3. Write the record

`data/models/<id>.yaml`, with `name`, `org` (must exist in `data/orgs.yaml`; add the org with a
quoted, pinned country source if new), `kind` (base | instruct | reasoning), `sources`, `facts`, and
when needed `not_disclosed` and `config_unused`.

Every fact has `value`, `source`, `at` and either:

- a JSON pointer into a `config` source (`/num_hidden_layers`, `/a + /b`, `/layer_types | count
  sliding_attention`, `/list | len`, `/list/0:40 | count X`, `/weight_map | has mtp.`), or
- a short verbatim `quote` from the cited source that states the value for this model.

`at` is always a human locator: page and section for papers, heading for cards, file and function
for code. `basis: inferred` is only for trivial arithmetic on quoted inputs and needs a `note` with
the arithmetic. Notes are one terse sentence and only when they prevent a misreading.

Fill fields in this order, checking the definition in `fields.yaml` for each:

1. `meta.released` (first public weights, or the report if earlier; a dated publisher page, not a
   repository creation date) and `meta.license` (model card front matter or license file).
2. Parameters: `params.total`, `params.active` exactly as reported; `params.note` for scope
   (embeddings, MTP, vision, n-gram tables); `params.active_non_embedding` only when the publisher
   states it or states the scope of the active count.
3. Shape, attention, FFN, MoE, norms, context from config + code. Confirm in the modeling code that
   each config key is actually read.
4. MTP: depth used in pretraining, modules, block, FFN type, shared embedding/head, loss weight,
   whether weights were released (check the safetensors index).
5. Recipe from the report, read end to end including appendices: tokens, stages (each stage item
   needs its own quote; add `source` per item when a stage comes from another document), optimizer,
   LR peak/final, schedule, warmup, batch (`train.batch` as text and `train.batch_tokens` as a
   number), weight decay, betas, grad clip, precision, init, stability, data, hardware.
6. Scaling: `scaling.parametrization`, `scaling.hp_rule`, `scaling.ladder` when stated.
7. KV cache: if the attention is compressed or shares caches across layers, describe it with
   `attn.kv_groups` (one quote per group); set `attn.indexer_dim` for DSA-style indexers.

Use the canonical vocabulary in `CONTRIBUTING.md` (SwiGLU, RMSNorm, pre/sandwich, GQA/MLA/SWA, ...).
Keep the publisher's precision; bounds stay strings (">30T").

## 4. Traps that have produced wrong values before

- MLA `head_dim` in configs is often the RoPE sub-dimension; QK head dim is nope + rope.
- `rope_theta` on NoPE layers or a different layer type; per-layer-type `rope_parameters`.
- `full_attention` layers that run sparse attention.
- Config keys the code never reads (clamp lists, `partial_rotary_factor`, `enable_lm_head_fp32`).
- MTP declared in config but absent from the weights; a separate draft model is not MTP.
- Checkpoint dtype or quantization is not training precision.
- `initializer_range`, `router_aux_loss_coef` and other config coefficients are not evidence.
- Numbers from ablations, proxy models, sibling sizes or predecessor runs.
- Parameter counts with different scopes.
- `sliding_window: 0` or null means no window.

## 5. Make it complete and check it

```sh
uv run archdb validate <id>         # schema, pinning, required fields
uv run archdb check-sources <id>    # every pointer and quote, plus config coverage
uv run archdb count-params <id>     # parameter counts from checkpoint headers
uv run archdb build                 # exports and site data
uv run pytest -q
```

- `validate` lists required fields that are neither stated nor in `not_disclosed`. Add a field to
  `not_disclosed` only after reading every primary source; it is a claim that none states it.
- `check-sources` lists config keys with values that no fact cites. Record them, mention them in
  the note of the fact they qualify, or list them in `config_unused` with a specific reason
  ("not read by modeling_x.py"). Global runtime/multimodal keys belong in `data/config_keys.yaml`.
- `count-params` must reproduce the stated total and active counts; if the site shows a
  "does not reproduce" reason, the tensor naming is probably new to `archdb/checkpoint.py`.
- Benchmarks go in `data/evals.yaml` (keep entries sorted by model id): independent leaderboards only (Artificial Analysis
  Intelligence Index via a Wayback snapshot of the model page; DeepSWE leaderboard JSON). Name the
  evaluated variant in the note; for a base record, use the post-trained sibling and say so.

Done means validate, check-sources (0 problems, 0 facts with neither), pytest and
`build --check` all pass.

## 6. Audit before finishing

Re-read every fact against its source as if trying to prove it wrong: right model and size, right
scope (global vs sliding-window layers, backbone vs MTP, pretraining vs post-training), dead config
keys, precision kept, free text short and free of adjectives. Re-check each `not_disclosed` entry
against the report. Delete anything you cannot defend.
