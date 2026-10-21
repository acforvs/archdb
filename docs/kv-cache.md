# KV cache calculations

These values measure the retained attention cache for one sequence after processing a given number
of tokens. They are not peak prefill memory or a full serving-memory estimate.

The website has two kinds of storage setting:

- **Uniform BF16, FP8 and FP4** count every cache scalar at 16, 8 or 4 bits. These compare cache shapes
  at a common precision. They exclude quantization metadata and do not claim that a released model
  uses that format.
- **Sourced storage** uses each component's documented format, quantization scales and padding.
  It is available only when every cache group has a complete recipe. It does not substitute BF16
  for missing precision information. The current recipes cover DeepSeek-V3.2, V4-Flash, V4-Pro and
  V4.1-Flash. The V4-Flash serving recipe belongs to the post-trained sibling of our base record.

`exports/models.csv` and the top-level derived values in `exports/archdb.json` use uniform BF16 at
131,072 tokens. The JSON also contains the sourced cache components and their evidence. In Python,
`derive(model.facts, context=131_072, kv_dtype="sourced")` selects sourced storage; the default is
`kv_dtype="bf16"`.

## Formula version `kv-2`

For a cache group with `L` owners, `E` scalar elements per entry, compression ratio `r` and context `S`:

```text
uniform bytes per entry = E × bits / 8
sourced bytes per entry = sum over components of:
    ceil(elements × bits / 8) + scale_count × scale_bits / 8 + padding_bytes

completed entries = floor(S / r)
windowed entries  = min(completed entries, window)
group bytes       = L × bytes per entry × entries

KV at context       = sum(group bytes)
KV bytes / token    = KV at context / S
global KV / token   = sum(L × bytes per entry / r) over non-windowed groups
KV / B active       = KV value / (reported active parameters / 1e9)
```

`global KV / token` is a long-context rate. At a context that does not complete a compression group,
the retained global payload is slightly smaller than that rate multiplied by context length.
Unfinished compressor state is excluded. This fixes the earlier formula's fractional cache entries.

Standard GQA stores `kv_heads × (key_dim + value_dim)` scalars per layer and token. Local layers can
have different head counts and separate key/value dimensions. MiMo's SWA uses key dim 192 and value
dim 128; counting both as 192 overstated its local cache by 20% in the previous formula.

MLA comparisons assume compact latent caching: `kv_lora_rank + rope_dim`, plus an indexer key where
one is owned. They do not count the expanded, per-head K/V tensors that some reference implementations
retain. Compressed attention and cross-layer reuse use explicit `attn.kv_groups`, counting each owner
once. GLM-5.3-Flash's IndexPool group describes compact pooled keys; its Transformers implementation
instead retains unpooled indexer states. These uniform comparisons are not measurements of that code.

SWA is included in finite-context totals and capped at the window. It contributes no long-context
global growth. Recurrent/SSM states, compressor tail state, draft/MTP caches, temporary prefill tensors,
allocator/page metadata and tensor-parallel replication are excluded. Sourced entry padding is included
where documented. Persistent prefix caches and SWA snapshots on SSD/host memory are a separate scope.

## Checks against published figures

[DeepSeek-V4.1 Figure 1(b)](https://arxiv.org/pdf/2609.19969v1#page=1) plots **global** KV, including
indexer keys, at the models' serving formats. It excludes bounded SWA. The sourced calculations give:

| Model | Global bytes/token | Calculation |
|---|---:|---|
| DeepSeek-V3.2 | 48,068 | 61 × (656 main KV + 132 indexer) |
| DeepSeek-V4-Flash | 3,514.25 | (21/4) × (584 main KV + 68 indexer) + (20/128) × 584 |
| DeepSeek-V4.1-Flash | 890 | (3/2 + 1) × (288 main KV + 68 indexer) |

The figure rounds V4-Flash to 3,514. V4-Pro uses its own layer counts and gives 5,031.4375 global
bytes/token, consistent with the roughly 9.5-fold reduction from V3.2 in its
[report](https://arxiv.org/pdf/2606.19348v1#page=1).

For V4.1, main KV is 512 FP4 values plus 32 one-byte scales: 288 bytes/entry. Indexer K is 128 FP4
values plus four one-byte scales: 68 bytes/entry. SWA remains FP8, with 512 values and 16 scale bytes.
At 128K context, including all 40 SWA rings gives **910.625 bytes/token**, versus **890** for global KV
alone. Dividing the uniform-BF16 global value, 3,200, by four would give 800 and miss the scale bytes.

The layouts are documented in the pinned
[FlashMLA README](https://github.com/deepseek-ai/FlashMLA/blob/07a1089857b63e74e3133630c02b083b75e8d4b2/README.md).
Indexer scales are sourced from the corresponding pinned inference code in each model record.
Tests check these published figures, finite SWA windows, separate value dimensions, partial recipes
and compression boundaries in both Python and the browser calculation.

The October 2026 audit rechecked 339 cache-related config inputs without finding value mismatches.
The pinned Apertus config required access, and the pinned AX-K2 config returned 404; those inputs
could not be independently re-fetched. This check does not establish the memory footprint of every
inference framework.
