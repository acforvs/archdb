export type Value = number | string | boolean | Stage[] | null;

/** Stage numbers; a string is a bound or approximation as stated, e.g. "~2B" or ">35T". */
type Amount = number | string;

export interface Stage {
  name: string;
  tokens?: Amount;
  seq_len?: Amount;
  lr?: Amount;
  batch_tokens?: Amount;
  batch_sequences?: Amount;
  steps?: Amount;
  note?: string;
  quote?: string;
  source?: string;
}

export interface Fact {
  value: Value;
  source: string;
  at: string;
  quote?: string;
  note?: string;
  basis?: "inferred";
}

export interface Derived {
  value: number | null;
  formula: string;
}

export interface Model {
  id: string;
  name: string;
  org: string;
  country: string;
  kind: "base" | "instruct" | "reasoning";
  weights: "open" | "closed";
  note: string | null;
  sources: Record<string, { url: string; kind: string }>;
  facts: Record<string, Fact>;
  not_disclosed: string[];
  derived: Record<string, Derived>;
  /** BF16 bytes added per token, plus [bytes per window entry, window] for sliding-window caches. */
  kv: { per_token: number; windows: [number, number][] } | null;
}

export interface Field {
  type: "int" | "float" | "str" | "bool" | "list" | "date";
  unit?: string;
  label: string;
  definition: string;
  derived?: boolean;
  required?: string;
  aliases?: string[];
}

export interface DB {
  groups: Record<string, string>;
  fields: Record<string, Field>;
  models: Model[];
}

/** A value shown in a cell: an authored fact, a computed one, or a value the sources do not disclose. */
export type Cell =
  | ({ kind: "fact" } & Fact)
  | { kind: "derived"; value: number | null; formula: string }
  | { kind: "undisclosed"; value: null };

/** Settings that change computed values: context length and KV-cache bytes per element. */
export interface CellOpts {
  ctx: number;
  kvdt: string;
}

export const KV_DTYPES: Record<string, number> = { bf16: 2, fp8: 1 };

/** KV fields that depend on the selected context and are recomputed in the browser. */
export const KV_CONTEXT_KEYS = [
  "kv.bytes_per_token", "kv.bytes_per_token_per_active", "kv.bytes_at_context", "kv.bytes_at_context_per_active",
];

export function cell(m: Model, key: string, o: CellOpts): Cell | undefined {
  const kvScale = (KV_DTYPES[o.kvdt] ?? 2) / 2;
  if (KV_CONTEXT_KEYS.includes(key)) {
    const d = m.derived[key];
    if (!d) return undefined;
    if (!m.kv) return { kind: "derived", value: null, formula: d.formula };
    const dt = o.kvdt.toUpperCase();
    const total = (m.kv.per_token * o.ctx + m.kv.windows.reduce((s, [b, w]) => s + b * Math.min(o.ctx, w), 0)) * kvScale;
    const active = m.facts["params.active"]?.value;
    const perB = typeof active === "number" ? active / 1e9 : null;
    const totalFormula =
      `${(m.kv.per_token * kvScale).toLocaleString("en-US")} B/token × ${o.ctx.toLocaleString("en-US")}` +
      m.kv.windows.map(([b, w]) => ` + ${(b * kvScale).toLocaleString("en-US")} B × min(context, ${w})`).join("") +
      ` (${dt})`;
    if (key === "kv.bytes_at_context") return { kind: "derived", value: total, formula: totalFormula };
    if (key === "kv.bytes_per_token")
      return { kind: "derived", value: total / o.ctx, formula: `(${totalFormula}) / ${o.ctx.toLocaleString("en-US")}` };
    if (perB === null) return { kind: "derived", value: null, formula: d.formula };
    if (key === "kv.bytes_at_context_per_active")
      return { kind: "derived", value: total / perB, formula: `(${totalFormula}) / ${perB}B active` };
    return { kind: "derived", value: total / o.ctx / perB, formula: `(${totalFormula}) / ${o.ctx.toLocaleString("en-US")} / ${perB}B active` };
  }
  const f = m.facts[key];
  if (f) return { kind: "fact", ...f };
  const d = m.derived[key];
  if (d) {
    if (key.startsWith("kv.") && d.value !== null && kvScale !== 1)
      return { kind: "derived", value: d.value * kvScale, formula: `${d.formula}, scaled to ${o.kvdt.toUpperCase()}` };
    return { kind: "derived", ...d };
  }
  if (m.not_disclosed.includes(key)) return { kind: "undisclosed", value: null };
  return undefined;
}

/** Scores in data/evals.yaml measure post-trained checkpoints; on a base record they belong to a sibling. */
export function siblingScore(m: Model, key: string): boolean {
  return key.startsWith("eval.") && m.kind === "base";
}

/** Numeric value of a cell for sorting, filtering and plotting; dates become fractional years. */
const SCALE: Record<string, number> = { K: 1e3, M: 1e6, B: 1e9, T: 1e12 };

export function numeric(field: Field, v: Value | undefined): number | null {
  if (typeof v === "number") return v;
  // Bounds stated by the publisher (">35T", "~0.22T") sort and filter at their stated value.
  const bound = typeof v === "string" ? /^[~<>≥≤]\s*([\d.]+)\s*([KMBT]?)$/.exec(v) : null;
  if (bound) return Number(bound[1]) * (SCALE[bound[2]] ?? 1);
  if (field.type === "date" && typeof v === "string") {
    const [y, mo = 1, d = 1] = v.split("-").map(Number);
    return y + (Date.UTC(y, mo - 1, d) - Date.UTC(y, 0, 1)) / (365.25 * 864e5);
  }
  return null;
}

export const PRESETS: Record<string, string[]> = {
  Overview: [
    "meta.released", "params.total", "params.active", "moe.active_fraction", "shape.layers", "shape.d_model",
    "attn.layout", "kv.bytes_per_token", "kv.bytes_at_context", "ctx.final", "train.tokens", "train.optimizer",
    "train.lr_peak",
  ],
  Shape: [
    "params.total", "params.active", "params.counted_active_non_embedding", "shape.layers", "shape.d_model", "shape.aspect", "attn.heads", "attn.kv_heads",
    "attn.group_size", "attn.head_dim", "attn.qk_width", "ffn.dim", "ffn.mult", "moe.experts", "moe.top_k",
    "moe.shared", "moe.expert_dim", "moe.shared_dim", "moe.granularity", "moe.latent_dim", "shape.vocab",
    "shape.tied_embeddings",
  ],
  "Attention & KV": [
    "attn.kind", "attn.layout", "attn.swa_ratio", "attn.linear_ratio", "attn.global_layers", "attn.local_layers",
    "attn.long_window_layers", "attn.linear_layers", "attn.heads", "attn.kv_heads", "attn.head_dim", "attn.v_head_dim",
    "attn.kv_lora_rank", "attn.rope_dim", "attn.window", "attn.local_heads", "attn.local_kv_heads",
    "attn.local_head_dim", "attn.position", "attn.rope_theta", "attn.local_rope_theta", "attn.rope_fraction",
    "attn.qk_norm", "attn.gate", "attn.scale", "attn.sinks", "attn.sparse", "attn.indexer_dim", "attn.kv_sharing",
    "attn.linear_kind", "kv.bytes_per_token", "kv.growth_per_token", "kv.bytes_per_token_per_active", "kv.bytes_at_context",
    "kv.bytes_at_context_per_active", "params.active_prefill",
  ],
  MoE: [
    "params.total", "params.active", "moe.active_fraction", "moe.experts", "moe.top_k", "moe.sparsity", "moe.shared",
    "moe.expert_dim", "moe.granularity", "moe.shared_dim", "moe.latent_dim", "moe.latent_ratio", "moe.dense_layers",
    "moe.router", "moe.norm_topk", "moe.routed_scale", "moe.groups", "moe.balancing",
  ],
  "Norms & FFN": [
    "norm.type", "norm.placement", "norm.eps", "norm.residual", "norm.embedding", "norm.output_scale",
    "norm.logit_softcap", "attn.qk_norm", "attn.softcap", "ffn.activation", "ffn.clamp", "ffn.dim", "ffn.bias",
    "attn.bias",
  ],
  MTP: [
    "mtp.depth", "mtp.modules", "mtp.block", "mtp.ffn", "mtp.shares_embedding", "mtp.params", "mtp.loss_weight",
    "mtp.released",
  ],
  Recipe: [
    "train.tokens", "train.optimizer", "train.lr_peak", "train.lr_min", "train.schedule", "train.warmup",
    "train.batch", "train.batch_tokens", "train.weight_decay", "train.betas", "train.grad_clip", "train.dropout", "train.precision", "train.stability",
    "train.init", "train.masking", "train.objective", "ctx.pretrain", "ctx.final", "ctx.extension", "train.averaging",
  ],
  Scaling: [
    "params.total", "params.active", "params.counted_active_non_embedding", "train.tokens", "train.lr_peak",
    "train.batch_tokens", "train.weight_decay", "train.init", "scaling.tpp", "scaling.tpp_active", "scaling.adamw_tau",
    "scaling.batch_ratio", "scaling.parametrization", "scaling.hp_rule", "scaling.ladder", "norm.output_scale",
    "norm.embedding",
  ],
  "Data & compute": [
    "meta.released", "meta.license", "train.tokens", "train.data", "train.cutoff", "train.hardware", "train.compute",
    "shape.tokenizer", "shape.vocab", "shape.memory",
  ],
  Evals: [
    "meta.released", "params.total", "params.active", "eval.aa_index", "eval.deepswe",
    "kv.bytes_per_token", "kv.bytes_at_context", "train.tokens",
  ],
};

export function isNumeric(f: Field): boolean {
  return f.type === "date" || f.type === "int" || f.type === "float";
}

export function numericFields(db: DB): string[] {
  return Object.keys(db.fields).filter((k) => isNumeric(db.fields[k]));
}

/** Text a field search matches: key, label, definition and aliases. */
export function fieldText(k: string, f: Field): string {
  return `${k} ${f.label} ${f.definition} ${(f.aliases ?? []).join(" ")}`.toLowerCase();
}

export function hfView(url: string): string {
  return url.replace("/resolve/", "/blob/");
}
