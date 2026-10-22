import type { Model } from "./data";

/** A categorical reading of a record, for breakdown charts. null means the sources do not say. */
export interface Feature {
  label: string;
  /** Categories in display order. */
  order: string[];
  of: (m: Model) => string | null;
}

const value = (m: Model, key: string) => m.facts[key]?.value;
const num = (m: Model, key: string) => {
  const v = value(m, key);
  return typeof v === "number" ? v : null;
};
const text = (m: Model, key: string) => String(value(m, key) ?? "");
const has = (m: Model, key: string) => key in m.facts;
const yesNo = (b: boolean | null) => (b === null ? null : b ? "yes" : "no");

const LINEAR = /DeltaNet|KDA|Lightning|Mamba|linear/;
const SPARSE = /DSA|CSA|HCA|MSA|QSA/;

export const FEATURES: Record<string, Feature> = {
  swa: {
    label: "Sliding-window attention",
    order: ["yes", "no"],
    of: (m) => (has(m, "attn.kind") ? yesNo((num(m, "attn.local_layers") ?? 0) > 0 || /SWA/.test(text(m, "attn.kind"))) : null),
  },
  linear: {
    label: "Linear / SSM layers",
    order: ["yes", "no"],
    of: (m) => (has(m, "attn.kind") ? yesNo((num(m, "attn.linear_layers") ?? 0) > 0 || LINEAR.test(text(m, "attn.kind"))) : null),
  },
  mla: {
    label: "MLA",
    order: ["yes", "no"],
    of: (m) => (has(m, "attn.kind") ? yesNo(/MLA/.test(text(m, "attn.kind"))) : null),
  },
  sparse: {
    label: "Sparse attention (DSA, CSA, MSA…)",
    order: ["yes", "no"],
    of: (m) => (has(m, "attn.kind") ? yesNo(SPARSE.test(text(m, "attn.kind")) || has(m, "attn.sparse")) : null),
  },
  moe: {
    label: "Mixture of experts",
    order: ["MoE", "dense"],
    of: (m) => (has(m, "moe.experts") ? "MoE" : has(m, "ffn.dim") ? "dense" : null),
  },
  size: {
    label: "Total parameters",
    order: ["≥ 1T", "500B–1T", "100–500B", "< 100B"],
    of: (m) => {
      const t = num(m, "params.total");
      if (t === null) return null;
      return t >= 1e12 ? "≥ 1T" : t >= 5e11 ? "500B–1T" : t >= 1e11 ? "100–500B" : "< 100B";
    },
  },
  optimizer: {
    label: "Optimizer",
    order: ["Muon family", "AdamW", "other"],
    of: (m) => {
      const v = text(m, "train.optimizer");
      if (!v) return null;
      return /Muon/.test(v) ? "Muon family" : /^Adam/.test(v) ? "AdamW" : "other";
    },
  },
  mtp: {
    label: "MTP in pretraining",
    order: ["yes", "no"],
    of: (m) => {
      const d = num(m, "mtp.depth");
      return d === null ? null : yesNo(d > 0);
    },
  },
  position: {
    label: "Positional encoding (global layers)",
    order: ["RoPE", "partial RoPE", "NoPE", "other"],
    of: (m) => {
      const v = text(m, "attn.position");
      if (!v) return null;
      if (/NoPE \(global|^NoPE/.test(v)) return "NoPE";
      if (/partial/i.test(v)) return "partial RoPE";
      return /RoPE/.test(v) ? "RoPE" : "other";
    },
  },
  norm: {
    label: "Norm placement",
    order: ["pre", "sandwich", "post", "parallel"],
    of: (m) => text(m, "norm.placement").split(/[ (]/)[0] || null,
  },
  qknorm: {
    label: "QK norm",
    order: ["yes", "no"],
    of: (m) => {
      const v = value(m, "attn.qk_norm");
      return typeof v === "boolean" ? yesNo(v) : null;
    },
  },
  precision: {
    label: "Training precision",
    order: ["FP4", "FP8", "BF16"],
    of: (m) => {
      const v = text(m, "train.precision");
      if (!v) return null;
      return /FP4/.test(v) ? "FP4" : /FP8/.test(v) ? "FP8" : /BF16/.test(v) ? "BF16" : null;
    },
  },
  tied: {
    label: "Tied embeddings",
    order: ["yes", "no"],
    of: (m) => {
      const v = value(m, "shape.tied_embeddings");
      return typeof v === "boolean" ? yesNo(v) : null;
    },
  },
  weights: { label: "Weights", order: ["open", "closed"], of: (m) => m.weights },
};

/** One bar per feature for the "all features" breakdown: categories in `yes` count as having it. */
export const OVERVIEW: { feat: string; label: string; yes: string[] }[] = [
  { feat: "moe", label: "Mixture of experts", yes: ["MoE"] },
  { feat: "qknorm", label: "QK norm", yes: ["yes"] },
  { feat: "swa", label: "Sliding-window attention", yes: ["yes"] },
  { feat: "mla", label: "MLA", yes: ["yes"] },
  { feat: "linear", label: "Linear / SSM layers", yes: ["yes"] },
  { feat: "sparse", label: "Sparse attention", yes: ["yes"] },
  { feat: "mtp", label: "MTP in pretraining", yes: ["yes"] },
  { feat: "optimizer", label: "Muon-family optimizer", yes: ["Muon family"] },
  { feat: "precision", label: "FP8 / FP4 training", yes: ["FP8", "FP4"] },
  { feat: "tied", label: "Tied embeddings", yes: ["yes"] },
];

/** How models are grouped into bars. */
export const GROUPS: Record<string, { label: string; of: (m: Model) => string | null }> = {
  features: { label: "Feature (all features)", of: () => null },
  country: { label: "Country", of: (m) => m.country },
  year: { label: "Release year", of: (m) => String(value(m, "meta.released") ?? "").slice(0, 4) || null },
  half: {
    label: "Release half-year",
    of: (m) => {
      const d = String(value(m, "meta.released") ?? "");
      return d ? `${d.slice(0, 4)} H${Number(d.slice(5, 7)) <= 6 ? 1 : 2}` : null;
    },
  },
  org: { label: "Organization", of: (m) => m.org },
  kind: { label: "Checkpoint", of: (m) => m.kind },
};
