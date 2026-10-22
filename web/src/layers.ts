import type { Model } from "./data";

export type Mixer = "global" | "swa" | "linear";

export interface Stack {
  layers: Mixer[];
  /** Detailed types in the same order, e.g. CSA + SWA rather than just global attention. */
  types: string[];
  ordered: boolean;
}

export const MIXER_LABEL: Record<Mixer, string> = { global: "global attention", swa: "sliding window", linear: "linear / SSM" };
export const LAYER_MIXERS: Record<string, Mixer> = {
  global: "global", GQA: "global", MHA: "global", MQA: "global", MLA: "global", "MLA+DSA": "global",
  DSA: "global", QSA: "global", MSA: "global", GDLA: "global",
  "CSA + SWA": "global", "HCA + SWA": "global", "CSA2 m=2 + SWA": "global", "CSA2 m=1 + SWA": "global",
  SWA: "swa", KDA: "linear", "Gated DeltaNet": "linear", Lightning: "linear", "Mamba-2": "linear",
};

const num = (m: Model, k: string) => {
  const v = m.facts[k]?.value;
  return typeof v === "number" ? v : null;
};

/** Use sourced per-layer order. A ratio or a sum alone never determines the sequence. */
export function layerStack(m: Model): Stack | null {
  const n = num(m, "shape.layers");
  const counts: Record<Mixer, number> = {
    global: num(m, "attn.global_layers") ?? -1,
    swa: num(m, "attn.local_layers") ?? 0,
    linear: num(m, "attn.linear_layers") ?? 0,
  };
  if (!n || !Number.isSafeInteger(n) || Object.values(counts).some((c) => !Number.isSafeInteger(c) || c < 0)
    || counts.global + counts.swa + counts.linear !== n) return null;

  const sequence = m.facts["attn.layout"]?.sequence;
  if (sequence?.length === n && sequence.every((type) => Object.hasOwn(LAYER_MIXERS, type))) {
    const layers = sequence.map((type) => LAYER_MIXERS[type]);
    if ((Object.keys(counts) as Mixer[]).every((k) => layers.filter((x) => x === k).length === counts[k]))
      return { layers, types: sequence, ordered: true };
  }
  if (!sequence && counts.swa === 0 && counts.linear === 0)
    return { layers: Array<Mixer>(n).fill("global"), types: Array<string>(n).fill("global attention"), ordered: true };
  const layers = (["global", "swa", "linear"] as Mixer[]).flatMap((k) => Array<Mixer>(counts[k]).fill(k));
  return { layers, types: layers.map((k) => MIXER_LABEL[k]), ordered: false };
}
