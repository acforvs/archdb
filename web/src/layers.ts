import type { Model } from "./data";

/** Token-mixer type of one layer: global attention (full, sparse or latent), sliding window, or linear / SSM. */
export type Mixer = "global" | "swa" | "linear";

export interface Stack {
  layers: Mixer[];
  /** False when the recorded layout cannot resolve the sequence; layers are then grouped by type. */
  ordered: boolean;
}

export const MIXER_LABEL: Record<Mixer, string> = { global: "global attention", swa: "sliding window", linear: "linear / SSM" };

function mixerOf(name: string): Mixer | null {
  if (/SWA|sliding|local/i.test(name)) return "swa";
  if (/KDA|DeltaNet|Lightning|Mamba|linear|GLA|RWKV/i.test(name)) return "linear";
  return /^(global|full|GQA|MHA|MQA|MLA(?:\+DSA)?|DSA|CSA\d*|HCA|QSA|MSA)$/i.test(name.trim()) ? "global" : null;
}

const num = (m: Model, k: string) => {
  const v = m.facts[k]?.value;
  return typeof v === "number" ? v : null;
};

/**
 * One entry per layer, built from the sourced counts (attn.global_layers, attn.local_layers, attn.linear_layers)
 * and the order stated in attn.layout. A repeating ratio such as "3 KDA : 1 MLA" is laid out in the order written,
 * after any leading runs ("2 SWA, then …"); notes such as "global first" or "leading global layer" are honored.
 * The resulting sequence must exactly reproduce all counts. Sums and ambiguous remainders are drawn grouped.
 */
export function layerStack(m: Model): Stack | null {
  const n = num(m, "shape.layers");
  const left: Record<Mixer, number> = {
    global: num(m, "attn.global_layers") ?? -1,
    swa: num(m, "attn.local_layers") ?? 0,
    linear: num(m, "attn.linear_layers") ?? 0,
  };
  if (!n || !Number.isSafeInteger(n) || Object.values(left).some((c) => !Number.isSafeInteger(c) || c < 0)
    || left.global + left.swa + left.linear !== n) return null;
  const counts = { ...left };
  const grouped = (): Stack => ({
    layers: (["global", "swa", "linear"] as Mixer[]).flatMap((k) => Array<Mixer>(counts[k]).fill(k)),
    ordered: false,
  });
  if (left.swa === 0 && left.linear === 0) return { layers: Array<Mixer>(n).fill("global"), ordered: true };

  const raw = String(m.facts["attn.layout"]?.value ?? "");
  // Parentheses group a pattern only when they hold a ratio; otherwise they are notes.
  const text = raw.replace(/\(([^()]*)\)/g, (all, inner: string) => (inner.includes(" : ") ? all : ""));
  const ratios = [...text.matchAll(/\(?(\d+) ([A-Za-z][\w\- ]*?) : (\d+) ([A-Za-z][\w\-+]*)/g)];
  const seq: Mixer[] = [];
  const runs = (s: string): Mixer[] | null => {
    const result: Mixer[] = [];
    for (const r of s.matchAll(/(\d+) ([A-Za-z][\w\-]*)/g)) {
      const k = mixerOf(r[2]);
      const count = Number(r[1]);
      if (!k || count > n) return null;
      result.push(...Array<Mixer>(count).fill(k));
    }
    return result;
  };

  if (ratios.length) {
    if (/leading global/i.test(raw)) seq.push("global");
    const prefix = runs(text.slice(0, ratios[0].index));
    if (!prefix) return grouped();
    seq.push(...prefix);
    for (let i = 0; i < ratios.length; i++) {
      const ratio = ratios[i];
      const first = mixerOf(ratio[2]), second = mixerOf(ratio[4]);
      const a = Number(ratio[1]), b = Number(ratio[3]);
      if (!first || !second || a + b > n || a + b === 0) return grouped();
      let block = [...Array<Mixer>(a).fill(first), ...Array<Mixer>(b).fill(second)];
      if (/(global|GQA|full) first/i.test(raw)) block = [...Array<Mixer>(b).fill(second), ...Array<Mixer>(a).fill(first)];
      const end = ratio.index! + ratio[0].length;
      const suffix = text.slice(end, ratios[i + 1]?.index);
      const after = /^\s*\)?\s*×\s*(\d+)/.exec(suffix);
      const before = /(\d+)\s*×\s*\(?\s*$/.exec(text.slice(0, ratio.index));
      const repetitions = after ? Number(after[1]) : before ? Number(before[1]) : null;
      const tail = runs(suffix.replace(/^\s*\)?\s*×\s*\d+/, ""));
      if (!tail || i < ratios.length - 1 && repetitions === null) return grouped();
      const length = repetitions === null ? n - seq.length - tail.length : repetitions * block.length;
      if (length < 0 || length > n - seq.length) return grouped();
      seq.push(...Array.from({ length }, (_, j) => block[j % block.length]), ...tail);
    }
  } else if (/then/.test(text)) {
    const sequence = runs(text);
    if (!sequence) return grouped();
    seq.push(...sequence);
  } else {
    return grouped();
  }
  return seq.length === n && (["global", "swa", "linear"] as Mixer[]).every((k) => seq.filter((x) => x === k).length === counts[k])
    ? { layers: seq, ordered: true } : grouped();
}
