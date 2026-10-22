import type { View } from "./App";

/** Named scatter plots; filters the reader set are kept. A Pareto front is drawn where one direction
 * is clearly better on both axes. */
export const PLOT_PRESETS: Record<string, Partial<View>> = {
  "Capability vs KV / token": { x: "kv.bytes_per_token", y: "eval.aa_index", xlog: true, ylog: false, pareto: "x-y+" },
  "Capability vs global KV": { x: "kv.growth_per_token", y: "eval.aa_index", xlog: true, ylog: false, pareto: "x-y+" },
  "Capability vs active params": {
    x: "params.counted_active_non_embedding", y: "eval.aa_index", xlog: true, ylog: false, pareto: "x-y+",
  },
  "KV / token / B active vs size": {
    x: "params.total", y: "kv.bytes_per_token_per_active", xlog: true, ylog: true, pareto: "",
  },
  "Tokens per param vs size": { x: "params.active", y: "scaling.tpp_active", xlog: true, ylog: true, pareto: "" },
  "Aspect ratio vs size": { x: "params.active", y: "shape.aspect", xlog: true, ylog: false, pareto: "" },
  "Batch vs training tokens": { x: "train.tokens", y: "train.batch_tokens", xlog: true, ylog: true, pareto: "" },
};

/** Named breakdowns. Those with `f` set that filter; the others keep the reader's filters. */
export const BREAKDOWN_PRESETS: Record<string, Partial<View>> = {
  "All features": { by: "features" },
  "SWA by country, 2026": { by: "country", feat: "swa", f: "meta.released>=2026-01" },
  "Model size, 2025 vs 2026": { by: "year", feat: "size", f: "meta.released>=2025-01" },
  "Muon over time": { by: "half", feat: "optimizer" },
  "Linear attention by country": { by: "country", feat: "linear" },
  "MTP over time": { by: "half", feat: "mtp" },
};

export function active(presets: Record<string, Partial<View>>, view: View): string | undefined {
  return Object.keys(presets).find((name) =>
    Object.entries(presets[name]).every(([k, v]) => view[k as keyof View] === v),
  );
}
