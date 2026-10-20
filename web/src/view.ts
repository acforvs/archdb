export interface View {
  tab: "table" | "plot" | "breakdown";
  preset: string;
  sort: string;
  dir: 1 | -1;
  x: string;
  y: string;
  xlog: boolean;
  ylog: boolean;
  ctx: number;
  q: string;
  model: string;
  focus: string;
  all: boolean;
  cols: string;
  side: boolean;
  /** Filter rules, see filters.ts. */
  f: string;
  /** Selected model ids, comma-separated, and whether to show only them. */
  sel: string;
  only: boolean;
  kvdt: string;
  pareto: string;
  /** Breakdown: group bars by, and the property split within each bar. */
  by: string;
  feat: string;
  /** Breakdown: leave out models that do not disclose the split property. */
  disc: boolean;
}

export const DEFAULTS: View = {
  tab: "table", preset: "Overview", sort: "params.total", dir: -1, x: "kv.bytes_per_token", y: "eval.aa_index",
  xlog: true, ylog: false, ctx: 131072, q: "", model: "", focus: "", all: false, cols: "", side: false,
  f: "", sel: "", only: false, kvdt: "bf16", pareto: "x-y+", by: "country", feat: "swa", disc: false,
};
export function parseView(query: string): View {
  const p = new URLSearchParams(query);
  const v: Record<string, unknown> = { ...DEFAULTS };
  for (const [k, raw] of p) {
    if (!Object.hasOwn(DEFAULTS, k)) continue;
    const d = DEFAULTS[k as keyof View];
    v[k] = typeof d === "boolean" ? raw === "1" : typeof d === "number" ? Number(raw) : raw;
  }
  return v as unknown as View;
}

export function serializeView(v: View, includeDefaults = false): string {
  const p = new URLSearchParams();
  for (const [k, val] of Object.entries(v))
    if (includeDefaults || val !== DEFAULTS[k as keyof View]) p.set(k, typeof val === "boolean" ? (val ? "1" : "0") : String(val));
  return p.toString();
}
