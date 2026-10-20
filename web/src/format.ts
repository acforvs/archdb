import type { Field, Value } from "./data";

export function si(v: number): string {
  for (const [d, s] of [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]] as const)
    if (Math.abs(v) >= d) return `${+(v / d).toPrecision(3)}${s}`;
  return String(v);
}

export function bytes(v: number): string {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${+v.toPrecision(3)} ${units[i]}`;
}

export function lr(v: number): string {
  const [m, e] = v.toExponential(2).split("e");
  return `${+m}e${+e}`;
}

export function context(v: number): string {
  if (v >= 1_000_000 && v % 1_048_576 === 0) return `${v / 1_048_576}M`;
  if (v >= 1024 && v % 1024 === 0) return `${v / 1024}K`;
  return v.toLocaleString("en-US");
}

/** Human formatting by field unit; `full` disables truncation of long text. */
export function format(key: string, field: Field | undefined, v: Value | undefined, full = false): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) return `${v.length} stages`;
  if (typeof v === "string") return !full && v.length > 44 ? `${v.slice(0, 42)}…` : v;
  const unit = field?.unit;
  if (key === "moe.active_fraction") return `${(v * 100).toFixed(1)}%`;
  if (key === "moe.sparsity") return `${+v.toFixed(1)}×`;
  if (key === "eval.deepswe") return `${(v * 100).toFixed(1)}%`;
  if (unit === "ratio") return Math.abs(v) >= 10 ? Math.round(v).toLocaleString("en-US") : String(+v.toFixed(2));
  if (key === "shape.aspect") return v.toFixed(0);
  if (unit === "params" || key === "train.batch_tokens" || (unit === "tokens" && v >= 1e9)) return si(v);
  if (unit === "tokens") return context(v);
  if (unit === "lr") return lr(v);
  if (unit === "bytes") return bytes(v);
  if (Number.isInteger(v) || Math.abs(v) >= 100) return Math.round(v).toLocaleString("en-US");
  if (Math.abs(v) < 1e-3) return lr(v);
  return String(+v.toPrecision(4));
}
