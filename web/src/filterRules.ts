import { cell, numeric, type CellOpts, type DB, type Model } from "./data";

/** Model attributes that filter by membership rather than by a field value. */
export const ATTRS = { country: "Country", org: "Organization", kind: "Checkpoint", weights: "Weights" } as const;
export type Attr = keyof typeof ATTRS;

export type Op = ">=" | "<=" | "=" | "!=" | "~";
export interface Rule {
  key: string;
  op: Op;
  value: string;
}

const RULE = /^([\w.]+)(>=|<=|!=|=|~)(.*)$/;

/** Rules are stored in the URL as `key op value` joined by ';', e.g. `params.total>=100B;country=China|France`. */
export function parseRules(s: string): Rule[] {
  return s
    .split(";")
    .map((r) => RULE.exec(r))
    .filter((m): m is RegExpExecArray => m !== null)
    .map(([, key, op, value]) => ({ key, op: op as Op, value }));
}

export function serializeRules(rules: Rule[]): string {
  return rules.map((r) => `${r.key}${r.op}${r.value}`).join(";");
}

const SUFFIX: Record<string, number> = {
  k: 1e3, m: 1e6, b: 1e9, t: 1e12, kib: 2 ** 10, mib: 2 ** 20, gib: 2 ** 30, tib: 2 ** 40,
};

/** Number typed by a person: 100B, 1.5T, 128K, 2 GiB, 3e-4, 2026-01 for dates. */
export function parseNumber(s: string, date = false): number | null {
  const t = s.trim().toLowerCase().replace(/,/g, "");
  if (date) return /^\d{4}(-\d{1,2}){0,2}$/.test(t) ? numeric({ type: "date", label: "", definition: "" }, t) : null;
  const m = /^(-?[\d.]+(?:e[-+]?\d+)?)\s*([a-z]*)$/.exec(t);
  if (!m) return null;
  const mult = m[2] ? SUFFIX[m[2]] : 1;
  return mult === undefined || Number.isNaN(Number(m[1])) ? null : Number(m[1]) * mult;
}

function test(db: DB, m: Model, r: Rule, o: CellOpts): boolean {
  if (r.key in ATTRS) {
    const allowed = r.value.split("|");
    const v = String(m[r.key as Attr]);
    return r.op === "!=" ? !allowed.includes(v) : allowed.includes(v);
  }
  const field = db.fields[r.key];
  if (!field) return true;
  const v = cell(m, r.key, o)?.value;
  if (v === null || v === undefined) return r.op === "!=";
  if (typeof v === "boolean") return (r.op === "!=") !== (String(v) === r.value);
  if (r.op === "~" || field.type === "str") {
    const hit = String(v).toLowerCase().includes(r.value.toLowerCase());
    return r.op === "!=" ? !hit : hit;
  }
  const a = numeric(field, v);
  const b = parseNumber(r.value, field.type === "date");
  if (a === null || b === null) return false;
  const eps = 1e-9 * Math.max(1, Math.abs(b));
  if (r.op === ">=") return a >= b - eps;
  if (r.op === "<=") return a <= b + eps;
  if (r.op === "=") return Math.abs(a - b) <= eps;
  return Math.abs(a - b) > eps;
}

export interface Scope extends CellOpts {
  q: string;
  f: string;
  sel: string;
  only: boolean;
}

/** Models left after name search, field filters and the "selected only" switch. Order follows db.models. */
export function scopeModels(db: DB, v: Scope): Model[] {
  const terms = v.q.toLowerCase().split(",").map((t) => t.trim()).filter(Boolean);
  const rules = parseRules(v.f);
  const sel = new Set(v.sel.split(",").filter(Boolean));
  return db.models.filter(
    (m) =>
      (!terms.length || terms.some((t) => `${m.id} ${m.name} ${m.org}`.toLowerCase().includes(t))) &&
      rules.every((r) => test(db, m, r, v)) &&
      (!v.only || sel.has(m.id)),
  );
}
