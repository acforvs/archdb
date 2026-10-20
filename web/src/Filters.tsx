import { useEffect, useMemo, useRef, useState } from "react";
import type { DB } from "./data";
import type { View } from "./App";
import { FieldSelect } from "./FieldSelect";
import { ATTRS, parseNumber, parseRules, serializeRules, type Attr, type Op, type Rule } from "./filterRules";
import { format } from "./format";

const OPS: Record<string, Op[]> = { num: [">=", "<=", "=", "!="], str: ["~", "!="], bool: ["=", "!="] };
const OP_LABEL: Record<Op, string> = { ">=": "≥", "<=": "≤", "=": "=", "!=": "≠", "~": "contains" };

function kindOf(db: DB, key: string) {
  const t = db.fields[key]?.type;
  return t === "str" ? "str" : t === "bool" ? "bool" : "num";
}

/** Filter button with a popover (organization, country, checkpoint, any field) and the active filters as chips. */
export function Filters({ db, view, set }: { db: DB; view: View; set: (p: Partial<View>) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const rules = parseRules(view.f);
  const save = (next: Rule[]) => set({ f: serializeRules(next) });

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  const options = useMemo(() => {
    const out = {} as Record<Attr, [string, number][]>;
    for (const a of Object.keys(ATTRS) as Attr[]) {
      const n = new Map<string, number>();
      for (const m of db.models) n.set(String(m[a]), (n.get(String(m[a])) ?? 0) + 1);
      out[a] = [...n].sort((x, y) => y[1] - x[1] || x[0].localeCompare(y[0]));
    }
    return out;
  }, [db]);

  const attrRule = (a: Attr) => rules.find((r) => r.key === a && r.op === "=");
  const toggleAttr = (a: Attr, v: string) => {
    const cur = attrRule(a)?.value.split("|").filter(Boolean) ?? [];
    const vals = cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v];
    const rest = rules.filter((r) => r !== attrRule(a));
    save(vals.length ? [...rest, { key: a, op: "=", value: vals.join("|") }] : rest);
  };

  const fieldKeys = useMemo(() => Object.keys(db.fields).filter((k) => db.fields[k].type !== "list"), [db]);
  const [draft, setDraft] = useState<Rule>({ key: "params.total", op: ">=", value: "" });
  const draftKind = kindOf(db, draft.key);
  const draftValid =
    draftKind === "num"
      ? parseNumber(draft.value, db.fields[draft.key]?.type === "date") !== null
      : draftKind === "bool" ? ["true", "false"].includes(draft.value) : draft.value.trim() !== "";
  const addDraft = () => {
    if (!draftValid) return;
    save([...rules, { ...draft, value: draft.value.trim() }]);
    setDraft({ ...draft, value: "" });
  };

  const label = (r: Rule) => {
    if (r.key in ATTRS) return `${ATTRS[r.key as Attr]} ${r.op === "!=" ? "not " : ""}${r.value.split("|").join(", ")}`;
    const f = db.fields[r.key];
    const n = parseNumber(r.value, f?.type === "date");
    const shown = kindOf(db, r.key) === "num" && n !== null && f?.type !== "date" ? format(r.key, f, n) : r.value;
    return `${f?.label ?? r.key} ${OP_LABEL[r.op]} ${shown}`;
  };

  return (
    <>
      <div className="picker" ref={ref}>
        <button className="chip-btn" aria-expanded={open} onClick={() => setOpen(!open)}>
          Filter{rules.length ? ` (${rules.length})` : ""} ▾
        </button>
        {open && (
          <div className="picker-pop filter-pop">
            {(Object.keys(ATTRS) as Attr[]).map((a) => (
              <fieldset key={a}>
                <legend>{ATTRS[a]}</legend>
                <div className="checks">
                  {options[a].map(([v, n]) => (
                    <label key={v}>
                      <input
                        type="checkbox"
                        checked={attrRule(a)?.value.split("|").includes(v) ?? false}
                        onChange={() => toggleAttr(a, v)}
                      />
                      {v} <span className="count">{n}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}
            <fieldset>
              <legend>Field</legend>
              <div className="rule-row">
                <FieldSelect
                  db={db}
                  value={draft.key}
                  fields={fieldKeys}
                  onChange={(key) => setDraft({ key, op: OPS[kindOf(db, key)][0], value: "" })}
                />
                <select value={draft.op} onChange={(e) => setDraft({ ...draft, op: e.target.value as Op })}>
                  {OPS[draftKind].map((o) => <option key={o} value={o}>{OP_LABEL[o]}</option>)}
                </select>
                {draftKind === "bool" ? (
                  <select value={draft.value} onChange={(e) => setDraft({ ...draft, value: e.target.value })}>
                    <option value="">…</option>
                    <option value="true">yes</option>
                    <option value="false">no</option>
                  </select>
                ) : (
                  <input
                    className="search"
                    placeholder={db.fields[draft.key]?.type === "date" ? "2026-01" : draftKind === "num" ? "100B, 128K, 3e-4" : "text"}
                    value={draft.value}
                    onChange={(e) => setDraft({ ...draft, value: e.target.value })}
                    onKeyDown={(e) => e.key === "Enter" && addDraft()}
                  />
                )}
                <button className="chip-btn" disabled={!draftValid} onClick={addDraft}>Add</button>
              </div>
              <p className="hint">Models without a value for the field are excluded by ≥, ≤, = and contains.</p>
            </fieldset>
          </div>
        )}
      </div>
      {rules.map((r, i) => (
        <span className="filter-chip" key={i}>
          {label(r)}
          <button aria-label="Remove filter" onClick={() => save(rules.filter((_, j) => j !== i))}>×</button>
        </span>
      ))}
    </>
  );
}
