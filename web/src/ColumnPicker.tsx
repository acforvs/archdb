import { useEffect, useRef, useState } from "react";
import { fieldText, type DB } from "./data";
import type { View } from "./App";
import { visibleColumns } from "./Table";

/** Searchable field list; picking fields switches the table to a custom column set. */
export function ColumnPicker({ db, view, set }: { db: DB; view: View; set: (p: Partial<View>) => void }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  const current = visibleColumns(db, view);

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

  const toggle = (k: string) => {
    const next = current.includes(k) ? current.filter((c) => c !== k) : [...current, k];
    set({ preset: "Custom", cols: next.join(",") });
  };
  const setGroup = (keys: string[], on: boolean) => {
    const next = on ? [...current, ...keys.filter((k) => !current.includes(k))] : current.filter((c) => !keys.includes(c));
    set({ preset: "Custom", cols: next.join(",") });
  };
  const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
  const match = (k: string) => terms.every((t) => fieldText(k, db.fields[k]).includes(t));

  return (
    <div className="picker right" ref={ref}>
      <button role="radio" aria-checked={view.preset === "Custom"} onClick={() => setOpen(!open)}>
        Custom{view.preset === "Custom" ? ` (${current.length})` : ""} ▾
      </button>
      {open && (
        <div className="picker-pop">
          <input autoFocus type="search" placeholder="Search fields" value={q} onChange={(e) => setQ(e.target.value)} />
          <div className="picker-list">
            {Object.entries(db.groups).map(([g, title]) => {
              const keys = Object.keys(db.fields).filter((k) => k.startsWith(g + ".") && match(k));
              if (!keys.length) return null;
              const on = keys.filter((k) => current.includes(k)).length;
              return (
                <div key={g}>
                  <label className="picker-group">
                    <input
                      type="checkbox"
                      checked={on === keys.length}
                      ref={(el) => {
                        if (el) el.indeterminate = on > 0 && on < keys.length;
                      }}
                      onChange={() => setGroup(keys, on < keys.length)}
                    />
                    {title}
                    <span className="count">{on}/{keys.length}</span>
                  </label>
                  {keys.map((k) => (
                    <label key={k} title={db.fields[k].definition}>
                      <input type="checkbox" checked={current.includes(k)} onChange={() => toggle(k)} />
                      {db.fields[k].label}
                      <code>{k}</code>
                    </label>
                  ))}
                </div>
              );
            })}
          </div>
          <div className="picker-foot">
            <button className="link" onClick={() => set({ preset: "Custom", cols: "" })}>Clear</button>
            <span>{current.length} columns</span>
          </div>
        </div>
      )}
    </div>
  );
}
