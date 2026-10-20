import { useEffect, useMemo, useRef, useState } from "react";
import { fieldText, type DB } from "./data";

/** Searchable, grouped field chooser (key, label, definition and aliases are searchable). */
export function FieldSelect({ db, value, fields, onChange, placeholder = "Choose a field" }: {
  db: DB; value: string; fields: string[]; onChange: (k: string) => void; placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const matches = useMemo(() => {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    return fields.filter((k) => terms.every((t) => fieldText(k, db.fields[k]).includes(t)));
  }, [q, fields, db]);

  useEffect(() => setActive(0), [q]);
  useEffect(() => {
    listRef.current?.querySelector(".active")?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const pick = (k: string) => {
    onChange(k);
    setOpen(false);
    setQ("");
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") setActive((a) => Math.min(a + 1, matches.length - 1));
    else if (e.key === "ArrowUp") setActive((a) => Math.max(a - 1, 0));
    else if (e.key === "Enter" && matches[active]) pick(matches[active]);
    else if (e.key === "Escape") setOpen(false);
    else return;
    e.preventDefault();
  };

  let prev = "";
  return (
    <div className="fselect" ref={ref}>
      <button className="fselect-btn" onClick={() => setOpen(!open)} aria-haspopup="listbox" aria-expanded={open}>
        {value ? db.fields[value]?.label ?? value : placeholder} <span aria-hidden>▾</span>
      </button>
      {open && (
        <div className="picker-pop">
          <input autoFocus type="search" placeholder="Search fields" value={q} onChange={(e) => setQ(e.target.value)}
            onKeyDown={onKey} />
          <div className="picker-list" role="listbox" ref={listRef}>
            {matches.map((k, i) => {
              const g = k.split(".")[0];
              const head = g !== prev ? (prev = g) : null;
              return (
                <div key={k}>
                  {head && <div className="picker-group">{db.groups[g]}</div>}
                  <div
                    role="option"
                    aria-selected={k === value}
                    className={`option${i === active ? " active" : ""}`}
                    title={db.fields[k].definition}
                    onMouseEnter={() => setActive(i)}
                    onClick={() => pick(k)}
                  >
                    {db.fields[k].label}
                    <code>{k}</code>
                  </div>
                </div>
              );
            })}
            {!matches.length && <div className="picker-empty">No field matches “{q}”.</div>}
          </div>
        </div>
      )}
    </div>
  );
}
