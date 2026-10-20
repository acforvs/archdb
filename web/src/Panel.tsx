import { useEffect, useRef } from "react";
import { cell, hfView, type CellOpts, type DB, type Model, type Stage } from "./data";
import { format, lr, si } from "./format";
import type { View } from "./App";

export function Panel({ db, model, view, onClose }: { db: DB; model: Model; view: View; onClose: () => void }) {
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    // Move focus into the panel and give it back to the opener on close.
    const opener = document.activeElement as HTMLElement | null;
    ref.current?.querySelector<HTMLElement>(".close")?.focus({ preventScroll: true });
    return () => opener?.focus({ preventScroll: true });
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    const panel = ref.current;
    if (!panel) return;
    const frame = requestAnimationFrame(() => {
      const focus = panel.querySelector<HTMLElement>(".fact.focus");
      if (!focus) {
        panel.scrollTop = 0;
        return;
      }
      const target = focus.getBoundingClientRect();
      const bounds = panel.getBoundingClientRect();
      panel.scrollTop += target.top - bounds.top - Math.max(0, (panel.clientHeight - target.height) / 2);
    });
    return () => cancelAnimationFrame(frame);
  }, [model.id, view.focus]);

  const groups = Object.keys(db.groups)
    .map((g) => ({
      g,
      keys: Object.keys(db.fields).filter((k) => k.startsWith(g + ".") && cell(model, k, view) !== undefined),
    }))
    .filter(({ keys }) => keys.length);
  const known = Object.keys(model.facts).length;

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="panel" ref={ref} aria-label={`${model.name} sources`}>
        <button className="close" onClick={onClose} aria-label="Close">×</button>
        <header>
          <h2>{model.name}</h2>
          <a className="model-link" href={`/models/${model.id}/`}>Link to this model</a>
          <div className="panel-meta">
            {model.org} · {model.country} · {model.kind} checkpoint
            {model.weights === "closed" && " · weights not public"} · {known} facts
            {model.not_disclosed.length > 0 && ` · ${model.not_disclosed.length} not disclosed`}
          </div>
          {model.note && <p className="panel-note">{model.note}</p>}
        </header>
        {groups.map(({ g, keys }) => (
          <section key={g}>
            <h3>{db.groups[g]}</h3>
            {keys.map((k) => (
              <FactRow key={k} db={db} model={model} k={k} o={view} focus={k === view.focus} />
            ))}
          </section>
        ))}
        <section>
          <h3>Sources</h3>
          {Object.entries(model.sources).map(([id, s]) => (
            <div className="source" key={id}>
              <span className="tag">{s.kind}</span> <b>{id}</b>
              <a href={hfView(s.url)} target="_blank" rel="noreferrer">{s.url.replace(/^https:\/\//, "")}</a>
            </div>
          ))}
        </section>
      </aside>
    </>
  );
}

function FactRow({ db, model, k, o, focus }: { db: DB; model: Model; k: string; o: CellOpts; focus: boolean }) {
  const c = cell(model, k, o)!;
  const field = db.fields[k];
  if (c.kind === "undisclosed")
    return (
      <div className={`fact undisclosed${focus ? " focus" : ""}`}>
        <div className="fact-head">
          <span className="fact-label" title={field.definition}>{field.label}</span>
          <span className="fact-value">not disclosed</span>
        </div>
      </div>
    );
  return (
    <div className={`fact${focus ? " focus" : ""}`}>
      <div className={`fact-head${typeof c.value === "string" && c.value.length > 28 ? " stacked" : ""}`}>
        <span className="fact-label" title={field.definition}>{field.label}</span>
        {Array.isArray(c.value) ? null : <span className="fact-value">{format(k, field, c.value, true)}</span>}
      </div>
      {Array.isArray(c.value) && <Stages stages={c.value} />}
      {c.kind === "derived" ? (
        <div className="fact-src">computed · {c.formula}</div>
      ) : (
        <>
          {c.quote && <blockquote>{c.quote}</blockquote>}
          <div className="fact-src">
            <a href={hfView(model.sources[c.source].url)} target="_blank" rel="noreferrer">{c.source}</a>
            <span> · </span>
            <code>{c.at}</code>
            {c.basis === "inferred" && <span className="tag warn">inferred</span>}
          </div>
          {c.note && <div className="fact-note">{c.note}</div>}
        </>
      )}
    </div>
  );
}

function Stages({ stages }: { stages: Stage[] }) {
  const cols = (["tokens", "seq_len", "lr", "batch_tokens", "batch_sequences", "steps"] as const).filter((c) =>
    stages.some((s) => s[c] !== undefined),
  );
  const fmt = (c: string, v: number | string | undefined) =>
    typeof v === "string" ? v :
    v === undefined ? "–" : c === "tokens" || c === "batch_tokens" ? si(v) : c === "lr" ? lr(v) : v.toLocaleString("en-US");
  return (
    <table className="stages">
      <thead>
        <tr>
          <th>stage</th>
          {cols.map((c) => <th key={c}>{c.replace("_", " ")}</th>)}
        </tr>
      </thead>
      <tbody>
        {stages.map((s, i) => (
          <tr key={i}>
            <td>
              {s.name}
              {s.note && <div className="fact-note">{s.note}</div>}
              {s.quote && <blockquote>{s.quote}</blockquote>}
            </td>
            {cols.map((c) => <td key={c} className="num">{fmt(c, s[c])}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
