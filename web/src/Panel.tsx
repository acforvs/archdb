import { Fragment, useEffect, useRef, useState } from "react";
import { cell, hfView, modelLinks, type CellOpts, type DB, type Fact, type Model, type Stage } from "./data";
import { format, lr, si } from "./format";
import { GLOSSARY, terms } from "./glossary";
import { LayerStrip, LayerTypeLegend, layerStack } from "./LayerStack";
import type { View } from "./App";

/** Key specs at the top of the panel; each value opens its own evidence. */
const SPECS: { label: string; parts: [string, string][] }[] = [
  { label: "Inputs", parts: [["meta.modality", ""]] },
  { label: "Parameters", parts: [["params.total", " total"], ["params.active", " active"], ["params.active_prefill", " active (prefill)"]] },
  { label: "Backbone non-emb.", parts: [["params.counted_non_embedding", " total"], ["params.counted_active_non_embedding", " active"]] },
  { label: "Largest layer", parts: [["params.counted_max_layer", ""]] },
  { label: "Layers × d_model", parts: [["shape.layers", ""], ["shape.d_model", ""]] },
  { label: "Layout", parts: [["attn.layout", ""]] },
  { label: "Attention", parts: [["attn.heads", " Q"], ["attn.kv_heads", " KV"], ["attn.head_dim", " head dim"]] },
  { label: "MLA", parts: [["attn.kv_lora_rank", " KV latent"], ["attn.q_lora_rank", " Q latent"], ["attn.rope_dim", " RoPE dim"]] },
  { label: "SWA", parts: [["attn.local_heads", " Q"], ["attn.local_kv_heads", " KV"], ["attn.window", " window"]] },
  { label: "Linear mixer", parts: [["attn.linear_heads", " heads"], ["attn.linear_head_dim", " head dim"]] },
  { label: "MoE", parts: [["moe.experts", " routed"], ["moe.top_k", " active"], ["moe.shared", " shared"], ["moe.expert_dim", " dim"]] },
  { label: "Dense FFN", parts: [["ffn.dim", ""]] },
  { label: "MTP", parts: [["mtp.depth", " depth"]] },
  { label: "Vocab · context", parts: [["shape.vocab", " vocab"], ["ctx.final", " context"]] },
  { label: "Training", parts: [["train.tokens", " tokens"], ["train.optimizer", ""], ["train.precision", ""]] },
  { label: "Hardware", parts: [["train.accelerators", ""], ["train.parallelism", ""]] },
];

const SHORT = ["train.optimizer", "train.precision"];

export function Panel({ db, model, view, onClose }: { db: DB; model: Model; view: View; onClose: () => void }) {
  const ref = useRef<HTMLElement>(null);
  const [quotes, setQuotes] = useState(() => {
    try { return localStorage.getItem("archdb-quotes") === "1"; } catch { return false; }
  });
  const toggleQuotes = () => {
    setQuotes(!quotes);
    try { localStorage.setItem("archdb-quotes", quotes ? "0" : "1"); } catch { /* storage unavailable */ }
  };
  const [focus, setFocus] = useState(view.focus);
  const [focusRequest, setFocusRequest] = useState(0);
  const showEvidence = (key: string) => {
    setFocus(key);
    setFocusRequest((n) => n + 1);
  };
  useEffect(() => setFocus(view.focus), [model.id, view.focus]);

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
    ref.current?.scrollTo({ top: 0 });
  }, [model.id]);

  useEffect(() => {
    const panel = ref.current;
    if (!panel || !focus) return;
    const frame = requestAnimationFrame(() => {
      const target = panel.querySelector<HTMLElement>(".fact.focus")?.getBoundingClientRect();
      if (!target) return;
      const bounds = panel.getBoundingClientRect();
      if (target.top >= bounds.top && target.bottom <= bounds.bottom) return;
      panel.scrollTop += target.top - bounds.top - Math.max(0, (panel.clientHeight - target.height) / 2);
    });
    return () => cancelAnimationFrame(frame);
  }, [model.id, focus, focusRequest]);

  const groups = Object.keys(db.groups)
    .map((g) => ({
      g,
      // Computed values this record cannot support (inputs missing) are left out rather than listed as "–".
      keys: Object.keys(db.fields).filter((k) => {
        const c = cell(model, k, view);
        return k.startsWith(g + ".") && c !== undefined && !(c.kind === "derived" && c.value === null);
      }),
    }))
    .filter(({ keys }) => keys.length);
  const links = modelLinks(model);
  const stack = layerStack(model);
  // The layer figure replaces the layout row when the layout can be drawn.
  const specs = SPECS.filter(({ label }) => !(label === "Layout" && stack)).map(({ label, parts }) => {
    const shown = parts.flatMap(([k, suffix]) => {
      const v = cell(model, k, view)?.value;
      if (v === null || v === undefined) return [];
      // Recipe strings carry their qualifiers in the evidence below; the summary keeps the leading term.
      const text = k === "mtp.depth" && v === 0 ? "none"
        : SHORT.includes(k) && typeof v === "string" ? v.split(/ \(|;|,/)[0]
        : format(k, db.fields[k], v, true);
      const unit = k === "mtp.depth" && v === 0 ? ""
        : k === "params.active" && model.facts["params.active_prefill"] ? " active (decode)" : suffix;
      return [{ k, text: text + unit }];
    });
    return { label, shown };
  }).filter(({ shown }) => shown.length);

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="panel" ref={ref} aria-label={`${model.name} sources`}>
        <button className="close" onClick={onClose} aria-label="Close">×</button>
        <header>
          <h2>{model.name}</h2>
          <div className="panel-meta">
            {model.org} · {model.country} · {model.kind} checkpoint
            {model.weights === "closed" && " · weights not public"}
          </div>
          <div className="panel-links">
            {links.hf && <a className="primary" href={links.hf} target="_blank" rel="noreferrer">Hugging Face ↗</a>}
            {links.report && <a href={links.report} target="_blank" rel="noreferrer">{links.reportLabel} ↗</a>}
            <a href={`/models/${model.id}/`} title="Permanent page for this model">Model page</a>
          </div>
        </header>
        {stack && (
          <figure className="stack-figure">
            <button className="stack-link" onClick={() => showEvidence("attn.layout")} aria-label="Show layer layout evidence">
            <LayerStrip stack={stack} unit={Math.max(4, Math.min(8, Math.floor(540 / stack.layers.length)))} height={30} />
            </button>
            <figcaption>
              <LayerTypeLegend stack={stack} />
              <span>
                {String(model.facts["attn.layout"]?.value ?? `${stack.layers.length} layers`)}
                {!stack.ordered && " · grouped by type; layer order not shown"}
              </span>
            </figcaption>
          </figure>
        )}
        <dl className="specs">
          {specs.map(({ label, shown }) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{shown.map((p, i) => (
                <Fragment key={p.k}>
                  {i > 0 && (label === "Layers × d_model" ? " × " : " · ")}
                  <button className="spec-value" onClick={() => showEvidence(p.k)}
                    title={`Source for ${db.fields[p.k].label}`} aria-label={`Show source for ${db.fields[p.k].label}`}>
                    {p.text}
                    {model.facts[p.k]?.basis === "inferred" && <sup className="mark" aria-label="inferred">†</sup>}
                    {db.fields[p.k].derived && <sup className="mark" aria-label="computed">ƒ</sup>}
                  </button>
                </Fragment>
              ))}</dd>
            </div>
          ))}
        </dl>
        {terms(String(model.facts["attn.kind"]?.value ?? "")).length > 0 && (
          <div className="glossary">
            {terms(String(model.facts["attn.kind"].value)).map((t) => (
              <p key={t}>
                <b>{t}</b>: {GLOSSARY[t].text}{" "}
                <a href={GLOSSARY[t].url} target="_blank" rel="noreferrer">paper ↗</a>
              </p>
            ))}
          </div>
        )}
        {model.note && <p className="panel-note">{model.note}</p>}
        <div className="facts-head">
          <h3>All values</h3>
          <nav className="jump" aria-label="Sections">
            {groups.map(({ g }) => (
              <button key={g} className="link" onClick={() =>
                ref.current?.querySelector(`[data-group="${g}"]`)?.scrollIntoView({ block: "start", behavior: "smooth" })}>
                {db.groups[g]}
              </button>
            ))}
          </nav>
          <label className="toggle">
            <input type="checkbox" checked={quotes} onChange={toggleQuotes} />
            Show sources
          </label>
        </div>
        {groups.map(({ g, keys }) => (
          <section key={g} data-group={g}>
            <h3>{db.groups[g]}</h3>
            {keys.map((k) => (
              <FactRow key={k} db={db} model={model} k={k} o={view} focus={k === focus} open={quotes}
                onPick={() => setFocus(focus === k ? "" : k)} />
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

function FactRow({ db, model, k, o, focus, open, onPick }: {
  db: DB; model: Model; k: string; o: CellOpts; focus: boolean; open: boolean; onPick: () => void;
}) {
  const c = cell(model, k, o)!;
  const field = db.fields[k];
  if (c.kind === "undisclosed")
    return (
      <div className={`fact undisclosed${focus ? " focus" : ""}`} data-key={k}>
        <div className="fact-head">
          <span className="fact-label" title={field.definition}>{field.label}</span>
          <span className="fact-value">not disclosed</span>
        </div>
      </div>
    );
  const list = Array.isArray(c.value);
  const shown = open || focus;
  return (
    <div className={`fact${focus ? " focus" : ""}${shown ? "" : " folded"}`} data-key={k}>
      <div className={`fact-head${typeof c.value === "string" && c.value.length > 28 ? " stacked" : ""}`}
        onClick={onPick} role="button" tabIndex={0} aria-expanded={shown}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onPick())}>
        <span className="fact-label" title={field.definition}>{field.label}</span>
        {Array.isArray(c.value) ? null : <span className="fact-value">{format(k, field, c.value, true)}</span>}
      </div>
      {list && (k === "attn.kv_groups" || k === "attn.kv_precision") && c.kind === "fact"
        ? <CacheEntries entries={c.value as Stage[]} model={model} parent={c} precision={k === "attn.kv_precision"} quotes={shown} />
        : list && <Stages stages={c.value as Stage[]} quotes={shown} />}
      {!shown ? null : c.kind === "derived" ? (
        <>
          <div className="fact-src">computed{c.formula_version && ` · ${c.formula_version}`} · {c.formula}</div>
          {k.startsWith("kv.") && (
            <div className="fact-src">
              {["attn.kv_groups", ...(o.kvdt === "sourced" ? ["attn.kv_precision"] : [])].filter((key) => model.facts[key]).map((key) => {
                const fact = model.facts[key];
                return <a key={key} href={hfView(model.sources[fact.source].url)} target="_blank" rel="noreferrer">{db.fields[key].label} · {fact.at} </a>;
              })}
            </div>
          )}
        </>
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

function CacheEntries({ entries, model, parent, precision, quotes }: { entries: Stage[]; model: Model; parent: Fact; precision: boolean; quotes: boolean }) {
  return (
    <div className="cache-entries">
      {entries.map((entry) => {
        const source = entry.source ?? parent.source;
        const bytes = Math.ceil((entry.elements ?? 0) * (entry.bits ?? 0) / 8)
          + (entry.scale_count ?? 0) * (entry.scale_bits ?? 0) / 8 + (entry.padding_bytes ?? 0);
        return (
          <div className="cache-entry" key={entry.name}>
            <b>{entry.name}</b>
            {precision ? (
              <>
                <div>{entry.format}</div>
                <div className="fact-note">
                  {entry.elements} × {entry.bits} bits
                  {entry.scale_count ? ` + ${entry.scale_count} × ${entry.scale_bits}-bit scales` : ""}
                  {entry.padding_bytes ? ` + ${entry.padding_bytes} padding byte(s)` : ""}
                  {` = ${bytes} bytes/entry`}
                </div>
              </>
            ) : (
              <div>{entry.layers} cache owners × {entry.elements} scalars/entry; {entry.window
                ? `keeps the last ${entry.window} entries`
                : `one entry per ${entry.tokens_per_entry ?? 1} token(s)`}.</div>
            )}
            {entry.note && <div className="fact-note">{entry.note}</div>}
            {quotes && <div className="fact-src">
              <a href={hfView(model.sources[source].url)} target="_blank" rel="noreferrer">{source}</a>
              {" · "}{entry.at ?? parent.at}
            </div>}
            {quotes && entry.quote && <blockquote>{entry.quote}</blockquote>}
          </div>
        );
      })}
    </div>
  );
}

function Stages({ stages, quotes }: { stages: Stage[]; quotes: boolean }) {
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
              {quotes && s.note && <div className="fact-note">{s.note}</div>}
              {quotes && s.quote && <blockquote>{s.quote}</blockquote>}
            </td>
            {cols.map((c) => <td key={c} className="num">{fmt(c, s[c])}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
