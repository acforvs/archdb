import { useMemo, useState } from "react";
import { KV_CONTEXT_KEYS, PRESETS, cell, numeric, siblingScore, type CellOpts, type DB, type Model } from "./data";
import { Compare } from "./Compare";

export function visibleColumns(db: DB, view: View): string[] {
  if (view.preset === "All fields") return Object.keys(db.fields);
  if (view.preset === "Custom") return view.cols.split(",").filter((k) => k in db.fields);
  return PRESETS[view.preset] ?? PRESETS.Overview;
}
import { context as fmtContext, format } from "./format";
import type { View } from "./App";

/** Sort key: numbers (dates as years, booleans as 0/1), lower-cased text, or null for missing values. */
function sortKey(db: DB, m: Model, k: string, o: CellOpts): number | string | null {
  if (k === "name") return m.name.toLowerCase();
  const v = cell(m, k, o)?.value;
  if (typeof v === "boolean") return Number(v);
  const n = db.fields[k] ? numeric(db.fields[k], v ?? null) : null;
  if (n !== null) return n;
  return typeof v === "string" ? v.toLowerCase() : null;
}

export function Table({ db, models, view, set }: {
  db: DB; models: Model[]; view: View; set: (p: Partial<View>) => void;
}) {
  const cols = visibleColumns(db, view);
  const [hover, setHover] = useState<{ m: Model; k: string; x: number; y: number } | null>(null);
  const selected = new Set(view.sel.split(",").filter(Boolean));
  const toggle = (id: string) => {
    const next = selected.has(id) ? [...selected].filter((s) => s !== id) : [...selected, id];
    set({ sel: next.join(",") });
  };

  const hasAny = (m: Model) => cols.length === 0 || cols.some((k) => cell(m, k, view)?.value != null);
  const rows = useMemo(
    () =>
      models
        .filter((m) => view.all || hasAny(m))
        .map((m) => [m, sortKey(db, m, view.sort, view)] as const)
        .sort(([, a], [, b]) =>
          a === null ? (b === null ? 0 : 1) : b === null ? -1 : (a < b ? -1 : a > b ? 1 : 0) * view.dir,
        )
        .map(([m]) => m),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [db, models, view.sort, view.dir, view.ctx, view.kvdt, view.all, cols.join()],
  );
  if (view.side) {
    const picked = view.only || selected.size === 0 ? rows : rows.filter((m) => selected.has(m.id));
    return <Compare db={db} models={picked} cols={cols} view={view} set={set} />;
  }
  const hidden = view.all ? 0 : models.filter((m) => !hasAny(m)).length;
  if (!rows.length)
    return (
      <div className="table-card empty-state">
        No models match.{" "}
        <button className="link" onClick={() => set({ q: "", f: "", only: false, all: true })}>Clear search and filters</button>
      </div>
    );

  const groups: { g: string; span: number }[] = [];
  for (const k of cols) {
    const g = k.split(".")[0];
    if (groups.at(-1)?.g === g) groups.at(-1)!.span++;
    else groups.push({ g, span: 1 });
  }
  const groupStart = new Set(cols.filter((k, i) => i > 0 && k.split(".")[0] !== cols[i - 1].split(".")[0]));
  const sortBy = (k: string) =>
    set({ sort: k, dir: view.sort === k ? (view.dir === 1 ? -1 : 1) : k === "name" ? 1 : -1 });
  const numericCol = (k: string) => !["str", "bool"].includes(db.fields[k].type);

  return (
    <div className="table-card">
      <div className="table-scroll">
        <table className="grid">
          <thead>
            <tr className="group-row">
              <th className="sticky" colSpan={1} />
              {groups.map(({ g, span }, i) => (
                <th key={g + i} colSpan={span} className={i > 0 ? "group-start" : ""}>{db.groups[g]}</th>
              ))}
            </tr>
            <tr>
              <th className="sticky sortable" aria-sort={ariaSort(view, "name")} onClick={() => sortBy("name")}>
                Model{arrow(view, "name")}
              </th>
              {cols.map((k) => (
                <th
                  key={k}
                  className={`sortable ${numericCol(k) ? "num" : ""} ${groupStart.has(k) ? "group-start" : ""}`}
                  aria-sort={ariaSort(view, k)}
                  title={db.fields[k].definition}
                  tabIndex={0}
                  onClick={() => sortBy(k)}
                  onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), sortBy(k))}
                >
                  {db.fields[k].label}
                  {db.fields[k].derived && <sup className="mark" aria-label="computed">ƒ</sup>}
                  {arrow(view, k)}
                  {k.startsWith("kv.") && (
                    <span className="sub">
                      {view.kvdt.toUpperCase()}
                      {KV_CONTEXT_KEYS.includes(k) ? ` · ${fmtContext(view.ctx)}` : " · long context"}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr key={m.id}>
                <th className={`sticky model${selected.has(m.id) ? " picked" : ""}`} scope="row">
                  <div className="model-cell">
                    <input
                      type="checkbox"
                      className="pick"
                      checked={selected.has(m.id)}
                      onChange={() => toggle(m.id)}
                      aria-label={`Select ${m.name}`}
                    />
                    <div>
                      <button onClick={() => set({ model: m.id, focus: "" })}>{m.name}</button>
                      <span>
                        {m.org} · {countryCode(m.country)}
                        {m.facts["meta.released"] && !cols.includes("meta.released") &&
                          ` · ${String(m.facts["meta.released"].value).slice(0, 7)}`}
                        {m.weights === "closed" && <b className="tag warn">closed weights</b>}
                      </span>
                    </div>
                  </div>
                </th>
                {cols.map((k) => {
                  const c = cell(m, k, view);
                  const v = c?.value;
                  const empty = v === null || v === undefined;
                  const unmodeled = empty && c?.kind === "derived";
                  const undisclosed = c?.kind === "undisclosed";
                  const cls = [
                    unmodeled ? "unmodeled" : undisclosed ? "undisclosed" : empty ? "empty" : "value",
                    numericCol(k) ? "num" : "",
                    groupStart.has(k) ? "group-start" : "",
                    typeof v === "string" && v.length > 28 ? "wide" : "",
                  ].join(" ");
                  return (
                    <td
                      key={k}
                      className={cls}
                      onClick={() => !empty && set({ model: m.id, focus: k })}
                      title={unmodeled ? c.formula : undisclosed ? "Not disclosed in the publisher's sources" : undefined}
                      onMouseEnter={(e) => !empty && setHover({ m, k, x: e.clientX, y: e.clientY })}
                      onMouseLeave={() => setHover(null)}
                    >
                      {unmodeled ? "not modeled" : undisclosed ? "n/d" : format(k, db.fields[k], v)}
                      {!empty && siblingScore(m, k) && (
                        <sup className="mark" aria-label="post-trained sibling">‡</sup>
                      )}
                      {!empty && c?.kind === "fact" && c.basis === "inferred" && (
                        <sup className="mark" aria-label="inferred">†</sup>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="footnote">
        Click a model or a value for its sources; tick models to compare them. <sup className="mark">ƒ</sup> computed
        from other values · <sup className="mark">†</sup> inferred by the curator · <sup className="mark">‡</sup> score of
        the post-trained sibling of a base checkpoint · n/d not disclosed by the publisher · – not applicable or not
        recorded.
        {hidden > 0 && (
          <>
            {" "}
            {hidden} {hidden === 1 ? "model discloses" : "models disclose"} none of these values and{" "}
            {hidden === 1 ? "is" : "are"} hidden.{" "}
            <button className="link" onClick={() => set({ all: true })}>Show all</button>
          </>
        )}
        {view.all && (
          <>
            {" "}
            <button className="link" onClick={() => set({ all: false })}>Hide models without values</button>
          </>
        )}
      </p>
      {hover && <Preview db={db} {...hover} o={view} />}
    </div>
  );
}

function Preview({ db, m, k, x, y, o }: { db: DB; m: Model; k: string; x: number; y: number; o: CellOpts }) {
  const c = cell(m, k, o);
  if (!c || c.kind === "undisclosed") return null;
  const left = Math.min(x + 14, window.innerWidth - 360);
  return (
    <div className="preview" style={{ left, top: y + 16 }} role="tooltip">
      <div className="preview-head">
        {db.fields[k].label} · <b>{format(k, db.fields[k], c.value, true)}</b>
      </div>
      {c.kind === "derived" ? (
        <div className="preview-src">computed: {c.formula}</div>
      ) : (
        <>
          {c.quote && <blockquote>{c.quote}</blockquote>}
          {c.note && <div className="preview-note">{c.note}</div>}
          {siblingScore(m, k) && (
            <div className="preview-note">Measured on the post-trained sibling of this base checkpoint.</div>
          )}
          <div className="preview-src">
            {c.source} · <code>{c.at}</code>
          </div>
        </>
      )}
    </div>
  );
}

function arrow(view: View, k: string) {
  return view.sort === k ? <span className="arrow">{view.dir === 1 ? "↑" : "↓"}</span> : null;
}
function ariaSort(view: View, k: string) {
  return view.sort === k ? (view.dir === 1 ? "ascending" : "descending") : undefined;
}


const CODES: Record<string, string> = {
  "United States": "US", China: "CN", "South Korea": "KR", France: "FR", Canada: "CA", Switzerland: "CH",
  "United Arab Emirates": "AE", "United Kingdom": "UK", Japan: "JP", Germany: "DE", India: "IN",
};

/** Short country label for row subtitles; the full name stays in filters and the source panel. */
export function countryCode(country: string): string {
  return CODES[country] ?? country;
}
