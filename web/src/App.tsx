import { Component, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { KV_STORAGE_LABELS, PRESETS, PRESET_ALIASES, PRESET_NOTES, numericFields, type DB } from "./data";
import { context as fmtContext } from "./format";
import { Table, visibleColumns } from "./Table";
import { Plot, PARETO } from "./Plot";
import { Breakdown } from "./Breakdown";
import { FEATURES, GROUPS } from "./features";
import { BREAKDOWN_PRESETS, PLOT_PRESETS, active } from "./presets";
import { Panel } from "./Panel";
import { ColumnPicker } from "./ColumnPicker";
import { FieldSelect } from "./FieldSelect";
import { Filters } from "./Filters";
import { ATTRS, parseRules, scopeModels, serializeRules } from "./filterRules";
import { DEFAULTS, parseView, serializeView, type View } from "./view";
import { Share } from "./Share";
export type { View } from "./view";

const CONTEXTS = [8192, 32768, 131072, 262144, 1048576];

function readHash(db: DB): View {
  const view = parseView(location.hash.slice(1));
  view.preset = PRESET_ALIASES[view.preset] ?? view.preset;
  if (!Number.isSafeInteger(view.ctx) || view.ctx <= 0) view.ctx = DEFAULTS.ctx;
  if (!Object.hasOwn(KV_STORAGE_LABELS, view.kvdt)) view.kvdt = DEFAULTS.kvdt;
  const known = (key: string) => Object.hasOwn(db.fields, key);
  view.cols = view.cols.split(",").filter(known).join(",");
  view.f = serializeRules(parseRules(view.f).filter((r) => known(r.key) || Object.hasOwn(ATTRS, r.key)));
  if (!known(view.focus)) view.focus = "";
  const numeric = new Set(numericFields(db));
  if (!numeric.has(view.x)) view.x = DEFAULTS.x;
  if (!numeric.has(view.y)) view.y = DEFAULTS.y;
  if (view.sort !== "name" && !known(view.sort)) {
    view.sort = visibleColumns(db, view).includes(DEFAULTS.sort) ? DEFAULTS.sort : "name";
    view.dir = view.sort === "name" ? 1 : DEFAULTS.dir;
  }
  return view;
}

function writeHash(view: View) {
  const hash = serializeView(view);
  history.replaceState(null, "", hash ? `#${hash}` : location.pathname);
}

export function App({ db }: { db: DB }) {
  const [view, setView] = useState<View>(() => readHash(db));
  const set = (patch: Partial<View>) => setView((v) => ({ ...v, ...patch }));
  // Remember sorting for each column set.
  const sorts = useRef<Record<string, Pick<View, "sort" | "dir">>>({});
  const changeColumns = (preset: string) => {
    sorts.current[view.preset] = { sort: view.sort, dir: view.dir };
    const columns = visibleColumns(db, { ...view, preset });
    const kept = sorts.current[preset];
    if (kept && (kept.sort === "name" || columns.includes(kept.sort))) {
      set({ preset, ...kept });
    } else if (view.sort === "name" || columns.includes(view.sort)) {
      set({ preset });
    } else {
      const sort = columns.includes(DEFAULTS.sort) ? DEFAULTS.sort : "name";
      set({ preset, sort, dir: sort === "name" ? 1 : DEFAULTS.dir });
    }
  };
  useEffect(() => {
    const update = () => setView(readHash(db));
    addEventListener("hashchange", update);
    return () => removeEventListener("hashchange", update);
  }, [db]);
  useEffect(() => writeHash(view), [view]);
  const numeric = useMemo(() => numericFields(db), [db]);
  const model = db.models.find((m) => m.id === view.model);
  const scoped = useMemo(() => scopeModels(db, view), [db, view]);
  const selected = view.sel.split(",").filter(Boolean);
  const chartPresets = view.tab === "plot" ? PLOT_PRESETS : BREAKDOWN_PRESETS;
  const chartPreset = active(chartPresets, view);
  // Axis controls stay folded while a preset is shown, unless opened or the view no longer matches a preset.
  const [customize, setCustomize] = useState(false);
  const showAxes = customize || !chartPreset;
  const kvShown =
    view.tab === "plot"
      ? [view.x, view.y].some((k) => k.startsWith("kv."))
      : view.tab === "table" && visibleColumns(db, view).some((k) => k.startsWith("kv."));

  return (
    <>
      <div className="controls">
      <div className="toolbar">
        <div className="tabs" role="tablist">
          {(["table", "plot", "breakdown"] as const).map((t) => (
            <button key={t} role="tab" aria-selected={view.tab === t} onClick={() => set(
              // An untouched breakdown opens on the all-features overview.
              t === "breakdown" && view.by === DEFAULTS.by && view.feat === DEFAULTS.feat ? { tab: t, by: "features" } : { tab: t },
            )}>
              {{ table: "Table", plot: "Plot", breakdown: "Breakdown" }[t]}
            </button>
          ))}
        </div>

        {view.tab === "table" ? (
          <div className="chips" role="radiogroup" aria-label="Column set">
            {[...Object.keys(PRESETS), "All fields"].map((p) => (
              <button key={p} role="radio" aria-checked={view.preset === p} title={PRESET_NOTES[p]} onClick={() => changeColumns(p)}>
                {p}
              </button>
            ))}
            <ColumnPicker db={db} view={view} set={set} />
          </div>
        ) : (
          <PresetChips presets={chartPresets} current={chartPreset} set={set}>
            {chartPreset && (
              <button className="link customize" aria-expanded={showAxes} onClick={() => setCustomize(!customize)}>
                {showAxes ? "Hide axes" : "Customize"}
              </button>
            )}
          </PresetChips>
        )}
        <Share view={view} />
      </div>

      {view.tab === "plot" && showAxes && (
        <div className="toolbar sub">
          <div className="axes">
            <AxisPicker label="x" value={view.x} log={view.xlog} fields={numeric} db={db}
              onChange={(x, xlog) => set({ x, xlog })} />
            <AxisPicker label="y" value={view.y} log={view.ylog} fields={numeric} db={db}
              onChange={(y, ylog) => set({ y, ylog })} />
            <label className="toggle" title="Highlight models no other model beats on both axes">
              Pareto front
              <select value={view.pareto} onChange={(e) => set({ pareto: e.target.value })}>
                {Object.entries(PARETO).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </label>
          </div>
        </div>
      )}
      {view.tab === "breakdown" && showAxes && (
        <div className="toolbar sub">
          <div className="axes">
            {view.by !== "features" && <label className="axis-picker">
              <span className="axis-name">split</span>
              <select value={view.feat} onChange={(e) => set({ feat: e.target.value })}>
                {Object.entries(FEATURES).map(([k, f]) => <option key={k} value={k}>{f.label}</option>)}
              </select>
            </label>}
            <label className="axis-picker">
              <span className="axis-name">by</span>
              <select value={view.by} onChange={(e) => set({ by: e.target.value })}>
                {Object.entries(GROUPS).map(([k, g]) => <option key={k} value={k}>{g.label}</option>)}
              </select>
            </label>
          </div>
        </div>
      )}

      <div className="toolbar scope">
        <input
          className="search"
          type="search"
          placeholder="Models: deepseek, qwen3.5"
          value={view.q}
          onChange={(e) => set({ q: e.target.value })}
        />
        <Filters db={db} view={view} set={set} />
        {(selected.length > 0 || view.only) && (
          <span className="selection">
            <label className="toggle">
              <input type="checkbox" checked={view.only} onChange={(e) => set({ only: e.target.checked })} />
              Only selected ({selected.length})
            </label>
            <button className="link" onClick={() => set({ sel: "", only: false })}>clear</button>
          </span>
        )}
        <span className="count-note">
          {scoped.length === db.models.length ? `${db.models.length} models` : `${scoped.length} of ${db.models.length} models`}
          {view.tab === "table" && view.sort !== "name" && !visibleColumns(db, view).includes(view.sort) &&
            ` · sorted by ${db.fields[view.sort]?.label ?? view.sort} ${view.dir === 1 ? "↑" : "↓"}`}
        </span>
        {view.tab === "breakdown" && (
          <label className="toggle" title="Scale each bar to the models whose sources state the property">
            <input type="checkbox" checked={view.disc} onChange={(e) => set({ disc: e.target.checked })} />
            Disclosed only
          </label>
        )}
        {view.tab === "table" && (view.side ? (
          <button className="chip-btn" onClick={() => set({ side: false })}>← Back to table</button>
        ) : (
          <button
            className="chip-btn"
            disabled={selected.length < 2}
            title={selected.length < 2 ? "Tick two or more models in the table to compare them" : "Fields as rows, selected models as columns"}
            onClick={() => set({ side: true })}
          >
            Compare selected{selected.length ? ` (${selected.length})` : ""}
          </button>
        ))}
        {kvShown && (
          <span className="context">
            KV cache
            <select value={view.kvdt} onChange={(e) => set({ kvdt: e.target.value })} aria-label="KV-cache dtype"
              title="Uniform precision counts payload only. Sourced storage uses each component's documented format, scales and padding.">
              {Object.entries(KV_STORAGE_LABELS).map(([d, label]) => <option key={d} value={d}>{label}</option>)}
            </select>
            at
            <select value={view.ctx} onChange={(e) => set({ ctx: Number(e.target.value) })} aria-label="Context length">
              {CONTEXTS.map((c) => <option key={c} value={c}>{fmtContext(c)}</option>)}
            </select>
            <span className="kv-note">
              {view.kvdt === "sourced"
                ? `With scales and padding; complete for ${scoped.filter((m) => m.kv?.sourced).length} of ${scoped.length} models.`
                : "Payload only."}
              {" "}Per-token values include bounded SWA caches; global KV excludes them.
            </span>
          </span>
        )}
      </div>
      </div>

      <main>
        {view.tab === "table" ? (
          <Table db={db} models={scoped} view={view} set={set} />
        ) : (
          view.tab === "plot" ? (
            <Plot db={db} models={scoped} view={view} open={(id) => set({ model: id, focus: view.y })} />
          ) : (
            <Breakdown models={scoped} view={view} open={(id) => set({ model: id, focus: "" })} />
          )
        )}
      </main>

      {model && (
        <Boundary key={model.id} onClose={() => set({ model: "", focus: "" })}>
          <Panel db={db} model={model} view={view} onClose={() => set({ model: "", focus: "" })} />
        </Boundary>
      )}
    </>
  );
}

function AxisPicker(props: {
  label: string; value: string; log: boolean; fields: string[]; db: DB; onChange: (k: string, log: boolean) => void;
}) {
  return (
    <span className="axis-picker">
      <span className="axis-name">{props.label}</span>
      <FieldSelect db={props.db} value={props.value} fields={props.fields} onChange={(k) => props.onChange(k, props.log)} />
      <label>
        <input type="checkbox" checked={props.log} onChange={(e) => props.onChange(props.value, e.target.checked)} />
        log
      </label>
    </span>
  );
}

function PresetChips({ presets, current, set, children }: {
  presets: Record<string, Partial<View>>; current: string | undefined; set: (p: Partial<View>) => void; children?: ReactNode;
}) {
  return (
    <div className="chips" role="radiogroup" aria-label="Preset views">
      {Object.keys(presets).map((name) => (
        <button key={name} role="radio" aria-checked={current === name} onClick={() => set(presets[name])}>
          {name}
        </button>
      ))}
      {children}
    </div>
  );
}

/** Keeps a rendering error in the source panel from blanking the page. */
class Boundary extends Component<{ children: ReactNode; onClose: () => void }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <aside className="panel" role="alert">
        <button className="close" onClick={this.props.onClose} aria-label="Close">×</button>
        <p>This record could not be displayed. The raw values are in facts.csv.</p>
      </aside>
    );
  }
}
