import { useEffect, useMemo, useState } from "react";
import type { Model } from "./data";
import { FEATURES, GROUPS, OVERVIEW } from "./features";
import type { View } from "./App";

const ND = "not disclosed";
/** Leading periods with fewer models than this are merged into one row. */
const MIN_PERIOD = 3;

interface Segment {
  cat: string;
  models: Model[];
}
interface Row {
  group: string;
  segments: Segment[];
  n: number;
  disclosed: number;
}

/** 100% stacked bars: how a categorical property splits within each group of models. */
export function Breakdown({ models, view, open }: { models: Model[]; view: View; open: (id: string) => void }) {
  const feature = FEATURES[view.feat] ?? FEATURES.swa;
  const group = GROUPS[view.by] ?? GROUPS.country;
  const [hot, setHot] = useState<{ row: string; seg: Segment; x: number; y: number } | null>(null);
  useEffect(() => setHot(null), [models, view.by, view.feat, view.disc]);

  const overview = view.by === "features";
  const { rows, cats } = useMemo(() => {
    if (overview) {
      const rows: Row[] = OVERVIEW.map(({ feat, label, yes }) => {
        const of = (m: Model) => {
          const c = FEATURES[feat].of(m);
          return c === null ? ND : yes.includes(c) ? "yes" : "no";
        };
        const segments = ["yes", "no", ...(view.disc ? [] : [ND])]
          .map((cat) => ({ cat, models: models.filter((m) => of(m) === cat) }))
          .filter((s) => s.models.length);
        return { group: label, segments, n: models.length, disclosed: models.filter((m) => of(m) !== ND).length };
      }).filter((r) => r.segments.length);
      return { rows, cats: ["yes", "no"] };
    }
    const byGroup = new Map<string, Model[]>();
    for (const m of models) {
      const g = group.of(m);
      if (g) byGroup.set(g, [...(byGroup.get(g) ?? []), m]);
    }
    const seen = new Set<string>();
    for (const m of models) {
      const c = feature.of(m);
      if (c) seen.add(c);
    }
    const cats = [...feature.order.filter((c) => seen.has(c)), ...[...seen].filter((c) => !feature.order.includes(c)).sort()];
    const timeline = view.by === "year" || view.by === "half";
    let ordered = [...byGroup.entries()].sort(([a, ma], [b, mb]) =>
      timeline ? a.localeCompare(b) : mb.length - ma.length || a.localeCompare(b),
    );
    // Fold sparse leading periods into one row so a single model does not make a whole bar.
    if (timeline) {
      let k = 0, n = 0;
      while (k < ordered.length - 1 && n < MIN_PERIOD) n += ordered[k++][1].length;
      if (k > 1) ordered = [[`≤ ${ordered[k - 1][0]}`, ordered.slice(0, k).flatMap(([, ms]) => ms)], ...ordered.slice(k)];
    }
    const rows: Row[] = ordered.map(([g, ms]) => {
      const segments = [...cats, ...(view.disc ? [] : [ND])]
        .map((cat) => ({ cat, models: ms.filter((m) => (feature.of(m) ?? ND) === cat) }))
        .filter((s) => s.models.length);
      return { group: g, segments, n: ms.length, disclosed: ms.filter((m) => feature.of(m) !== null).length };
    }).filter((r) => r.segments.length);
    return { rows, cats };
  }, [models, feature, group, view.by, view.disc, overview]);

  // Two-way splits read as "has it or not": the accent for the first category, a pale tint for the second.
  const order = overview ? ["yes", "no"] : feature.order;
  const binary = cats.length <= 2 && cats.every((c) => order.slice(0, 2).includes(c)) && order.length === 2;
  const color = (cat: string) =>
    cat === ND ? undefined : binary && cat === order[1] ? "var(--series-off)" : `var(--series-${(cats.indexOf(cat) % 6) + 1})`;
  const segClass = (cat: string) => `seg${cat === ND ? " nd" : binary && cat === order[1] ? " off" : ""}`;
  const pct = (k: number, n: number) => `${Math.round((100 * k) / n)}%`;
  const hotRow = hot && rows.find((r) => r.group === hot.row);

  return (
    <div className={`plot-card breakdown${overview ? " overview" : ""}`}>
      <div className="legend">
        <b>{overview ? "Share of models with each feature" : feature.label}</b>
        {[...cats, ...(view.disc ? [] : [ND])].map((c) => (
          <span key={c}>
            <i className={segClass(c)} style={{ background: color(c) }} />
            {c}
          </span>
        ))}
      </div>
      <div className="bars" onMouseLeave={() => setHot(null)}>
        {rows.map((r) => (
          <div className="bar-row" key={r.group}>
            <span className="bar-label">{r.group}</span>
            <div className="bar">
              {r.segments.map((s) => (
                <button
                  key={s.cat}
                  className={segClass(s.cat)}
                  style={{ flexGrow: s.models.length, background: color(s.cat) }}
                  aria-label={`${r.group}, ${s.cat}: ${s.models.length} of ${r.n}`}
                  onMouseEnter={(e) => setHot({ row: r.group, seg: s, x: e.clientX, y: e.clientY })}
                  onMouseMove={(e) => setHot({ row: r.group, seg: s, x: e.clientX, y: e.clientY })}
                  onClick={() => s.models.length === 1 && open(s.models[0].id)}
                >
                  {s.models.length}
                </button>
              ))}
            </div>
            <span className="bar-n">{view.disc ? r.disclosed : r.n}</span>
          </div>
        ))}
        {!rows.length && <p className="empty-state">No models match the filters.</p>}
      </div>
      {hot && hotRow && (
        <div className="preview" style={{ left: Math.min(hot.x + 14, innerWidth - 300), top: hot.y + 16, position: "fixed" }}>
          <div className="preview-head">
            {hot.row} · <b>{hot.seg.cat}</b> · {hot.seg.models.length}
            {hot.seg.cat !== ND &&
              ` (${pct(hot.seg.models.length, hotRow.disclosed)} of disclosed)`}
          </div>
          <div className="preview-src">{hot.seg.models.map((m) => m.name).join(", ")}</div>
        </div>
      )}
      <p className="footnote">
        Numbers are model counts; the right column is the row total{view.disc ? " of models that disclose it" : ""}.
        {!view.disc && " Hatched: not disclosed."} Hover a segment for its models.
        {overview && " Change “by” to split one feature by country, organization or release date."}
        {(view.by === "year" || view.by === "half") &&
          " archdb covers recent strong models, so earlier periods hold fewer of them: compare shares, not counts."}
      </p>
    </div>
  );
}
