import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { KV_CONTEXT_KEYS, cell, numeric, siblingScore, type DB, type Model } from "./data";
import { format, si } from "./format";
import type { View } from "./App";

const M = { l: 76, r: 28, t: 24, b: 52 };

/** Pareto directions: which way is better on x and on y. */
export const PARETO: Record<string, string> = {
  "": "off",
  "x-y+": "low x, high y",
  "x+y+": "high x, high y",
  "x-y-": "low x, low y",
  "x+y-": "high x, low y",
};

interface Point { m: Model; x: number; y: number }

/** Reference power laws drawn through the median model, keyed by "x|y". */
const GUIDES: Record<string, { exp: number; label: string; note: string }> = {
  "train.tokens|train.batch_tokens": {
    exp: 0.47,
    label: "B ∝ D^0.47 through the median model",
    note: "Grey line: B ∝ D^0.47 (Power Lines) through the median model; points above it use a larger batch for their token budget.",
  },
};



/** Points not dominated by any other point under the chosen directions, ordered along x. */
function paretoFront(pts: Point[], mode: string): Set<Point> {
  if (!mode) return new Set();
  const sx = mode[1] === "+" ? 1 : -1;
  const sy = mode[3] === "+" ? 1 : -1;
  const better = (a: Point, b: Point) =>
    sx * a.x >= sx * b.x && sy * a.y >= sy * b.y && (sx * a.x > sx * b.x || sy * a.y > sy * b.y);
  return new Set(pts.filter((p) => !pts.some((q) => better(q, p))));
}

export function Plot({ db, models, view, open }: {
  db: DB; models: Model[]; view: View; open: (id: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 900, h: 560 });
  const [hotId, setHotId] = useState<string | null>(null);

  useLayoutEffect(() => {
    const el = ref.current!;
    // Fit the plot below the controls so the x axis stays on screen.
    const fit = () => setSize({
      w: el.clientWidth,
      h: Math.max(340, Math.min(680, innerHeight - el.getBoundingClientRect().top - window.scrollY - 64)),
    });
    const obs = new ResizeObserver(fit);
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const isDate = (k: string) => db.fields[k].type === "date";
  const xlog = view.xlog && !isDate(view.x);
  const ylog = view.ylog && !isDate(view.y);
  const pts = useMemo(() => {
    const num = (m: Model, k: string) => numeric(db.fields[k], cell(m, k, view)?.value ?? null);
    return models
      .map((m) => ({ m, x: num(m, view.x), y: num(m, view.y) }))
      .filter((p): p is Point => p.x !== null && p.y !== null && (!xlog || p.x > 0) && (!ylog || p.y > 0));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [db, models, view.x, view.y, view.ctx, view.kvdt, xlog, ylog]);
  const hot = pts.find((p) => p.m.id === hotId) ?? null;
  const front = useMemo(() => paretoFront(pts, view.pareto), [pts, view.pareto]);
  const frontLine = [...front].sort((a, b) => a.x - b.x);

  const { w, h } = size;
  const sx = scale(pts.map((p) => p.x), xlog, M.l, w - M.r, db.fields[view.x].unit === "bytes", isDate(view.x));
  const sy = scale(pts.map((p) => p.y), ylog, h - M.b, M.t, db.fields[view.y].unit === "bytes", isDate(view.y));
  const fmtAxis = (k: string) => (v: number) =>
    isDate(k) ? yearMonth(v) : db.fields[k]?.unit !== "bytes" && Math.abs(v) >= 1e6 ? si(v) : format(k, db.fields[k], v);
  const fx = fmtAxis(view.x);
  const fy = fmtAxis(view.y);
  const guide = GUIDES[`${view.x}|${view.y}`];
  const guideLine = useMemo(() => {
    if (!guide || pts.length < 2) return [];
    const ks = pts.map((p) => p.y / p.x ** guide.exp).sort((a, b) => a - b);
    const k = ks[Math.floor(ks.length / 2)];
    const [lo, hi] = [Math.min(...pts.map((p) => p.x)), Math.max(...pts.map((p) => p.x))];
    return Array.from({ length: 41 }, (_, i) => {
      const x = xlog ? lo * (hi / lo) ** (i / 40) : lo + ((hi - lo) * i) / 40;
      return { x, y: k * x ** guide.exp };
    });
  }, [guide, pts, xlog]);
  const selected = new Set(view.sel.split(","));
  // With a front, label the front, the next layer behind it, the extremes and ticked models; otherwise all that fit.
  const second = useMemo(() => paretoFront(pts.filter((p) => !front.has(p)), view.pareto), [pts, front, view.pareto]);
  const labelled = !front.size ? pts : pts.filter((p) =>
    front.has(p) || second.has(p) || selected.has(p.m.id) || [Math.min, Math.max].some((f) =>
      p.x === f(...pts.map((q) => q.x)) || p.y === f(...pts.map((q) => q.y))));
  const avoid = [frontLine.map((p) => [sx(p.x), sy(p.y)]), guideLine.map((p) => [sx(p.x), sy(p.y)])];
  // Points drawn within a few pixels of each other share one label and one hover card.
  const stacks = new Map<Point, Point[]>();
  for (const p of pts)
    stacks.set(p, pts.filter((q) => Math.hypot(sx(q.x) - sx(p.x), sy(q.y) - sy(p.y)) < 4));
  const labels = placeLabels(pts, labelled, stacks, sx, sy, w, front, avoid);
  const missing = models.length - pts.length;

  return (
    <div className="plot-card" ref={ref}>
      <svg width={w} height={h} role="img" aria-label={`${db.fields[view.y].label} against ${db.fields[view.x].label}`}>
        <g className="grid">
          {sx.ticks.map((t) => (
            <g key={`x${t}`}>
              <line x1={sx(t)} x2={sx(t)} y1={M.t} y2={h - M.b} />
              <text x={sx(t)} y={h - M.b + 18} textAnchor="middle">{fx(t)}</text>
            </g>
          ))}
          {sy.ticks.map((t) => (
            <g key={`y${t}`}>
              <line x1={M.l} x2={w - M.r} y1={sy(t)} y2={sy(t)} />
              <text x={M.l - 10} y={sy(t) + 4} textAnchor="end">{fy(t)}</text>
            </g>
          ))}
        </g>
        <text className="axis-title" x={(M.l + w - M.r) / 2} y={h - 12} textAnchor="middle">
          {axisTitle(db, view, view.x, xlog)}
        </text>
        <text className="axis-title" transform={`translate(18 ${(M.t + h - M.b) / 2}) rotate(-90)`} textAnchor="middle">
          {axisTitle(db, view, view.y, ylog)}
        </text>
        <clipPath id="plot-area"><rect x={M.l} y={M.t} width={w - M.l - M.r} height={h - M.t - M.b} /></clipPath>
        {guideLine.length > 0 && (
          <g className="guide" clipPath="url(#plot-area)">
            <polyline points={guideLine.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} />
            <text x={sx(guideLine[0].x) + 4} y={sy(guideLine[0].y) + 16}>{guide.label}</text>
          </g>
        )}
        {frontLine.length > 1 && (
          <polyline className="front" points={frontLine.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} />
        )}
        {pts.map((p) => (
          <circle
            key={p.m.id}
            className={`dot${hot === p ? " hot" : ""}${front.size && !front.has(p) ? " off-front" : ""}${
              siblingScore(p.m, view.x) || siblingScore(p.m, view.y) ? " sibling" : ""
            }`}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={hot === p ? 6.5 : 5}
          />
        ))}
        {labels.map(({ p, x, y, name }) => (
          <text key={p.m.id} className={`dot-label${front.size && !front.has(p) ? " off-front" : ""}`} x={x} y={y}>
            {name}
          </text>
        ))}
        {pts.map((p) => (
          <circle
            key={`hit-${p.m.id}`}
            className="hit"
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={13}
            onMouseEnter={() => setHotId(p.m.id)}
            onMouseLeave={() => setHotId(null)}
            onClick={() => open(p.m.id)}
          />
        ))}
      </svg>
      {hot && (
        <div className="preview" style={{ left: Math.min(sx(hot.x) + 18, w - 260), top: sy(hot.y) + 12 }}>
          <div className="preview-head"><b>{hot.m.name}</b> · {hot.m.org}</div>
          {stacks.get(hot)!.length > 1 && (
            <div className="preview-src">Same point: {stacks.get(hot)!.filter((q) => q !== hot).map((q) => q.m.name).join(", ")}</div>
          )}
          <div>{db.fields[view.x].label}: <b>{fx(hot.x)}</b></div>
          <div>{db.fields[view.y].label}: <b>{fy(hot.y)}</b></div>
        </div>
      )}
      <p className="footnote">
        {pts.length} of {models.length} models plotted.
        {missing > 0 && ` ${missing} lack one of the two values and are omitted, not drawn at zero.`}
        {front.size > 0 && ` ${front.size} on the Pareto front (${PARETO[view.pareto]}).`}
        {[view.x, view.y].some((k) => k.startsWith("eval.")) && " Hollow points are base checkpoints plotted with their post-trained sibling's score."}
        {guide && ` ${guide.note}`}
        {labelled.length < pts.length && " Labels show the front, the layer behind it, the extremes and ticked models; hover the others."}
        {" "}Click a point for its sources.
      </p>
    </div>
  );
}

function axisTitle(db: DB, view: View, k: string, log: boolean) {
  const dt = view.kvdt.toUpperCase();
  const extra = KV_CONTEXT_KEYS.includes(k) ? ` (${dt}, ${view.ctx.toLocaleString("en-US")} tokens)` : k.startsWith("kv.") ? ` (${dt})` : "";
  return `${db.fields[k].label}${extra}${log ? " · log scale" : ""}`;
}

function yearMonth(v: number): string {
  const y = Math.floor(v);
  const m = Math.min(12, Math.floor((v - y) * 12) + 1);
  return `${y}-${String(m).padStart(2, "0")}`;
}

type Scale = ((v: number) => number) & { ticks: number[] };

function scale(vals: number[], log: boolean, a: number, b: number, binary = false, date = false): Scale {
  const maxTicks = Math.abs(b - a) < 420 ? 4 : 8;
  const t = (v: number) => (log ? Math.log10(v) : v);
  let lo = vals.length ? Math.min(...vals.map(t)) : 0;
  let hi = vals.length ? Math.max(...vals.map(t)) : 1;
  const pad = (hi - lo || 1) * 0.08;
  lo -= pad;
  hi += pad;
  const f = ((v: number) => a + ((t(v) - lo) / (hi - lo)) * (b - a)) as Scale;
  f.ticks = log ? (binary ? binaryTicks(lo, hi, maxTicks) : logTicks(lo, hi)) : date ? monthTicks(lo, hi) : linTicks(lo, hi);
  return f;
}

function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  const steps = hi - lo < 1.6 ? [1, 2, 5] : [1];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++)
    for (const s of steps) {
      const v = s * 10 ** e;
      if (Math.log10(v) >= lo && Math.log10(v) <= hi) out.push(v);
    }
  return out;
}

/** Powers of two (1 GiB, 2 GiB, 4 GiB, ...), thinned to at most `max` ticks. */
function binaryTicks(lo: number, hi: number, max: number): number[] {
  const all: number[] = [];
  for (let e = Math.ceil(lo / Math.log10(2)); e * Math.log10(2) <= hi; e++) all.push(2 ** e);
  const step = Math.ceil(all.length / max);
  return all.filter((_, i) => i % step === 0);
}

/** Month starts every 1, 2, 3 or 6 months, whichever gives at most eight ticks. */
function monthTicks(lo: number, hi: number): number[] {
  const step = [1, 2, 3, 6, 12].find((s) => ((hi - lo) * 12) / s <= 8) ?? 12;
  const out: number[] = [];
  for (let i = Math.ceil((lo * 12) / step) * step; i <= hi * 12; i += step) out.push(i / 12);
  return out;
}

function linTicks(lo: number, hi: number): number[] {
  const raw = (hi - lo) / 5;
  const step = 10 ** Math.floor(Math.log10(raw));
  const s = step * ([1, 2, 5, 10].find((k) => raw <= step * k) ?? 10);
  const out: number[] = [];
  for (let v = Math.ceil(lo / s) * s; v <= hi; v += s) out.push(+v.toPrecision(12));
  return out;
}

/** Greedy direct labels: keep a label only if it overlaps no other label, point or line. A stack of
 * coinciding points gets one label naming the first and counting the rest. */
function placeLabels(
  pts: Point[], want: Point[], stacks: Map<Point, Point[]>, sx: Scale, sy: Scale, w: number, first: Set<Point>,
  lines: number[][][],
) {
  const boxes: number[][] = [];
  const out: { p: Point; x: number; y: number; name: string }[] = [];
  const done = new Set<Point>();
  const dots = pts.map((p) => [sx(p.x), sy(p.y)]);
  const along: number[][] = [];
  for (const line of lines)
    for (let i = 1; i < line.length; i++) {
      const [[x0, y0], [x1, y1]] = [line[i - 1], line[i]];
      const n = Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 3);
      for (let j = 0; j <= n; j++) along.push([x0 + ((x1 - x0) * j) / n, y0 + ((y1 - y0) * j) / n]);
    }
  const sorted = [...want].sort((a, b) => Number(first.has(b)) - Number(first.has(a)) || b.y - a.y);
  for (const p of sorted) {
    if (done.has(p)) continue;
    const stack = stacks.get(p) ?? [p];
    const name = stack.length > 1 ? `${p.m.name} +${stack.length - 1}` : p.m.name;
    const cx = sx(p.x), cy = sy(p.y);
    const width = name.length * 6.4;
    const spots: [number, number][] = [
      [9, 0], [-9 - width, 0], [-width / 2, -14], [-width / 2, 16], [9, -12], [9, 12], [-9 - width, -12], [-9 - width, 12],
    ];
    const own = (x: number, y: number) => Math.hypot(x - cx, y - cy) < 4;
    for (const [dx, dy] of spots) {
      const box = [cx + dx, cy + dy - 7, cx + dx + width, cy + dy + 7];
      if (box[0] < M.l || box[2] > w - M.r) continue;
      const inside = ([x, y]: number[]) => x > box[0] - 2 && x < box[2] + 2 && y > box[1] - 2 && y < box[3] + 2;
      const hitsLabel = boxes.some((b) => !(box[2] < b[0] || box[0] > b[2] || box[3] < b[1] || box[1] > b[3]));
      const hitsDot = dots.some(([x, y]) => x > box[0] - 4 && x < box[2] + 4 && y > box[1] - 4 && y < box[3] + 4 && !own(x, y));
      if (!hitsLabel && !hitsDot && !along.some(inside)) {
        boxes.push(box);
        out.push({ p, x: box[0], y: cy + dy + 4, name });
        stack.forEach((q) => done.add(q));
        break;
      }
    }
  }
  return out;
}
