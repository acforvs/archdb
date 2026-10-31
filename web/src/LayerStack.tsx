import { MIXER_LABEL, type Kind, type Mixer, type Stack } from "./layers";

export { layerStack } from "./layers";
export type { Kind, Mixer } from "./layers";

/** Paler bars for more compressed caches; hollow bars for layers that read another layer's KV cache. */
const typeVariant = (type: string) =>
  type.includes("shared KV") ? " shared-kv" : type.startsWith("HCA") || type.startsWith("CSA2 m=1") ? " compressed-variant" : "";

/** Layer counts and diagram ordering. */
function describe(s: Stack): string {
  const counts = new Map<string, number>();
  for (const type of s.types) counts.set(type, (counts.get(type) ?? 0) + 1);
  const parts = [...counts].map(([type, c]) => `${c} ${type}`).join(" + ");
  return `${parts} = ${s.layers.length} token mixers${s.ordered ? "; input to output" : "; grouped by type, layer order not shown"}`;
}

/** Draw ordered layers from left to right; show only counts when order is unresolved. */
export function LayerStrip({ stack, unit = 4, height = 18, title }: {
  stack: Stack; unit?: number; height?: number; title?: string;
}) {
  const description = title ? `${title}; ${describe(stack)}` : describe(stack);
  if (!stack.ordered) {
    const counts = new Map<Mixer, number>();
    for (const k of stack.layers) counts.set(k, (counts.get(k) ?? 0) + 1);
    const labels: Record<Mixer, string> = { global: "global", swa: "SWA", linear: "linear / SSM" };
    return (
      <span className="layer-counts" role="img" aria-label={description} title={description}>
        {[...counts].map(([k, count]) => <span key={k}><i className={`l-${k}`} />{count} {labels[k]}</span>)}
        <small>Order not shown</small>
      </span>
    );
  }
  const bars: { x: number; k: Kind; type: string }[] = [];
  let x = 0;
  stack.kinds.forEach((k, i) => {
    bars.push({ x, k, type: stack.types[i] });
    x += unit;
  });
  const w = Math.max(1, unit - 1);
  return (
    <svg className="strip" width={x} height={height} viewBox={`0 0 ${x} ${height}`}
      role="img" aria-label={description}>
      <title>{description}</title>
      {/* Taller bars mark layers whose KV cache covers the full context (full or sparse), without relying on color. */}
      {bars.map((b, i) => {
        const variant = typeVariant(b.type);
        const h = b.k === "global" || b.k === "sparse" ? height : Math.round(height * 0.55);
        return <rect key={i} x={b.x} y={height - h} width={w} height={h} className={`l-${b.k}${variant}`} data-layer-type={b.type}>
          <title>Token mixer {i + 1}: {b.type}</title>
        </rect>;
      })}
    </svg>
  );
}

export function LayerTypeLegend({ stack }: { stack: Stack }) {
  if (!stack.ordered) return null;
  const types = [...new Set(stack.types)];
  return <span className="strip-legend">{types.map((type) => {
    const i = stack.types.indexOf(type);
    const variant = typeVariant(type);
    return <span key={type}><i className={`l-${stack.kinds[i]}${variant}`} />{type}</span>;
  })}</span>;
}

export function StripLegend({ kinds }: { kinds?: Kind[] }) {
  return (
    <span className="strip-legend">
      {(kinds ?? (["global", "sparse", "swa", "linear"] as Kind[])).map((k) => (
        <span key={k}><i className={`l-${k}`} />{MIXER_LABEL[k]}</span>
      ))}
    </span>
  );
}
