import { MIXER_LABEL, type Mixer, type Stack } from "./layers";

export { layerStack } from "./layers";
export type { Mixer } from "./layers";

/** Layer counts and diagram ordering. */
function describe(s: Stack): string {
  const counts = new Map<Mixer, number>();
  for (const k of s.layers) counts.set(k, (counts.get(k) ?? 0) + 1);
  const parts = [...counts].map(([k, c]) => `${c} ${MIXER_LABEL[k]}`).join(" + ");
  return `${parts} = ${s.layers.length} layers${s.ordered ? "" : "; grouped by type, layer order not shown"}`;
}

/** Draw one bar per layer, with input layers on the left. */
export function LayerStrip({ stack, unit = 4, height = 18, title }: {
  stack: Stack; unit?: number; height?: number; title?: string;
}) {
  const gap = stack.ordered ? 0 : unit;
  const bars: { x: number; k: Mixer }[] = [];
  let x = 0;
  stack.layers.forEach((k, i) => {
    if (i > 0 && k !== stack.layers[i - 1]) x += gap;
    bars.push({ x, k });
    x += unit;
  });
  const w = Math.max(1, unit - 1);
  return (
    <svg className={`strip${stack.ordered ? "" : " unordered"}`} width={x} height={height} viewBox={`0 0 ${x} ${height}`}
      role="img" aria-label={title ? `${title}; ${describe(stack)}` : describe(stack)}>
      <title>{title ? `${title}; ${describe(stack)}` : describe(stack)}</title>
      {/* Taller bars make global-attention positions visible without relying on color. */}
      {bars.map((b, i) => {
        const h = b.k === "global" ? height : Math.round(height * 0.55);
        return <rect key={i} x={b.x} y={height - h} width={w} height={h} className={`l-${b.k}`} />;
      })}
    </svg>
  );
}

export function StripLegend({ kinds }: { kinds?: Mixer[] }) {
  return (
    <span className="strip-legend">
      {(kinds ?? (["global", "swa", "linear"] as Mixer[])).map((k) => (
        <span key={k}><i className={`l-${k}`} />{MIXER_LABEL[k]}</span>
      ))}
    </span>
  );
}
