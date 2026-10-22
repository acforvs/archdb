import { cell, type DB, type Model } from "./data";
import { format } from "./format";
import { LayerStrip, layerStack } from "./LayerStack";
import { ScrollArea } from "./ScrollArea";
import type { View } from "./App";

export const COMPARE_MAX = 8;

/** Fields as rows, models as columns: for reading a few models side by side. */
export function Compare({ db, models, cols, view, set }: {
  db: DB; models: Model[]; cols: string[]; view: View; set: (p: Partial<View>) => void;
}) {
  if (models.length > COMPARE_MAX)
    return (
      <div className="table-card empty-state">
        Compare shows up to {COMPARE_MAX} models; {models.length} match now. Tick the models to compare in the table, or
        narrow the list with the model search (comma-separated, e.g. “beam, inkling”) or filters.
      </div>
    );
  const rows = cols.filter((k) => view.all || models.some((m) => cell(m, k, view)?.value != null));
  let prev = "";
  return (
    <div className="table-card">
      <ScrollArea>
        <table className="grid compare">
          <thead>
            <tr>
              <th className="sticky">Field</th>
              {models.map((m) => (
                <th key={m.id} className="model-col">
                  <button className="link" onClick={() => set({ model: m.id, focus: "" })}>{m.name}</button>
                  <span>{m.org} · {m.country} · {m.kind}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((k) => {
              const g = k.split(".")[0];
              const header = g !== prev ? (prev = g) : null;
              const values = models.map((m) => cell(m, k, view)?.value);
              const differ = new Set(values.map((v) => JSON.stringify(v ?? null))).size > 1;
              return [
                header && (
                  <tr key={`g-${g}`} className="group-sep">
                    {/* The label sits in the pinned column; a cell spanning the row could not stay pinned. */}
                    <th className="sticky">{db.groups[g]}</th>
                    <td colSpan={models.length} />
                  </tr>
                ),
                <tr key={k} className={differ ? "" : "same"}>
                  <th className="sticky field" title={db.fields[k].definition}>{db.fields[k].label}</th>
                  {models.map((m, i) => {
                    const v = values[i];
                    const empty = v === null || v === undefined;
                    const nd = cell(m, k, view)?.kind === "undisclosed";
                    const stack = k === "attn.layout" ? layerStack(m) : null;
                    return (
                      <td
                        key={m.id}
                        className={nd ? "undisclosed" : empty ? "empty" : "value wrap"}
                        onClick={() => !empty && set({ model: m.id, focus: k })}
                      >
                        {nd ? "n/d" : stack ? (
                          <>
                            <LayerStrip stack={stack} />
                            {!stack.ordered && <span className="strip-status">counts only</span>}
                            <span className="strip-text">{format(k, db.fields[k], v, true)}</span>
                          </>
                        ) : format(k, db.fields[k], v, true)}
                      </td>
                    );
                  })}
                </tr>,
              ];
            })}
          </tbody>
        </table>
      </ScrollArea>
      <p className="footnote">Rows where all models agree are dimmed. Click a value for its source.</p>
    </div>
  );
}
