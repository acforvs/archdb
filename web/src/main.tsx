import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import type { DB } from "./data";
import "./styles.css";

const root = createRoot(document.getElementById("root")!);

fetch("data.json")
  .then((r) => r.json() as Promise<DB>)
  .then((db) =>
    root.render(
      <StrictMode>
        <App db={db} />
      </StrictMode>,
    ),
  )
  .catch(() => root.render(<p className="load-error">Could not load data.json. Run `uv run archdb build` first.</p>));
