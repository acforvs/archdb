import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import type { DB } from "./data";
import "./styles.css";

const root = createRoot(document.getElementById("root")!);

declare const __DATA_VERSION__: string;

// Same URL as the preload in index.html; the version changes whenever the dataset does.
fetch(`data.json?v=${__DATA_VERSION__}`)
  .then((r) => r.json() as Promise<DB>)
  .then((db) =>
    root.render(
      <StrictMode>
        <App db={db} />
      </StrictMode>,
    ),
  )
  .catch(() => root.render(<p className="load-error">The model data could not be loaded. Please reload the page.</p>));
