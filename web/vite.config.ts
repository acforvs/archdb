import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

// Version the dataset URL so changed data bypasses the browser cache.
const dataVersion = createHash("sha256")
  .update(readFileSync(new URL("./public/data.json", import.meta.url)))
  .digest("hex")
  .slice(0, 12);

export default defineConfig({
  plugins: [react(), {
    name: "archdb-html",
    transformIndexHtml(html) {
      const generated = (name: string) => readFileSync(new URL(`./.generated/${name}`, import.meta.url), "utf8");
      return html
        .replace('href="./data.json"', `href="./data.json?v=${dataVersion}"`)
        .replace("<!-- archdb-head -->", generated("site-head.html"))
        .replace("<!-- archdb-header -->", generated("header.html"))
        .replace("<!-- archdb-footer -->", generated("footer.html"))
        .replace("<!-- archdb-catalog -->", generated("home.html"))
        .replace("</head>", generated("structured-data.html") + "</head>");
    },
  }],
  define: { __DATA_VERSION__: JSON.stringify(dataVersion) },
  base: "./",
  build: { outDir: "../site", emptyOutDir: true },
});
