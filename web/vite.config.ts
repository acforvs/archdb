import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { readFileSync } from "node:fs";

export default defineConfig({
  plugins: [react(), {
    name: "archdb-html",
    transformIndexHtml(html) {
      const generated = (name: string) => readFileSync(new URL(`./.generated/${name}`, import.meta.url), "utf8");
      return html
        .replace("<!-- archdb-head -->", generated("site-head.html"))
        .replace("<!-- archdb-header -->", generated("header.html"))
        .replace("<!-- archdb-footer -->", generated("footer.html"))
        .replace("<!-- archdb-catalog -->", generated("home.html"))
        .replace("</head>", generated("structured-data.html") + "</head>");
    },
  }],
  base: "./",
  build: { outDir: "../site", emptyOutDir: true },
});
