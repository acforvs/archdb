# Development

```sh
uv sync
uv run archdb build
npm --prefix web ci && npm --prefix web run build
uv run archdb serve          # http://127.0.0.1:8000
```

`npm --prefix web run dev` serves the site with hot reload. The explorer, model pages, and exports
can be hosted as static files. Short links use the Netlify Functions in `web/netlify/functions/`.

“Share” saves the current settings and copies a short URL. These links redirect to the full view
URL. The `shared-views-v1` site-wide [Netlify Blobs](https://docs.netlify.com/build/data-and-storage/netlify-blobs/)
store persists across deploys; identical views reuse a link. Share first checks a lookup keyed by
the full configuration hash. Successful lookups use Netlify's durable CDN cache, shared by visitors;
missing lookups are not cached. Only configurations not found in storage need a creation request.
Each browser tab also remembers up to 200 results. Cache misses, including after invalidation or
expiry, still invoke a function. Creation is limited to 30 requests per minute per IP;
lookups and short-link redirects each allow 120. These limits reduce abuse from one IP
but do not cap total usage across visitors.
Existing links and static hosting work without this service; the share dialog can
also copy a full URL. To test short links locally, run `netlify dev --dir site` after building.

`archdb build` also generates model pages under `/models/`, PNG link previews under `/social/`,
`robots.txt`, and `sitemap.xml`. Vite includes the generated catalog and dataset metadata in the
homepage HTML. The catalog is a `noscript` fallback; JavaScript-enabled browsers show a loading
message until the interactive table is ready. These files come from the model records; generated
pages and images are ignored by Git.
The explorer and model pages share the header and footer generated in `archdb/pages.py`, the base
styles in `web/public/site.css`, and the theme preference managed by `web/public/theme.js`.

## Deployment

Netlify uses [`netlify.toml`](../netlify.toml) to build the records and frontend into `site/`. To deploy
an already built checkout with an authenticated Netlify CLI:

```sh
netlify deploy --no-build --dir site    # preview
netlify deploy --prod --no-build --dir site
```

The deployed site contains the generated JSON, CSV files, and data license. Source downloads in
`.cache/` and Netlify account settings in `.netlify/` stay local.

After deploying, the sitemap is at `https://archdb.vladsavinov.com/sitemap.xml`. Add the site to
Google Search Console, verify ownership, and submit that URL to track indexing and search traffic.
Model links under `/models/` have individual link previews. Explorer settings in `#` URLs share the
homepage preview.

## Checks

```sh
uv run archdb validate           # schema, types, citations
uv run archdb check-sources      # fetch sources, verify every pointer and quote (needs pdftotext)
uv run archdb build --check      # exports/ matches the records
uv run archdb count-params       # re-count parameters from checkpoint headers (network)
uv run pytest
npm --prefix web run test:unit
```

Browser checks use the built site in `site/`:

```sh
npm --prefix web exec -- playwright install chromium
npm --prefix web run test:e2e
```

Set `ARCHDB_BASE_URL` to check a deployed site instead of the local build. To use an installed
browser, set `ARCHDB_BROWSER_EXECUTABLE` to its executable path.

To check WebKit, the browser engine used by Safari:

```sh
npm --prefix web exec -- playwright install webkit
ARCHDB_BROWSER=webkit npm --prefix web run test:e2e
```
