const assert = require("node:assert/strict");
const { after, before, test } = require("node:test");
const { createServer } = require("node:http");
const { readFile } = require("node:fs/promises");
const { resolve, sep, extname } = require("node:path");
const browsers = require("playwright");

let browser, server, baseURL = process.env.ARCHDB_BASE_URL;
const site = resolve(__dirname, "../../site");
const mime = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
  ".xml": "application/xml", ".txt": "text/plain",
};

before(async () => {
  if (!baseURL) {
    const { createView, lookupView, openView } = await import("../netlify/lib/shared-views.mjs");
    const views = new Map();
    const store = {
      async get(key) { return views.get(key) ?? null; },
      async setJSON(key, value) {
        if (views.has(key)) return { modified: false };
        views.set(key, value);
        return { modified: true };
      },
    };
    server = createServer(async (req, res) => {
      const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
      const reply = async (response) => {
        res.writeHead(response.status, Object.fromEntries(response.headers));
        res.end(Buffer.from(await response.arrayBuffer()));
      };
      if (pathname === "/api/views" || pathname.startsWith("/api/views/")) {
        const chunks = [];
        for await (const chunk of req) chunks.push(chunk);
        const request = new Request(`http://${req.headers.host}${req.url}`, {
          method: req.method, headers: req.headers,
          ...(["GET", "HEAD"].includes(req.method) ? {} : { body: Buffer.concat(chunks) }),
        });
        await reply(await (pathname === "/api/views" ? createView : lookupView)(request, store));
        return;
      }
      const path = resolve(site, "." + pathname + (pathname.endsWith("/") ? "index.html" : ""));
      if (!path.startsWith(site + sep)) { res.writeHead(403).end(); return; }
      try {
        const body = await readFile(path);
        res.writeHead(200, { "Content-Type": mime[extname(path)] || "application/octet-stream" });
        res.end(body);
      } catch {
        await reply(await openView(new Request(`http://${req.headers.host}${req.url}`, { method: req.method }), store));
      }
    });
    await new Promise((done) => server.listen(0, "127.0.0.1", done));
    baseURL = `http://127.0.0.1:${server.address().port}`;
  }
  const engine = browsers[process.env.ARCHDB_BROWSER || "chromium"];
  assert.ok(engine?.launch, "ARCHDB_BROWSER must name a Playwright browser");
  browser = await engine.launch({
    headless: true,
    ...(process.env.ARCHDB_BROWSER_EXECUTABLE ? { executablePath: process.env.ARCHDB_BROWSER_EXECUTABLE } : {}),
  });
});

after(async () => {
  await browser?.close();
  if (server) {
    server.closeAllConnections();
    await new Promise((done) => server.close(done));
  }
});

async function withPlot(hash, check) {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  try {
    await page.goto(`${baseURL}/#${hash}`);
    await page.locator("circle.hit").last().waitFor();
    await check(page);
    assert.deepEqual(errors, [], "Opening or updating a plot must not throw");
  } finally { await page.close(); }
}

test("KV storage settings distinguish uniform payload from sourced formats and preserve numeric order", async () => {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  try {
    const hash = new URLSearchParams({
      preset: "Custom", cols: "kv.growth_per_token,kv.bytes_per_token", q: "DeepSeek",
      sort: "kv.growth_per_token", dir: "1", all: "1",
    });
    await page.goto(`${baseURL}/#${hash}`);
    const row = (name) => page.locator("table.grid tbody tr").filter({ has: page.getByRole("button", { name, exact: true }) });
    const v41 = row("DeepSeek-V4.1-Flash");
    await v41.waitFor();
    assert.equal(await v41.locator("td").first().innerText(), "3.13 KiB");
    const storage = page.getByRole("combobox", { name: "KV-cache dtype" });
    await storage.selectOption("fp4");
    await page.waitForFunction(() => document.querySelector(".grid tbody tr td")?.textContent === "800 B");
    assert.equal(await v41.locator("td").first().innerText(), "800 B");
    await storage.selectOption("sourced");
    await page.waitForFunction(() => document.querySelector(".grid tbody tr td")?.textContent === "890 B");
    assert.equal(await v41.locator("td").first().innerText(), "890 B");
    assert.equal(await v41.locator("td").nth(1).innerText(), "911 B");
    const names = await page.locator("table.grid tbody tr .model-cell button").allTextContents();
    assert.deepEqual(names.slice(0, 4), ["DeepSeek-V4.1-Flash", "DeepSeek-V4-Flash-Base", "DeepSeek-V4-Pro", "DeepSeek-V3.2"]);
    assert.equal(await row("DeepSeek-V3-Base").locator("td").first().innerText(), "not sourced");
    assert.match(await page.locator(".kv-note").innerText(), /4 of 5 models/);
    await v41.locator("td").first().click();
    const panel = page.getByRole("complementary", { name: "DeepSeek-V4.1-Flash sources" });
    await panel.waitFor();
    assert.equal(await panel.locator(".fact.focus .fact-value").innerText(), "890 B");
    assert.match(await panel.locator(".fact.focus .fact-src").first().innerText(), /scales/);
    assert.ok((await panel.locator(".cache-entry").allTextContents()).some((text) => text.includes("288 bytes/entry")));
    await panel.getByRole("button", { name: "Close", exact: true }).click();
    await page.evaluate(() => {
      const p = new URLSearchParams(location.hash.slice(1));
      p.set("f", "kv.growth_per_token<=1K");
      location.hash = p.toString();
    });
    await page.waitForFunction(() => document.querySelectorAll(".grid tbody tr").length === 1, null, { timeout: 5000 });
    assert.equal(await page.locator(".grid tbody tr .model-cell button").innerText(), "DeepSeek-V4.1-Flash");
    assert.deepEqual(errors, []);
  } finally { await page.close(); }
});

test("the global-KV plot uses sourced values and opens its evidence", async () => {
  await withPlot("tab=plot&x=kv.growth_per_token&kvdt=sourced&q=DeepSeek", async (page) => {
    assert.equal(await page.locator("circle.hit").count(), 4);
    assert.match(await page.locator(".kv-note").innerText(), /global KV excludes/);
    const points = page.locator("circle.hit");
    for (let i = 0; i < await points.count(); i++) {
      await points.nth(i).hover();
      if (await page.locator(".preview-head b").innerText() === "DeepSeek-V4.1-Flash") {
        assert.match(await page.locator(".preview").innerText(), /890 B/);
        await points.nth(i).click({ noWaitAfter: true });
        await page.getByRole("complementary", { name: "DeepSeek-V4.1-Flash sources" }).waitFor();
        return;
      }
    }
    assert.fail("V4.1 must have a point at its sourced global-KV value");
  });
});

test("the KV preset names every plotted model and identifies models with missing scores", async () => {
  await withPlot("tab=plot", async (page) => {
    const points = await page.locator("circle.hit").count();
    assert.equal(await page.getByRole("checkbox", { name: "All model names" }).isChecked(), true);
    assert.equal(await page.locator(".dot-label").count(), points);
    assert.ok((await page.locator(".dot-label").allTextContents()).includes("DeepSeek-V3.2"));
    await page.getByRole("checkbox", { name: "All model names" }).uncheck();
    await page.waitForFunction(() => document.querySelectorAll(".dot-label").length < document.querySelectorAll("circle.hit").length);
    await page.getByRole("checkbox", { name: "All model names" }).check();
    await page.setViewportSize({ width: 520, height: 950 });
    await page.waitForFunction(() => document.querySelectorAll(".dot-label").length === document.querySelectorAll("circle.hit").length);
    const missing = page.locator(".plot-missing");
    await missing.locator("summary").click();
    await missing.getByRole("button", { name: "Laguna-XS.2", exact: true }).waitFor();
    await missing.getByRole("button", { name: "Laguna-XS.2", exact: true }).click();
    await page.getByRole("complementary", { name: "Laguna-XS.2 sources" }).waitFor();
  });
});

for (const [name, hash] of [
  ["capability vs active parameters", "tab=plot&preset=MoE&sort=shape.aspect&x=params.counted_active_non_embedding&cols=shape.aspect%2Cparams.active%2Cparams.total"],
  ["capability vs KV", "tab=plot"],
]) {
  test(`hovering and clicking a point opens its sources: ${name}`, async () => {
    await withPlot(hash, async (page) => {
      const point = page.locator("circle.hit").last();
      await point.hover();
      const name = await page.locator(".preview-head b").innerText();
      await point.click({ noWaitAfter: true });
      const panel = page.getByRole("complementary", { name: `${name} sources` });
      await panel.waitFor({ timeout: 5000 });
      assert.equal(await panel.locator("h2").innerText(), name);
      assert.equal(await page.locator(".masthead").count(), 1);
      await panel.getByRole("button", { name: "Close", exact: true }).click();
      await page.locator("circle.hit").last().hover();
      await page.locator("circle.hit").last().click({ noWaitAfter: true });
      await panel.waitFor({ timeout: 5000 });
    });
  });
}

test("filtering out a hovered point clears its preview", async () => {
  await withPlot("tab=plot", async (page) => {
    await page.locator("circle.hit").last().hover();
    await page.locator(".preview-head b").waitFor();
    await page.evaluate(() => {
      const params = new URLSearchParams(location.hash.slice(1));
      params.set("q", "no-matching-model");
      location.hash = params.toString();
    });
    await page.getByText("0 of 0 models plotted.", { exact: false }).waitFor({ timeout: 5000 });
    assert.equal(await page.locator(".preview").count(), 0);
    assert.equal(await page.locator(".masthead").count(), 1);
  });
});

test("the logo returns a customized workspace to its defaults", async () => {
  const hash = new URLSearchParams({
    tab: "plot", preset: "MoE", sort: "shape.aspect", dir: "1",
    x: "params.counted_active_non_embedding", cols: "shape.aspect,params.active,params.total",
    q: "deepseek", f: "params.total>=100B", sel: "deepseek-v3", ctx: "32768", kvdt: "fp8",
  });
  await withPlot(hash.toString(), async (page) => {
    await page.getByRole("button", { name: "Customize", exact: true }).click();
    await page.getByRole("button", { name: "Hide axes", exact: true }).waitFor();
    const home = page.getByRole("link", { name: "archdb home", exact: true });
    await home.locator("img").click();
    await page.locator("table.grid tbody tr").first().waitFor();
    assert.equal(new URL(page.url()).hash, "");
    assert.equal(await page.getByRole("tab", { name: "Table", exact: true }).getAttribute("aria-selected"), "true");
    assert.equal(await page.getByRole("radio", { name: "Overview", exact: true }).getAttribute("aria-checked"), "true");
    assert.equal(await page.getByPlaceholder("Models: deepseek, qwen3.5").inputValue(), "");
    assert.equal(await page.locator(".pick:checked").count(), 0);
    const data = await (await page.request.get(`${baseURL}/data.json`)).json();
    assert.equal(await page.locator("table.grid tbody tr").count(), data.models.length);
    const sorting = page.locator('th[aria-sort="descending"]');
    assert.match(await sorting.innerText(), /Total params/);
    await page.getByRole("tab", { name: "Plot", exact: true }).click();
    await page.getByRole("button", { name: "Customize", exact: true }).waitFor();
    assert.equal(await page.getByRole("button", { name: "Hide axes", exact: true }).count(), 0);
    assert.equal(await page.getByRole("combobox", { name: "Context length" }).inputValue(), "131072");
    assert.equal(await page.getByRole("combobox", { name: "KV-cache dtype" }).inputValue(), "bf16");
    await home.focus();
    await page.keyboard.press("Enter");
    await page.locator("table.grid tbody tr").first().waitFor();
    assert.equal(new URL(page.url()).hash, "");
  });
});

for (const side of [false, true]) {
  test(`a cell opens the model scrolled to its field (${side ? "side by side" : "table"})`, async () => {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    try {
      const hash = new URLSearchParams({
        preset: "Custom", cols: "params.total,kv.bytes_at_context", q: "DeepSeek-V3-Base",
        ...(side ? { side: "1" } : {}),
      });
      await page.goto(`${baseURL}/#${hash}`);
      const row = side
        ? page.locator("table.compare tbody tr").filter({ has: page.locator("th.field", { hasText: /^KV at context$/ }) })
        : page.locator("table.grid tbody tr").filter({ has: page.getByRole("button", { name: "DeepSeek-V3-Base", exact: true }) });
      await row.locator("td").last().click();
      const panel = page.getByRole("complementary", { name: "DeepSeek-V3-Base sources" });
      await panel.waitFor();
      await panel.evaluate((node) => Promise.all(node.getAnimations().map((a) => a.finished)));
      assert.equal(await panel.locator(".fact.focus .fact-label").innerText(), "KV at context");
      await page.waitForFunction(() => {
        const panel = document.querySelector("aside.panel"), field = panel?.querySelector(".fact.focus");
        if (!panel || !field) return false;
        const p = panel.getBoundingClientRect(), f = field.getBoundingClientRect();
        return panel.scrollTop > 0 && f.top >= p.top && f.bottom <= p.bottom;
      }, null, { timeout: 5000 });
      await panel.getByRole("button", { name: "Close", exact: true }).click();
      await page.getByRole("button", { name: "DeepSeek-V3-Base", exact: true }).click();
      await panel.waitFor();
      await panel.evaluate((node) => Promise.all(node.getAnimations().map((a) => a.finished)));
      assert.equal(await panel.evaluate((node) => node.scrollTop), 0);
      assert.equal(await panel.locator(".fact.focus").count(), 0);
      assert.deepEqual(errors, []);
    } finally { await page.close(); }
  });
}

test("model facts and share metadata are available without JavaScript", async () => {
  const page = await browser.newPage({ javaScriptEnabled: false });
  try {
    await page.goto(baseURL + "/");
    await page.getByRole("heading", { name: "archdb", exact: true }).waitFor();
    const model = page.getByRole("link", { name: "DeepSeek-V3-Base", exact: true });
    assert.equal(await model.getAttribute("href"), "/models/deepseek-v3/");
    await model.click();
    await page.getByRole("heading", { name: "DeepSeek-V3-Base", exact: true }).waitFor();
    assert.equal(await page.locator('[id="moe.expert_dim"] .value').innerText(), "2,048");
    assert.equal(await page.locator('link[rel="canonical"]').getAttribute("href"), "https://archdb.vladsavinov.com/models/deepseek-v3/");
    const image = await page.locator('meta[property="og:image"]').getAttribute("content");
    assert.equal(image, "https://archdb.vladsavinov.com/social/deepseek-v3.png");
    assert.equal(await page.locator('meta[name="twitter:image"]').getAttribute("content"), image);
    const response = await page.request.get(baseURL + new URL(image).pathname);
    assert.equal(response.status(), 200);
    assert.match(response.headers()["content-type"], /image\/png/);
    const png = await response.body();
    assert.equal(png.readUInt32BE(16), 1200);
    assert.equal(png.readUInt32BE(20), 630);
    const sitemap = await page.request.get(baseURL + "/sitemap.xml");
    assert.equal(sitemap.status(), 200);
    assert.ok((await sitemap.text()).includes("https://archdb.vladsavinov.com/models/deepseek-v3/"));
    const missing = await page.request.get(baseURL + "/models/no-such-model/");
    assert.equal(missing.status(), 404);
  } finally { await page.close(); }
});

test("the header offers GitHub stars and model details have a shareable page", async () => {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
  try {
    await page.goto(baseURL + "/#q=DeepSeek-V3-Base");
    const star = page.getByRole("link", { name: "Star on GitHub", exact: true });
    await star.waitFor();
    assert.equal(await star.getAttribute("href"), "https://github.com/acforvs/archdb");
    await page.getByRole("button", { name: "DeepSeek-V3-Base", exact: true }).click();
    const panel = page.getByRole("complementary", { name: "DeepSeek-V3-Base sources" });
    assert.equal(
      await panel.getByRole("link", { name: "Hugging Face ↗", exact: true }).getAttribute("href"),
      "https://huggingface.co/deepseek-ai/DeepSeek-V3-Base",
    );
    await panel.getByRole("link", { name: "Model page", exact: true }).click();
    assert.equal(new URL(page.url()).pathname, "/models/deepseek-v3/");
    await page.getByRole("link", { name: "Open in the comparison table", exact: true }).click();
    await page.getByRole("complementary", { name: "DeepSeek-V3-Base sources" }).waitFor();
    assert.equal(await page.locator(".masthead").count(), 1);
  } finally { await page.close(); }
});

test("the explorer, directory and model pages share the saved theme", async () => {
  const context = await browser.newContext({ colorScheme: "dark" });
  const page = await context.newPage();
  try {
    await page.goto(baseURL + "/");
    const toggle = page.getByRole("button", { name: "Toggle dark mode", exact: true });
    await toggle.waitFor();
    const header = await page.locator(".masthead").innerText();
    const font = await page.locator("body").evaluate((el) => getComputedStyle(el).fontFamily);
    assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
    await toggle.click();
    assert.equal(await page.evaluate(() => localStorage.getItem("theme")), "light");
    await page.getByRole("link", { name: "All models", exact: true }).click();
    await page.waitForURL(new URL("/models/", baseURL).href);
    assert.equal(await page.locator(".masthead").innerText(), header);
    assert.equal(await page.locator("body").evaluate((el) => getComputedStyle(el).fontFamily), font);
    assert.equal(await page.getByRole("link", { name: "All models", exact: true }).getAttribute("aria-current"), "page");
    assert.equal(await page.locator("html").getAttribute("data-theme"), "light");
    assert.equal(await page.locator("body").evaluate((el) => getComputedStyle(el).backgroundColor), "rgb(244, 241, 234)");
    await page.getByRole("link", { name: "DeepSeek-V3-Base", exact: true }).click();
    await page.waitForURL(new URL("/models/deepseek-v3/", baseURL).href);
    assert.equal(await page.locator(".masthead").innerText(), header);
    assert.equal(await page.locator("html").getAttribute("data-theme"), "light");
    await toggle.click();
    assert.equal(await page.locator("body").evaluate((el) => getComputedStyle(el).backgroundColor), "rgb(23, 22, 17)");
    await page.getByRole("link", { name: "Compare models", exact: true }).click();
    await toggle.waitFor();
    assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
    await page.emulateMedia({ colorScheme: "light" });
    assert.equal(await page.locator("html").getAttribute("data-theme"), "dark");
    const directory = await context.newPage();
    await directory.goto(baseURL + "/models/");
    await toggle.click();
    await directory.waitForFunction(() => document.documentElement.dataset.theme === "light");
    assert.equal(await directory.locator("body").evaluate((el) => getComputedStyle(el).backgroundColor), "rgb(244, 241, 234)");
    await page.reload();
    await toggle.waitFor();
    assert.equal(await page.locator("html").getAttribute("data-theme"), "light");
  } finally { await context.close(); }
});

test("a reload keeps the shared header and never flashes the static directory", async () => {
  const page = await browser.newPage();
  let release, started;
  const gate = new Promise((resolve) => { release = resolve; });
  const requested = new Promise((resolve) => { started = resolve; });
  await page.route("**/data.json?v=*", async (route) => {
    started();
    await gate;
    await route.continue();
  });
  try {
    await page.goto(baseURL + "/", { waitUntil: "domcontentloaded" });
    await requested;
    await page.getByRole("status").waitFor();
    assert.equal(await page.locator(".masthead").count(), 1);
    assert.equal(await page.getByRole("status").isVisible(), true);
    assert.equal(await page.locator(".catalog").count(), 0);
    assert.equal(await page.locator(".static-home").count(), 0);
    release();
    await page.locator("table.grid tbody tr").first().waitFor();
    assert.equal(await page.locator(".masthead").count(), 1);
    assert.equal(await page.locator(".catalog").count(), 0);
  } finally { release(); await page.close(); }
});

test("theme follows system changes until a preference is chosen, even without storage", async () => {
  const context = await browser.newContext({ colorScheme: "light" });
  await context.addInitScript(() => {
    Object.defineProperty(window, "localStorage", { get() { throw new Error("Storage blocked"); } });
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  try {
    await page.goto(baseURL + "/models/");
    assert.equal(await page.locator("html").getAttribute("data-theme"), "light");
    await page.emulateMedia({ colorScheme: "dark" });
    await page.waitForFunction(() => document.documentElement.dataset.theme === "dark");
    await page.getByRole("button", { name: "Toggle dark mode", exact: true }).click();
    assert.equal(await page.locator("html").getAttribute("data-theme"), "light");
    assert.equal(await page.locator("body").evaluate((el) => getComputedStyle(el).backgroundColor), "rgb(244, 241, 234)");
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test("Share creates a reusable short link that restores a view in a fresh browser", async () => {
  const query = "preset=Custom&sort=train.batch_tokens&x=train.tokens&y=train.batch_tokens&ylog=1&cols=train.tokens%2Ctrain.batch_tokens%2Ctrain.batch%2Cparams.counted_active_non_embedding%2Cscaling.tpp_active&pareto=&by=half&feat=mtp";
  const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
  let context;
  const requests = [];
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.startsWith("/api/views")) requests.push(request);
  });
  try {
    await page.goto(baseURL + "/#" + query);
    await page.getByRole("button", { name: "Share", exact: true }).click();
    const field = page.getByRole("textbox", { name: "Share link", exact: true });
    await field.waitFor();
    const short = await field.inputValue();
    assert.equal(new URL(short).origin, new URL(baseURL).origin);
    assert.match(new URL(short).pathname, /^\/[A-Za-z0-9_-]{10,43}$/);
    assert.ok(short.length < 80);
    const initialRequests = requests.length;
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Share", exact: true }).click();
    await field.waitFor();
    assert.equal(await field.inputValue(), short);
    assert.equal(requests.length, initialRequests, "This tab should remember the existing link");
    context = await browser.newContext();
    const fresh = await context.newPage();
    const freshRequests = [];
    fresh.on("request", (request) => {
      if (new URL(request.url()).pathname.startsWith("/api/views")) freshRequests.push(request);
    });
    await fresh.goto(short);
    await fresh.locator("table.grid tbody tr").first().waitFor();
    const restored = new URLSearchParams(new URL(fresh.url()).hash.slice(1));
    for (const [key, value] of new URLSearchParams(query)) assert.equal(restored.get(key), value, key);
    assert.equal(await fresh.getByRole("radio", { name: /^Custom/ }).getAttribute("aria-checked"), "true");
    assert.match(await fresh.locator('th[aria-sort="descending"]').innerText(), /Batch/);
    await fresh.getByRole("button", { name: "Share", exact: true }).click();
    const reused = fresh.getByRole("textbox", { name: "Share link", exact: true });
    await reused.waitFor();
    assert.equal(await reused.inputValue(), short);
    assert.equal(freshRequests.length, 1);
    assert.equal(freshRequests[0].method(), "GET", "A new visitor should reuse the lookup without a creation request");
    assert.equal((await fresh.request.get(new URL("/no-such-short-view", baseURL).href)).status(), 404);
  } finally { await context?.close(); await page.close(); }
});

test("Share can copy the full URL if saving or clipboard access fails", async () => {
  const page = await browser.newPage();
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", { value: { writeText: async () => { throw new Error("Denied"); } } });
  });
  await page.route("**/api/views**", (route) => route.fulfill({ status: 503, body: "Unavailable" }));
  try {
    await page.goto(baseURL + "/#preset=MoE&q=deepseek");
    await page.getByRole("button", { name: "Share", exact: true }).click();
    await page.getByRole("alert").waitFor();
    await page.getByRole("button", { name: "Use full link", exact: true }).click();
    const link = await page.getByRole("textbox", { name: "Share link", exact: true }).inputValue();
    assert.equal(new URLSearchParams(new URL(link).hash.slice(1)).get("q"), "deepseek");
    assert.equal(await page.getByRole("button", { name: "Copy link", exact: true }).count(), 1);
    await page.keyboard.press("Escape");
    await page.getByRole("group", { name: "Share this view", exact: true }).waitFor({ state: "detached" });
    assert.equal(await page.getByRole("group", { name: "Share this view", exact: true }).count(), 0);
    assert.equal(await page.getByRole("button", { name: "Share", exact: true }).evaluate((el) => el === document.activeElement), true);
  } finally { await page.close(); }
});

test("switching column presets replaces hidden sorts and preserves visible sorts", async () => {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1050 } });
  const hash = new URLSearchParams({
    preset: "Custom", sort: "train.batch_tokens", dir: "1", cols: "train.tokens,train.batch_tokens",
    q: "deepseek", ctx: "32768", sel: "deepseek-v3",
  });
  try {
    await page.goto(baseURL + "/#" + hash);
    await page.locator('th[aria-sort="ascending"]').waitFor();
    assert.match(await page.locator('th[aria-sort="ascending"]').innerText(), /Batch/);
    await page.getByRole("radio", { name: "Overview", exact: true }).click();
    assert.match(await page.locator('th[aria-sort="descending"]').innerText(), /Total params/);
    assert.equal(await page.locator(".count-note").innerText().then(t => t.includes("sorted by")), false);
    const current = new URLSearchParams(new URL(page.url()).hash.slice(1));
    assert.equal(current.has("sort"), false);
    assert.equal(current.has("dir"), false);
    for (const key of ["q", "ctx", "sel"]) assert.equal(current.get(key), hash.get(key));
    const active = page.getByRole("columnheader").filter({ hasText: /^Active params/ });
    await active.click();
    await active.click();
    await page.getByRole("radio", { name: "MoE", exact: true }).click();
    assert.match(await page.locator('th[aria-sort="ascending"]').innerText(), /Active params/);
    assert.equal(new URLSearchParams(new URL(page.url()).hash.slice(1)).get("sort"), "params.active");
    await page.getByRole("radio", { name: "Attention & KV", exact: true }).click();
    assert.match(await page.locator('th[aria-sort="ascending"]').innerText(), /Model/);
    assert.equal(new URLSearchParams(new URL(page.url()).hash.slice(1)).get("sort"), "name");
    assert.equal(await page.locator(".count-note").innerText().then(t => t.includes("sorted by")), false);
  } finally { await page.close(); }
});

test("old shared views discard retired columns, filters, focus and sorts", async () => {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const legacy = new URLSearchParams({
    preset: "Custom", sort: "scaling.batch_ratio", dir: "-1",
    cols: "train.tokens,scaling.batch_ratio,train.batch_tokens",
    f: "scaling.batch_ratio>=10;params.active>=10B", q: "deepseek", ctx: "32768",
    model: "deepseek-v3", focus: "scaling.batch_ratio", sel: "deepseek-v3",
  });
  try {
    const saved = await page.request.post(baseURL + "/api/views", { data: { query: legacy.toString() } });
    assert.equal(saved.status(), 200);
    const { path } = await saved.json();
    await page.goto(baseURL + path);
    await page.locator("table.grid tbody tr").first().waitFor();
    await page.getByRole("complementary", { name: "DeepSeek-V3-Base sources" }).waitFor();
    await page.waitForFunction(() => !location.hash.includes("scaling.batch_ratio"));
    const restored = new URLSearchParams(new URL(page.url()).hash.slice(1));
    assert.equal(restored.get("cols"), "train.tokens,train.batch_tokens");
    assert.equal(restored.get("sort"), "name");
    assert.equal(restored.get("dir"), "1");
    assert.equal(restored.get("f"), "params.active>=10B");
    assert.equal(restored.has("focus"), false);
    for (const key of ["preset", "q", "ctx", "model", "sel"]) assert.equal(restored.get(key), legacy.get(key));
    assert.equal(await page.locator(".fact.focus").count(), 0);
    assert.match(await page.locator('th[aria-sort="ascending"]').innerText(), /Model/);
    assert.deepEqual(errors, []);
    const data = await (await page.request.get(baseURL + "/data.json")).json();
    assert.equal(Object.hasOwn(data.fields, "scaling.batch_ratio"), false);
    assert.ok(data.models.every((m) => !Object.hasOwn(m.derived, "scaling.batch_ratio")));
  } finally { await page.close(); }
});

for (const axis of ["x", "y"]) {
  test(`old plot URLs replace a retired ${axis} axis`, async () => {
    const hash = new URLSearchParams({ tab: "plot", [axis]: "scaling.batch_ratio" });
    await withPlot(hash.toString(), async (page) => {
      await page.waitForFunction(() => !location.hash.includes("scaling.batch_ratio"));
      assert.equal(await page.locator(".plot-card svg").count(), 1);
    });
  });
}

test("the batch versus tokens plot has no inferred critical-batch guide", async () => {
  await withPlot("tab=plot&x=train.tokens&y=train.batch_tokens&ylog=1&pareto=", async (page) => {
    assert.equal(await page.locator(".plot-card .guide").count(), 0);
    assert.equal((await page.locator(".plot-card").innerText()).includes("0.47"), false);
  });
});

test("compact model specs open the evidence for the value clicked", async () => {
  const page = await browser.newPage({ viewport: { width: 1440, height: 950 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  try {
    await page.goto(`${baseURL}/#model=deepseek-v3`);
    const panel = page.getByRole("complementary", { name: "DeepSeek-V3-Base sources" });
    await panel.waitFor();
    const active = panel.getByRole("button", { name: "Show source for Active params", exact: true });
    await active.click();
    await page.waitForFunction(() => document.querySelector('.fact.focus')?.dataset.key === 'params.active');
    assert.match(await panel.locator('.fact.focus').innerText(), /37B/);
    assert.equal(await panel.locator('.fact.focus blockquote').count(), 1);
    await panel.evaluate((el) => { el.scrollTop = 0; });
    await active.click();
    await page.waitForFunction(() => {
      const panel = document.querySelector('.panel').getBoundingClientRect();
      const fact = document.querySelector('.fact.focus').getBoundingClientRect();
      return fact.top >= panel.top && fact.bottom <= panel.bottom;
    });
    await panel.getByRole("button", { name: "Show source for Total params", exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.fact.focus')?.dataset.key === 'params.total');
    assert.deepEqual(errors, []);
  } finally { await page.close(); }
});

test("layer layouts show Muse Spark's actual order and use counts for unresolved layouts", async () => {
  const page = await browser.newPage({ viewport: { width: 1440, height: 950 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  try {
    const hash = new URLSearchParams({ preset: "Custom", cols: "attn.layout", q: "Muse-Spark" });
    await page.goto(`${baseURL}/#${hash}`);
    const row = page.locator("table.grid tbody tr");
    await row.waitFor();
    const expected = "SG" + "SSSG".repeat(15);
    const order = async (strip) => strip.locator("rect").evaluateAll((bars) => bars.map((bar) =>
      bar.classList.contains("l-global") ? "G" : "S").join(""));
    assert.equal(await order(row.locator("svg.strip")), expected);
    assert.equal(await row.locator(".layer-counts").count(), 0);
    await row.getByRole("button", { name: "Muse-Spark-Open-816B-A42B", exact: true }).click();
    const panel = page.locator(".panel");
    await panel.waitFor();
    assert.equal(await order(panel.locator("svg.strip")), expected);

    hash.set("q", "MiMo-V2-Flash-Base");
    await page.goto(`${baseURL}/#${hash}`);
    await row.locator("svg.strip").waitFor();
    assert.equal(await order(row.locator("svg.strip")), "GSSSSG" + "SSSSSG".repeat(7));
    await row.getByRole("button", { name: "MiMo-V2-Flash-Base", exact: true }).click();
    await panel.locator("svg.strip").waitFor();
    assert.equal(await order(panel.locator("svg.strip")), "GSSSSG" + "SSSSSG".repeat(7));

    hash.set("q", "DeepSeek-V4.1-Flash");
    await page.goto(`${baseURL}/#${hash}`);
    await row.locator("svg.strip").waitFor();
    assert.deepEqual(await row.locator("svg.strip rect").evaluateAll((bars) => bars.map((b) => b.dataset.layerType)),
      ["SWA", "SWA", ...Array(18).fill("CSA2 m=2 + SWA"), ...Array(20).fill("CSA2 m=1 + SWA")]);
    await page.getByRole("checkbox", { name: "Select the models shown", exact: true }).check();
    hash.set("side", "1");
    await page.goto(`${baseURL}/#${hash}`);
    await page.locator(".compare svg.strip").waitFor();
    assert.equal(await page.locator(".compare svg.strip rect").count(), 40);

    await page.route("**/data.json*", async (route) => {
      const response = await route.fetch();
      const db = await response.json();
      delete db.models.find((m) => m.id === "muse-spark-open").facts["attn.layout"].sequence;
      await route.fulfill({ response, json: db });
    });
    hash.set("q", "Muse-Spark");
    hash.delete("side");
    await page.goto(`${baseURL}/#${hash}`);
    await page.reload();
    await row.locator(".layer-counts").waitFor();
    assert.equal(await row.locator("svg.strip").count(), 0);
    assert.match(await row.locator(".layer-counts").innerText(), /Order not shown/);
    await row.getByRole("button", { name: "Muse-Spark-Open-816B-A42B", exact: true }).click();
    await panel.locator(".layer-counts").waitFor();
    assert.equal(await panel.locator("svg.strip").count(), 0);
    assert.deepEqual(errors, []);
  } finally { await page.close(); }
});

for (const mobile of [false, true]) {
  test(`all model catalog cells open their model page (${mobile ? "touch" : "desktop"})`, async () => {
    const page = await browser.newPage(mobile
      ? { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true }
      : { viewport: { width: 1440, height: 950 } });
    try {
      await page.goto(`${baseURL}/models/`);
      const row = page.locator(".catalog tbody tr").first();
      const href = await row.locator("th a").getAttribute("href");
      const name = await row.locator("th a").innerText();
      for (const i of [0, 4]) {
        const cell = row.locator("td").nth(i);
        await cell.scrollIntoViewIfNeeded();
        const box = await cell.boundingBox();
        const position = { x: 5, y: box.height - 3 };
        await (mobile ? cell.tap({ position }) : cell.click({ position }));
        await page.waitForURL(new URL(href, baseURL).href);
        assert.equal(await page.locator("main h1").innerText(), name);
        await page.goBack();
      }
    } finally { await page.close(); }
  });
}

test("the row and detail links open the exact base checkpoint on Hugging Face", async () => {
  const page = await browser.newPage({ viewport: { width: 1440, height: 950 } });
  try {
    await page.goto(`${baseURL}/#q=DeepSeek-V4-Flash-Base`);
    const row = page.locator('table.grid tbody tr');
    await row.waitFor();
    assert.equal(await row.getByRole('link', { name: 'HF ↗', exact: true }).getAttribute('href'),
      'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base');
    await row.getByRole('button', { name: 'DeepSeek-V4-Flash-Base', exact: true }).click();
    const panel = page.getByRole('complementary', { name: 'DeepSeek-V4-Flash-Base sources' });
    assert.equal(await panel.getByRole('link', { name: 'Hugging Face ↗', exact: true }).getAttribute('href'),
      'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base');
  } finally { await page.close(); }
});

test("wide tables keep their scrollbar visible and support keyboard, wheel and drag", async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  try {
    await page.goto(baseURL);
    const region = page.getByRole('region', { name: 'Scrollable model table', exact: true });
    await region.waitFor();
    const bounds = await region.boundingBox();
    assert.ok(bounds.y + bounds.height <= 900, 'the horizontal scrollbar stays inside the viewport');
    const model = page.locator('.grid tbody .model').first();
    const pinned = (await model.boundingBox()).x;
    await region.focus();
    await page.mouse.move(0, 0);
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => document.querySelector('.table-scroll').scrollLeft > 100);
    assert.equal((await model.boundingBox()).x, pinned, 'model names stay pinned');
    await region.evaluate((el) => { el.scrollLeft = 0; });
    await page.locator('.grid thead .sticky').last().hover();
    await page.mouse.wheel(0, 300);
    await page.waitForFunction(() => document.querySelector('.table-scroll').scrollLeft > 100);
    await region.evaluate((el) => { el.scrollLeft = 0; });
    const cell = page.locator('.grid tbody tr').first().locator('td').first();
    const target = await cell.boundingBox();
    await page.mouse.move(target.x + target.width / 2, target.y + target.height / 2);
    await page.mouse.down();
    await page.mouse.move(target.x + target.width / 2 - 100, target.y + target.height / 2, { steps: 8 });
    await page.mouse.up();
    assert.ok(await region.evaluate((el) => el.scrollLeft) > 50);
    assert.equal(await page.locator('.panel').count(), 0, 'panning must not open a cell');
    await page.getByRole('button', { name: 'Scroll right', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.table-scroll').scrollLeft > 250);
  } finally { await page.close(); }
});

test("the feature overview handles empty filters and stale hover previews", async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  try {
    await page.goto(`${baseURL}/#tab=breakdown&by=features`);
    await page.locator('.bar-row').first().waitFor();
    assert.equal(await page.locator('.bar-row').count(), 10);
    await page.locator('.bar .seg').first().hover();
    await page.locator('.preview').waitFor();
    await page.getByPlaceholder('Models: deepseek, qwen3.5').fill('no-such-model-here');
    await page.getByText('No models match the filters.', { exact: true }).waitFor();
    assert.equal(await page.locator('.preview').count(), 0);
    assert.equal(await page.locator('.bar-row').count(), 0);
    assert.deepEqual(errors, []);
  } finally { await page.close(); }
});

test("selecting models enables the comparison view", async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  try {
    await page.goto(`${baseURL}/#q=DeepSeek-V3`);
    await page.locator('.grid tbody tr').first().waitFor();
    assert.equal(await page.getByRole('button', { name: 'Compare selected', exact: true }).isDisabled(), true);
    await page.getByRole('checkbox', { name: 'Select the models shown', exact: true }).check();
    await page.getByRole('button', { name: 'Compare selected (2)', exact: true }).click();
    await page.locator('.grid.compare').waitFor();
    assert.equal(await page.locator('.compare .model-col').count(), 2);
    await page.getByRole('button', { name: '← Back to table', exact: true }).click();
    await page.locator('.grid:not(.compare)').waitFor();
    assert.equal(await page.getByRole('checkbox', { name: 'Deselect the models shown', exact: true }).isChecked(), true);
  } finally { await page.close(); }
});
