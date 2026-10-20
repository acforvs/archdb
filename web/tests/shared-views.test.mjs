import assert from "node:assert/strict";
import { test } from "node:test";
import { createView, normalizeQuery, openView, saveView } from "../netlify/lib/shared-views.mjs";
import { parseView } from "../src/view.ts";

class Store {
  rows = new Map();
  async get(key) { return this.rows.get(key) ?? null; }
  async setJSON(key, value, options) {
    assert.equal(options.onlyIfNew, true);
    if (this.rows.has(key)) return { modified: false };
    this.rows.set(key, structuredClone(value));
    return { modified: true };
  }
}
const query = "preset=Custom&sort=scaling.batch_ratio&x=train.tokens&y=train.batch_tokens&ylog=1&cols=train.tokens%2Cscaling.batch_ratio%2Ctrain.batch_tokens%2Ctrain.batch%2Cparams.counted_active_non_embedding%2Cscaling.tpp_active&pareto=&by=half&feat=mtp";
const post = (body, headers = {}) => new Request("https://archdb.example/api/views", {
  method: "POST", headers: { "Content-Type": "application/json", ...headers }, body: JSON.stringify(body),
});

test("the supplied research view round trips with ordered columns and all defaults", async () => {
  const store = new Store();
  const created = await createView(post({ query }), store);
  assert.equal(created.status, 200);
  const { path } = await created.json();
  assert.match(path, /^\/[A-Za-z0-9_-]{10}$/);
  const opened = await openView(new Request("https://archdb.example" + path), store);
  assert.equal(opened.status, 302);
  const target = opened.headers.get("Location");
  assert.ok(target.startsWith("/#"));
  assert.deepEqual(parseView(target.slice(2)), parseView(query));
  assert.equal(new URLSearchParams(target.slice(2)).size, Object.keys(parseView(query)).length);
  assert.ok(opened.headers.get("Cache-Control").includes("s-maxage"));
});

test("equivalent views reuse a link, including concurrent requests", async () => {
  const store = new Store();
  const first = normalizeQuery("q=deepseek&preset=MoE");
  const second = normalizeQuery("preset=MoE&q=deepseek");
  const codes = await Promise.all(Array.from({ length: 10 }, (_, i) => saveView(i % 2 ? first : second, store)));
  assert.equal(new Set(codes).size, 1);
  assert.equal(store.rows.size, 1);
});

test("a short-code collision never overwrites another view", async () => {
  const store = new Store();
  const normalized = normalizeQuery(query);
  const original = await saveView(normalized, store);
  const other = { version: 1, query: normalizeQuery("q=qwen") };
  store.rows.set(original, other);
  const code = await saveView(normalized, store);
  assert.equal(code.length, 12);
  assert.deepEqual(store.rows.get(original), other);
  assert.equal(code, await saveView(normalized, store));
});

test("unknown, duplicate, malformed and oversized settings are rejected", async () => {
  for (const invalid of [null, {}, "url=https://example.org", "q=a&q=b", "ctx=NaN", "ctx=-1", "dir=0", "xlog=true", "tab=other", "q=" + "x".repeat(17000)]) {
    assert.equal((await createView(post({ query: invalid }), new Store())).status, 400);
  }
  assert.equal((await createView(post({ query: "" }, { "Content-Length": "65537" }), new Store())).status, 413);
  assert.equal((await createView(post({ query: "" }, { Origin: "https://other.example" }), new Store())).status, 403);
  assert.equal((await createView(post({ query: "" }, { "Content-Type": "text/plain" }), new Store())).status, 415);
  assert.equal((await createView(new Request("https://archdb.example/api/views"), new Store())).status, 405);
});

test("missing links return 404 and storage failures allow retrying", async () => {
  const store = new Store();
  for (const path of ["/nope", "/ABCDEFGHIJ", "/models/no-such-model/"]) {
    assert.equal((await openView(new Request("https://archdb.example" + path), store)).status, 404);
  }
  const failing = { get() { throw new Error("Storage unavailable"); } };
  assert.equal((await createView(post({ query }), failing)).status, 503);
  assert.equal((await openView(new Request("https://archdb.example/ABCDEFGHIJ"), failing)).status, 503);
});
