import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { layerStack } from "../src/layers.ts";

const db = JSON.parse(await readFile(new URL("../../exports/archdb.json", import.meta.url), "utf8"));
const model = (id) => db.models.find((m) => m.id === id);
const code = (id) => layerStack(model(id)).layers.map((k) => ({ global: "G", swa: "S", linear: "L" })[k]).join("");

test("each strip has one entry per layer and matches the sourced counts", () => {
  for (const m of db.models) {
    const s = layerStack(m);
    if (!s) continue;
    assert.equal(s.layers.length, m.facts["shape.layers"].value, m.id);
    const n = (k) => s.layers.filter((x) => x === k).length;
    assert.equal(n("global"), m.facts["attn.global_layers"].value, m.id);
    assert.equal(n("swa"), m.facts["attn.local_layers"]?.value ?? 0, m.id);
    assert.equal(n("linear"), m.facts["attn.linear_layers"]?.value ?? 0, m.id);
  }
});

test("stated orders: repeating ratios, leading runs and position notes", () => {
  assert.equal(code("kimi-k3"), "LLLG".repeat(23) + "G");
  assert.equal(code("glm53-flash"), "LLLG".repeat(11) + "L");
  assert.equal(code("motif-3"), "GSSS".repeat(13) + "G");
  assert.equal(code("step-3-5-flash"), "G" + "SSSG".repeat(11));
  assert.equal(code("k-exaone-2"), "GS" + "SSSG".repeat(19));
  assert.equal(code("beam-1-0"), "SSSSSG".repeat(8) + "SSSG");
  assert.equal(code("deepseek-v4-flash-base"), "SS" + "G".repeat(41));
  assert.equal(code("solar-open-2"), "GLLL".repeat(12));
  assert.equal(code("laguna-s-2-1"), "GSSS".repeat(12));
  assert.equal(code("laguna-xs-2-1"), "GSSS".repeat(10));
  assert.equal(code("laguna-xs-2"), "GSSS".repeat(10));
});

test("sums without an order are drawn grouped and flagged", () => {
  for (const id of ["mimo-v2-6-pro", "nemotron-3-ultra", "deepseek-v41"]) assert.equal(layerStack(model(id)).ordered, false, id);
  assert.equal(layerStack(model("longcat-2")), null);
});

test("a ratio cannot place an unexplained remainder at the end of the stack", () => {
  for (const id of ["mimo-v2-flash-base", "mimo-v2-6-flash", "muse-spark-open"]) {
    assert.equal(layerStack(model(id)).ordered, false, id);
  }
});

test("zero-length patterns and unknown mixer names do not invent an order", () => {
  const base = structuredClone(model("beam-1-0"));
  for (const text of ["0 SWA : 0 global", "5 mystery : 1 global"]) {
    base.facts["attn.layout"].value = text;
    assert.equal(layerStack(base).ordered, false, text);
  }
});
