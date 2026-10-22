import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { layerStack } from "../src/layers.ts";

const db = JSON.parse(await readFile(new URL("../../exports/archdb.json", import.meta.url), "utf8"));
const model = (id) => db.models.find((m) => m.id === id);
const code = (id) => layerStack(model(id)).layers.map((k) => ({ global: "G", swa: "S", linear: "L" })[k]).join("");
const globalPositions = (id) => layerStack(model(id)).layers.flatMap((k, i) => k === "global" ? [i] : []);

test("each strip matches every count and uses the recorded sequence", () => {
  for (const m of db.models) {
    const s = layerStack(m);
    if (!s) continue;
    assert.equal(s.layers.length, m.facts["shape.layers"].value, m.id);
    for (const [kind, field] of [["global", "global_layers"], ["swa", "local_layers"], ["linear", "linear_layers"]])
      assert.equal(s.layers.filter((x) => x === kind).length, m.facts[`attn.${field}`]?.value ?? 0, m.id);
    if (m.facts["attn.layout"]?.sequence) {
      assert.equal(s.ordered, true, m.id);
      assert.deepEqual(s.types, m.facts["attn.layout"].sequence, m.id);
    }
  }
  assert.equal(layerStack(model("longcat-2")), null, "attention sublayers must not be confused with macro blocks");
});

test("the pinned layouts preserve prefixes, tails and irregular layer positions", () => {
  assert.equal(code("kimi-k3"), "LLLG".repeat(23) + "G");
  assert.equal(code("glm53-flash"), "LLLG".repeat(11) + "L");
  assert.equal(code("motif-3"), "GSSS".repeat(13) + "G");
  assert.equal(code("step-3-5-flash"), "G" + "SSSG".repeat(11));
  assert.equal(code("k-exaone-2"), "GS" + "SSSG".repeat(19));
  assert.equal(code("beam-1-0"), "SSSSSG".repeat(8) + "SSSG");
  assert.equal(code("deepseek-v4-flash-base"), "SS" + "G".repeat(41));
  assert.equal(code("deepseek-v41"), "SS" + "G".repeat(38));
  assert.equal(code("solar-open-2"), "GLLL".repeat(12));
  assert.equal(code("muse-spark-open"), "SG" + "SSSG".repeat(15));
  for (const id of ["laguna-s-2-1", "laguna-xs-2-1", "laguna-xs-2"])
    assert.equal(code(id), "GSSS".repeat(model(id).facts["shape.layers"].value / 4));
  for (const id of ["mimo-v2-flash-base", "mimo-v2-6-flash"])
    assert.deepEqual(globalPositions(id), [0, 5, 11, 17, 23, 29, 35, 41, 47]);
  for (const id of ["mimo-v2-6-pro", "mimo25-pro-base"])
    assert.deepEqual(globalPositions(id), [0, 7, 15, 23, 31, 39, 47, 55, 62, 69]);
  assert.deepEqual(globalPositions("nemotron3-super"), [4, 9, 14, 20, 26, 32, 38, 43]);
  assert.deepEqual(globalPositions("nemotron-3-ultra"), [4, 8, 13, 18, 22, 27, 32, 36, 41, 46, 50, 55]);
});

test("DeepSeek preserves compression types, encoder/decoder split and excludes drafters", () => {
  assert.deepEqual(layerStack(model("deepseek-v41")).types,
    ["SWA", "SWA", ...Array(18).fill("CSA2 m=2 + SWA"), ...Array(20).fill("CSA2 m=1 + SWA")]);
  assert.deepEqual(layerStack(model("deepseek-v4-flash-base")).types,
    ["SWA", "SWA", ...Array.from({ length: 41 }, (_, i) => i % 2 ? "HCA + SWA" : "CSA + SWA")]);
  assert.deepEqual(layerStack(model("deepseek-v4-pro")).types,
    ["HCA + SWA", "HCA + SWA", ...Array.from({ length: 59 }, (_, i) => i % 2 ? "HCA + SWA" : "CSA + SWA")]);
});

test("a missing, malformed or contradictory sequence never falls back to guessing from a ratio", () => {
  const m = structuredClone(model("muse-spark-open"));
  const n = m.facts["shape.layers"].value;
  m.facts["attn.layout"].value = "3 SWA : 1 global";
  for (const sequence of [undefined, [], ["SWA"], Array(n).fill("GQA"), Array(n).fill("mystery")]) {
    m.facts["attn.layout"].sequence = sequence;
    assert.equal(layerStack(m).ordered, false);
  }
});
