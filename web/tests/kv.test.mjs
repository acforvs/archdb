import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { cell, KV_CONTEXT_KEYS } from "../src/data.ts";

const db = JSON.parse(await readFile(new URL("../../exports/archdb.json", import.meta.url), "utf8"));
const model = (id) => db.models.find((m) => m.id === id);
const value = (id, key, kvdt = "sourced", ctx = 131072) => cell(model(id), key, { kvdt, ctx })?.value;

test("sourced global KV reproduces DeepSeek-V4.1 Figure 1(b)", () => {
  assert.equal(value("deepseek-v32", "kv.growth_per_token"), 48068);
  assert.equal(value("deepseek-v4-flash-base", "kv.growth_per_token"), 3514.25);
  assert.equal(value("deepseek-v41", "kv.growth_per_token"), 890);
});

test("finite SWA storage is included and remains FP8 in V4.1's sourced recipe", () => {
  assert.equal(value("deepseek-v41", "kv.bytes_per_token"), 910.625);
  assert.equal(value("deepseek-v41", "kv.bytes_at_context"), 119357440);
  assert.equal(value("deepseek-v41", "kv.bytes_per_token_per_active"), 910.625 / 16);
  assert.equal(value("deepseek-v41", "kv.bytes_at_context_per_active"), 119357440 / 16);
  assert.equal(value("deepseek-v41", "kv.bytes_at_context", "sourced", 3), 65496);
});

test("uniform FP4 excludes metadata and does not impersonate the sourced recipe", () => {
  assert.equal(value("deepseek-v41", "kv.growth_per_token", "bf16"), 3200);
  assert.equal(value("deepseek-v41", "kv.growth_per_token", "fp8"), 1600);
  assert.equal(value("deepseek-v41", "kv.growth_per_token", "fp4"), 800);
  assert.match(cell(model("deepseek-v41"), "kv.growth_per_token", { kvdt: "fp4", ctx: 131072 }).formula, /overhead excluded/);
});

test("a missing sourced recipe stays unknown for every KV metric", () => {
  for (const key of ["kv.growth_per_token", ...KV_CONTEXT_KEYS]) {
    assert.equal(value("laguna-xs-2", key), null);
    assert.match(cell(model("laguna-xs-2"), key, { kvdt: "sourced", ctx: 131072 }).formula, /No complete sourced/);
  }
});

test("the browser's default-context numbers agree with every exported KV metric", () => {
  for (const m of db.models) {
    for (const key of ["kv.growth_per_token", ...KV_CONTEXT_KEYS]) {
      if (m.derived[key]) assert.equal(cell(m, key, { kvdt: "bf16", ctx: 131072 })?.value, m.derived[key].value, `${m.id} ${key}`);
    }
  }
});

test("MiMo SWA uses 192 key and 128 value dimensions", () => {
  for (const id of ["mimo-v2-flash-base", "mimo-v2-6-flash"]) {
    assert.equal(value(id, "kv.bytes_per_token", "bf16"), 23235);
  }
  for (const id of ["mimo25-pro-base", "mimo-v2-6-pro"]) {
    assert.equal(value(id, "kv.bytes_per_token", "bf16"), 51500);
  }
});
