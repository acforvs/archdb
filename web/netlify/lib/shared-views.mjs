import { createHash } from "node:crypto";
import { DEFAULTS, SHARE_HASH_PREFIX, parseView, serializeView } from "../../src/view.ts";

const MAX_QUERY = 16384;
export const CODE = /^[A-Za-z0-9_-]{10,43}$/;
const LENGTHS = [10, 12, 16, 24, 43];
const digestOf = (query) => createHash("sha256").update(SHARE_HASH_PREFIX + query).digest("base64url");
const json = (value, status = 200) => Response.json(value, { status, headers: { "Cache-Control": "no-store" } });

export function normalizeQuery(query) {
  if (typeof query !== "string" || Buffer.byteLength(query) > MAX_QUERY) throw new Error("Invalid view");
  const params = new URLSearchParams(query);
  const seen = new Set();
  for (const [key, value] of params) {
    if (!Object.hasOwn(DEFAULTS, key) || seen.has(key)) throw new Error("Invalid field");
    seen.add(key);
    if (typeof DEFAULTS[key] === "boolean" && !["0", "1"].includes(value)) throw new Error("Invalid boolean");
    if (typeof DEFAULTS[key] === "number" && (!value.trim() || !Number.isFinite(Number(value)))) throw new Error("Invalid number");
  }
  const view = parseView(query);
  if (!["table", "plot", "breakdown"].includes(view.tab)) throw new Error("Invalid tab");
  if (![1, -1].includes(view.dir) || !Number.isSafeInteger(view.ctx) || view.ctx <= 0) throw new Error("Invalid number");
  const normalized = serializeView(view, true);
  if (Buffer.byteLength(normalized) > MAX_QUERY) throw new Error("View is too large");
  return normalized;
}

export async function saveView(query, store) {
  const digest = digestOf(query);
  for (const length of LENGTHS) {
    const code = digest.slice(0, length);
    const existing = await store.get(code, { type: "json" });
    if (existing?.query === query) return code;
    if (existing) continue;
    const { modified } = await store.setJSON(code, { version: 1, query }, { onlyIfNew: true });
    if (modified || (await store.get(code, { type: "json" }))?.query === query) return code;
  }
  throw new Error("Could not allocate a link");
}

export async function lookupView(request, store) {
  if (!["GET", "HEAD"].includes(request.method)) return new Response("Method not allowed", { status: 405, headers: { Allow: "GET, HEAD" } });
  const digest = new URL(request.url).pathname.split("/").pop();
  if (!/^[A-Za-z0-9_-]{43}$/.test(digest ?? "")) return json({ error: "Not found" }, 404);
  try {
    for (const length of LENGTHS) {
      const code = digest.slice(0, length);
      const saved = await store.get(code, { type: "json" });
      if (!saved) break;
      if (typeof saved.query === "string" && digestOf(saved.query) === digest) {
        return Response.json({ path: "/" + code }, { headers: {
          "Cache-Control": "public, max-age=3600",
          "Netlify-CDN-Cache-Control": "public, durable, max-age=31536000",
          "X-Robots-Tag": "noindex",
        } });
      }
    }
    return json({ error: "Not found" }, 404);
  } catch {
    return json({ error: "Could not look up this view. Try again." }, 503);
  }
}

export async function createView(request, store) {
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405, headers: { Allow: "POST" } });
  const origin = request.headers.get("origin");
  if (origin && origin !== new URL(request.url).origin) return json({ error: "Invalid origin" }, 403);
  if (request.headers.get("content-type")?.split(";")[0].trim() !== "application/json") return json({ error: "Expected JSON" }, 415);
  if (Number(request.headers.get("content-length")) > 65536) return json({ error: "View is too large" }, 413);
  let query;
  try {
    const body = await request.text();
    if (Buffer.byteLength(body) > 65536) return json({ error: "View is too large" }, 413);
    query = normalizeQuery(JSON.parse(body)?.query);
  } catch {
    return json({ error: "Invalid view" }, 400);
  }
  try {
    return json({ path: "/" + await saveView(query, store) });
  } catch {
    return json({ error: "Could not save this view. Try again." }, 503);
  }
}

export async function openView(request, store) {
  if (!["GET", "HEAD"].includes(request.method)) return new Response("Method not allowed", { status: 405, headers: { Allow: "GET, HEAD" } });
  const code = new URL(request.url).pathname.slice(1).replace(/\/$/, "");
  if (!CODE.test(code)) return new Response("Not found", { status: 404 });
  try {
    const saved = await store.get(code, { type: "json" });
    if (!saved) return new Response("This shared view does not exist.", { status: 404 });
    const query = normalizeQuery(saved.query);
    return new Response(null, { status: 302, headers: {
      Location: "/#" + query,
      "Cache-Control": "public, max-age=3600, s-maxage=31536000",
      "Netlify-CDN-Cache-Control": "public, durable, max-age=31536000",
      "X-Robots-Tag": "noindex",
    } });
  } catch {
    return new Response("Could not open this view. Please try again.", { status: 503, headers: { "Cache-Control": "no-store" } });
  }
}
