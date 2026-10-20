import { serializeView, viewDigest, type View } from "./view";

const known = new Map<string, string>();
const pending = new Map<string, Promise<string>>();

async function readPath(response: Response): Promise<string> {
  if (!response.ok) throw new Error("Could not create a link. Try again.");
  const { path } = await response.json();
  if (typeof path !== "string" || !/^\/[A-Za-z0-9_-]{10,43}$/.test(path)) {
    throw new Error("Could not create a link. Try again.");
  }
  return path;
}

export async function shortLinkPath(view: View): Promise<string> {
  const query = serializeView(view, true);
  const digest = await viewDigest(query);
  const cached = known.get(digest);
  if (cached) return cached;
  const active = pending.get(digest);
  if (active) return active;

  const request = (async () => {
    const existing = await fetch(`/api/views/${digest}`);
    const response = existing.status === 404
      ? await fetch("/api/views", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      })
      : existing;
    const path = await readPath(response);
    if (known.size >= 200) known.delete(known.keys().next().value!);
    known.set(digest, path);
    return path;
  })();
  pending.set(digest, request);
  try { return await request; }
  finally { pending.delete(digest); }
}
