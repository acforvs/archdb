import { useEffect, useRef, useState } from "react";
import { serializeView, type View } from "./view";
import { shortLinkPath } from "./shortLink";

export function Share({ view }: { view: View }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [url, setUrl] = useState("");
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => { setOpen(false); }, [view]);
  useEffect(() => {
    if (open && url && !busy) input.current?.focus();
  }, [open, url, busy]);
  useEffect(() => {
    if (!open) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setOpen(false); trigger.current?.focus(); }
    };
    document.addEventListener("keydown", close);
    return () => document.removeEventListener("keydown", close);
  }, [open]);
  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, []);
  async function copy(link: string) {
    try {
      await navigator.clipboard.writeText(link);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }
  async function share() {
    setOpen(true);
    setBusy(true);
    setError("");
    setUrl("");
    setCopied(false);
    try {
      const path = await shortLinkPath(view);
      const link = new URL(path, location.origin).href;
      setUrl(link);
      void copy(link);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create a link. Try again.");
    } finally { setBusy(false); }
  }
  return (
    <div className="share" ref={root}>
      <button className="share-trigger" ref={trigger} onClick={share} disabled={busy} aria-expanded={open}>
        Share
      </button>
      {open && <div className="share-pop" role="group" aria-label="Share this view">
        <p className="share-title">Share this view</p>
        {busy && <p role="status">Creating link…</p>}
        {url && <>
          <input ref={input} aria-label="Share link" readOnly value={url} onFocus={(event) => event.currentTarget.select()} onClick={(event) => event.currentTarget.select()} />
          <button onClick={() => copy(url)}>{copied ? "Copied" : "Copy link"}</button>
          <p role="status">{copied ? "Link copied." : "Copy the link to share these settings."}</p>
        </>}
        {error && <><p role="alert">{error}</p><button onClick={share}>Try again</button>{" "}
          <button onClick={() => {
            const full = new URL("#" + serializeView(view), location.href).href;
            setError(""); setUrl(full); copy(full);
          }}>Use full link</button>
        </>}
      </div>}
    </div>
  );
}
