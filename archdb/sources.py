"""Fetch cited sources and check facts against them.

Config facts whose location is a JSON Pointer must resolve to the stored value. Facts with a quote must
have that quote appear in the source text (whitespace, hyphenation and ligatures normalized).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
import unicodedata
import urllib.request
from pathlib import Path

import yaml

from .data import DATA, ROOT, Loader, Model

CACHE = ROOT / ".cache" / "sources"
TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9-]*(\s+[a-zA-Z_:][-\w:.]*\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+))*\s*/?>")
POINTER = re.compile(r"^(/[^\s,;|]+)(\s*\+\s*/[^\s,;|]+)*(\s*\|\s*(count\s+\S+|has\s+\S+|len))?$")


def fetch(url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / hashlib.sha256(url.encode()).hexdigest()[:24]
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "archdb-source-check"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
        if body[:2] == b"\x1f\x8b":  # some archives serve gzip without Content-Encoding
            body = gzip.decompress(body)
        path.write_bytes(body)
    return path


def text(url: str) -> str:
    """Normalized text of a source. Markup is stripped from everything except PDF extractions."""
    path = fetch(url)
    raw = path.read_bytes()
    if raw[:5] == b"%PDF-":
        txt = path.with_suffix(".txt")
        if not txt.exists():
            subprocess.run(["pdftotext", "-layout", str(path), str(txt)], check=True)
        return normalize(txt.read_text(errors="replace"))
    return normalize(raw.decode(errors="replace"), html=True)


def normalize(s: str, html: bool = False) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"-\s*\n\s*", "", s)  # words hyphenated across lines
    if html:
        s = TAG.sub(" ", s)
    s = re.sub(r"[*_`#|]", "", s)  # markdown emphasis and table pipes
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def resolve(doc, pointer: str):
    """JSON Pointer, extended with a final 'a:b' slice segment for lists and strings."""
    for part in pointer.strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(doc, (list, str)) and re.fullmatch(r"-?\d*:-?\d*", part):
            lo, hi = (int(x) if x else None for x in part.split(":"))
            doc = doc[lo:hi]
        else:
            doc = doc[int(part)] if isinstance(doc, list) else doc[part]
    return doc


def lookup(doc, at: str):
    """Evaluate '/a', '/a + /b' (sum), '/a | len', '/a | count X' (occurrences of X in a list or string),
    or '/a | has X' (whether any key or element contains the substring X)."""
    at, _, op = at.partition("|")
    found = [resolve(doc, p.strip()) for p in at.split("+")]
    found = sum(found) if len(found) > 1 else found[0]
    if op.strip() == "len":
        return len(found)
    if op.strip().startswith("has"):
        needle = op.strip().removeprefix("has").strip()
        return any(needle in str(x) for x in found) if not isinstance(found, str) else needle in found
    if op:
        target = op.strip().removeprefix("count").strip()
        return sum(1 for x in found if str(x) == target) if isinstance(found, list) else found.count(target)
    return found


def same(stored, found) -> bool:
    if isinstance(stored, bool) or isinstance(found, bool):
        return type(stored) is type(found) and stored == found
    if isinstance(stored, (int, float)) and isinstance(found, (int, float)):
        return abs(stored - found) <= 1e-9 * max(1, abs(found))
    return str(stored).strip().lower() == str(found).strip().lower()


def check_model(m: Model) -> tuple[list[str], dict[str, int]]:
    problems, stats = [], {"pointer_ok": 0, "quote_ok": 0, "unchecked": 0}
    for key, fact in m.facts.items():
        src = m.sources.get(fact.get("source"))
        if src is None:
            continue
        at = str(fact.get("at", ""))
        try:
            if src["kind"] == "config" and POINTER.match(at):
                found = lookup(json.loads(fetch(src["url"]).read_text()), at)
                if same(fact["value"], found):
                    stats["pointer_ok"] += 1
                else:
                    shown = repr(found) if len(repr(found)) < 120 else repr(found)[:120] + "..."
                    problems.append(f"{m.id}: {key} = {fact['value']!r} but {at} = {shown}")
            elif isinstance(fact["value"], list):
                missing = [st.get("name") for st in fact["value"] if not st.get("quote")]
                wrong = [
                    st.get("name")
                    for st in fact["value"]
                    if st.get("quote")
                    and normalize(st["quote"], html=True)
                    not in text(m.sources[st.get("source", fact["source"])]["url"])
                ]
                if missing:
                    stats["unchecked"] += 1
                elif wrong:
                    problems.append(f"{m.id}: {key} stage quotes not found for {wrong}")
                else:
                    stats["quote_ok"] += 1
            elif fact.get("quote"):
                if normalize(fact["quote"], html=True) in text(src["url"]):
                    stats["quote_ok"] += 1
                else:
                    problems.append(f"{m.id}: {key} quote not found in {fact['source']}")
            else:
                stats["unchecked"] += 1
        except (KeyError, IndexError, TypeError) as e:
            problems.append(f"{m.id}: {key} pointer {at} does not resolve ({e!r})")
        except OSError as e:
            problems.append(f"{m.id}: {key} source {fact['source']} unavailable ({e})")
    return problems, stats


def ignored_config_keys(path: Path = DATA / "config_keys.yaml") -> re.Pattern:
    doc = yaml.load(path.read_text(), Loader)
    names = [re.escape(k) for group in doc.values() for k in group.get("keys", [])]
    names += [p for group in doc.values() for p in group.get("patterns", [])]
    return re.compile("|".join(f"(?:{n})" for n in names))


def config_leaves(doc, ignore: re.Pattern):
    """(name, value) of every non-empty config entry, descending into nested dicts except ignored ones."""
    for k, v in doc.items():
        if ignore.fullmatch(k):
            continue
        if isinstance(v, dict) and v:
            yield from config_leaves(v, ignore)
        elif v not in (None, False, 0, "", [], {}, "default") or isinstance(v, float) and v != 0:
            yield k, v


def coverage(m: Model, ignore: re.Pattern | None = None) -> list[str]:
    """Config keys with a value that no fact cites or mentions and config_unused does not explain."""
    ignore = ignore or ignored_config_keys()
    cited = " ".join(str(f.get(part, "")) for f in m.facts.values() for part in ("at", "note", "quote"))
    cited += " ".join(
        str(st.get("note", "")) for f in m.facts.values() if isinstance(f["value"], list) for st in f["value"]
    )
    missing = []
    for src in m.sources.values():
        if src["kind"] != "config":
            continue
        doc = json.loads(fetch(src["url"]).read_text())
        if "weight_map" in doc:
            continue
        for key, _ in config_leaves(doc, ignore):
            if key in m.config_unused or re.search(rf"(?<![\w]){re.escape(key)}(?![\w])", cited):
                continue
            missing.append(key)
    return [
        f"{m.id}: config key {k} is neither cited nor listed in config_unused" for k in dict.fromkeys(missing)
    ]


def check_orgs(orgs: dict) -> list[str]:
    """Each organization's country is backed by a quote from its cited page."""
    problems = []
    for name, org in orgs.items():
        try:
            if normalize(org["quote"], html=True) not in text(org["url"]):
                problems.append(f"orgs.yaml: {name} quote not found at {org['url']}")
        except OSError as e:
            problems.append(f"orgs.yaml: {name} source unavailable ({e})")
    return problems
