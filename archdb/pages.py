"""Build model pages and social previews from the same records as the explorer."""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlencode
from xml.etree.ElementTree import Element, SubElement, tostring

from PIL import Image, ImageDraw, ImageFont

ORIGIN = "https://archdb.vladsavinov.com"
REPO = "https://github.com/acforvs/archdb"
DESCRIPTION = "LLM architectures and pretraining recipes, collected from papers, configs and released code."
SITE_HEAD = """<script src="/theme.js"></script>
<link rel="icon" href="/logo.svg" type="image/svg+xml">
<link rel="stylesheet" href="/site.css">
<link rel="stylesheet" href="/model-pages.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Instrument+Serif&display=swap" rel="stylesheet">"""
GITHUB_MARK = """<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" fill="currentColor">
<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>"""


def escape(value) -> str:
    return html.escape(str(value), quote=True)


def json_script(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def number(value: float) -> str:
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= scale:
            return f"{value / scale:.4g}{suffix}"
    return f"{value:g}"


def display(value, spec: dict) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        unit = spec.get("unit", "")
        if unit in {"params", "tokens"}:
            return number(value) + (" parameters" if unit == "params" else " tokens")
        if unit == "bytes":
            return f"{value:,.0f} bytes"
        return f"{value:,}" if isinstance(value, int) else f"{value:,.6g}"
    return str(value)


def model_description(model: dict) -> str:
    facts = model["facts"]
    parts = []
    for key, suffix in (
        ("params.total", " parameters"),
        ("params.active", " active in decode" if "params.active_prefill" in facts else " active"),
        ("shape.layers", " layers"),
    ):
        value = facts.get(key, {}).get("value")
        if isinstance(value, (int, float)):
            parts.append(number(value) + suffix)
    text = f"Architecture, pretraining details and sources for {model['name']}."
    return text + (" " + ", ".join(parts) + "." if parts else "")


def metadata(title: str, description: str, path: str, image: str) -> str:
    url, image_url = ORIGIN + path, ORIGIN + image
    tags = {
        "og:type": "website",
        "og:site_name": "archdb",
        "og:title": title,
        "og:description": description,
        "og:url": url,
        "og:image": image_url,
        "og:image:width": "1200",
        "og:image:height": "630",
        "og:image:alt": title,
    }
    head = f'<title>{escape(title)}</title><meta name="description" content="{escape(description)}">'
    head += f'<link rel="canonical" href="{escape(url)}">'
    head += "".join(f'<meta property="{key}" content="{escape(value)}">' for key, value in tags.items())
    for name, value in {
        "card": "summary_large_image",
        "title": title,
        "description": description,
        "image": image_url,
        "image:alt": title,
    }.items():
        head += f'<meta name="twitter:{name}" content="{escape(value)}">'
    return head


def site_header(data: dict | None = None, *, home: bool = False) -> str:
    tag = "h1" if home else "span"
    compare_current = ' aria-current="page"' if home else ""
    models_current = "" if home else ' aria-current="page"'
    return f'''<header class="masthead">
<a class="brand" href="/" aria-label="archdb home"><img src="/logo.svg" alt="" width="30" height="30">
<div><{tag} class="brand-name">archdb</{tag}>
<p>LLM architectures and pretraining recipes, with sources.</p></div></a>
<nav class="meta" aria-label="Site"><a href="/"{compare_current}>Compare models</a>
<a href="/models/"{models_current}>All models</a><span class="downloads">
<a href="/models.csv" download="archdb-models.csv" title="One row per model">models.csv</a>
<a href="/facts.csv" download="archdb-facts.csv" title="One row per value, with source and quote">facts.csv</a>
<a href="/data.json" download="archdb.json">json</a></span>
<a class="github-star" href="{REPO}" target="_blank" rel="noreferrer">{GITHUB_MARK} Star on GitHub</a>
<button class="icon" type="button" data-theme-toggle aria-label="Toggle dark mode">
<span class="theme-icon" aria-hidden="true"></span></button></nav></header>'''


def site_footer() -> str:
    return f'''<footer class="site-footer"><p>Corrections and new models are welcome as
<a href="{REPO}/issues" target="_blank" rel="noreferrer">issues</a> or pull requests ·
Data <a href="/LICENSE-DATA">CC0 1.0</a> · Code
<a href="{REPO}/blob/main/LICENSE" target="_blank" rel="noreferrer">MIT</a> ·
Quotes remain under their sources' licenses</p></footer>'''


def page(
    title: str, description: str, path: str, body: str, image: str = "/og.png", structured=None, data=None
) -> str:
    head = metadata(title, description, path, image)
    if structured:
        head += f'<script type="application/ld+json">{json_script(structured)}</script>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
{SITE_HEAD}{head}</head><body class="model-page">{site_header(data)}
<main>{body}</main>{site_footer()}</body></html>"""


def dataset_metadata(data: dict) -> dict:
    return {
        "@context": "https://schema.org",
        "@graph": [
            {"@type": "WebSite", "@id": ORIGIN + "/#website", "name": "archdb", "url": ORIGIN + "/"},
            {
                "@type": "Dataset",
                "@id": ORIGIN + "/#dataset",
                "name": "archdb",
                "description": DESCRIPTION,
                "url": ORIGIN + "/",
                "isAccessibleForFree": True,
                "license": "https://creativecommons.org/publicdomain/zero/1.0/",
                "variableMeasured": [field["label"] for field in data["fields"].values()],
                "distribution": [
                    {"@type": "DataDownload", "encodingFormat": mime, "contentUrl": ORIGIN + "/" + name}
                    for name, mime in (
                        ("models.csv", "text/csv"),
                        ("facts.csv", "text/csv"),
                        ("data.json", "application/json"),
                    )
                ],
            },
        ],
    }


def catalog(data: dict) -> str:
    rows = []
    for model in sorted(data["models"], key=lambda m: m["name"].lower()):
        cells = []
        for key in ("params.total", "params.active", "attn.kind", "shape.layers", "shape.d_model"):
            fact = model["facts"].get(key)
            text = display(fact["value"], data["fields"][key]) if fact else "Not disclosed"
            cells.append(f"<td>{escape(text)}</td>")
        rows.append(
            f'<tr><th scope="row"><a href="/models/{escape(model["id"])}/">{escape(model["name"])}</a>'
            f"<span>{escape(model['org'])}</span></th>{''.join(cells)}</tr>"
        )
    return f"""<div class="catalog"><table><caption>Model architectures</caption><thead><tr>
<th>Model</th><th>Total parameters</th><th>Active parameters</th><th>Attention</th><th>Layers</th><th>Hidden size</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table></div>"""


def evidence(fact: dict, model: dict) -> str:
    src = model["sources"][fact["source"]]
    url = src["url"].replace("/resolve/", "/blob/")
    quote = f"<blockquote>{escape(fact['quote'])}</blockquote>" if fact.get("quote") else ""
    note = f"<p>{escape(fact['note'])}</p>" if fact.get("note") else ""
    return (
        f'<details class="evidence"><summary>Source: <a href="{escape(url)}">{escape(src["kind"])}</a>'
        f" · {escape(fact['at'])}</summary>{quote}{note}</details>"
    )


def list_value(items: list, model: dict, parent: dict) -> str:
    rows = []
    for item in items:
        values = "".join(
            f"<dt>{escape(key.replace('_', ' ').capitalize())}</dt><dd>{escape(value)}</dd>"
            for key, value in item.items()
            if key not in {"name", "quote", "note", "source"}
        )
        proof = dict(parent, **{k: item[k] for k in ("quote", "note", "source", "at") if k in item})
        rows.append(
            f"<li><h4>{escape(item.get('name', ''))}</h4><dl>{values}</dl>{evidence(proof, model)}</li>"
        )
    return '<ul class="components">' + "".join(rows) + "</ul>"


def model_links(model: dict) -> list[tuple[str, str]]:
    """Link to the checkpoint, rather than a sibling repository used for shared code."""
    links = []
    sources = list(model["sources"].values())
    repositories = [s for s in sources if s["kind"] == "config" and s["url"].endswith("/config.json")]
    repositories += [s for s in sources if s["kind"] == "model-card"]
    for source in repositories:
        match = re.match(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/", source["url"])
        if match:
            links.append(("Hugging Face", "https://huggingface.co/" + match[1]))
            break
    for kind, label in (("paper", "Paper"), ("blog", "Blog")):
        url = next((s["url"] for s in model["sources"].values() if s["kind"] == kind), None)
        if url:
            links.append((label, url.replace("/resolve/", "/blob/")))
            break
    return links


def model_body(model: dict, data: dict) -> str:
    name, mid = escape(model["name"]), escape(model["id"])
    query = urlencode({"model": model["id"], "q": model["id"]})
    body = f'<p class="breadcrumb"><a href="/models/">Models</a> / {name}</p><h1>{name}</h1>'
    body += f'<p class="subtitle">{escape(model["org"])} · {escape(model["kind"])} checkpoint</p>'
    if model.get("note"):
        body += f"<p>{escape(model['note'])}</p>"
    outbound = "".join(
        f' <a class="button secondary" href="{escape(url)}" target="_blank" rel="noreferrer">{label} ↗</a>'
        for label, url in model_links(model)
    )
    body += f'<p class="actions"><a class="button" href="/#{escape(query)}">Open in the comparison table</a>{outbound}</p>'
    body += '<p class="context-note">KV calculations use one sequence at 131,072 tokens and 16-bit cache elements. '
    body += "They include bounded sliding-window caches. This is retained cache payload, not peak prefill memory. "
    body += "The comparison table also offers sourced mixed-precision layouts, including quantization scales.</p>"
    sections, links = [], []
    for group, label in data["groups"].items():
        rows = []
        for key, spec in data["fields"].items():
            if key.split(".")[0] != group:
                continue
            fact = model["facts"].get(key)
            derived = model["derived"].get(key)
            if fact:
                value = (
                    list_value(fact["value"], model, fact)
                    if isinstance(fact["value"], list)
                    else f'<p class="value">{escape(display(fact["value"], spec))}</p>'
                )
                if fact.get("basis") == "inferred":
                    value += '<p class="basis">Inferred from the source; see the explanation below.</p>'
                value += evidence(fact, model)
            elif derived:
                text = display(derived["value"], spec) if derived["value"] is not None else "Not calculated"
                value = f'<p class="value">{escape(text)}</p><p class="formula">Calculated: {escape(derived["formula"])}</p>'
            elif key in model["not_disclosed"]:
                value = '<p class="undisclosed">Not disclosed</p>'
            else:
                continue
            rows.append(
                f'<div class="field" id="{escape(key)}"><dt>{escape(spec["label"])}</dt>'
                f'<dd>{value}<p class="definition">{escape(spec["definition"])}</p></dd></div>'
            )
        if rows:
            links.append(f'<a href="#{escape(group)}">{escape(label)}</a>')
            sections.append(
                f'<section id="{escape(group)}"><h2>{escape(label)}</h2><dl>{"".join(rows)}</dl></section>'
            )
    body += '<nav class="sections" aria-label="Model fields">' + "".join(links) + "</nav>" + "".join(sections)
    return (
        body + f'<p><a href="/models/">All models</a> · <a href="/models/{mid}/">Link to this model</a></p>'
    )


def card(path: Path, data: dict, model: dict | None = None) -> None:
    image = Image.new("RGB", (1200, 630), "#f4f1ea")
    draw = ImageDraw.Draw(image)

    def text(x, y, value, size=28, fill="#1d1b17"):
        draw.text((x, y), str(value), font=ImageFont.load_default(size=size), fill=fill)

    draw.rectangle((64, 46, 112, 94), fill="#1d1b17")
    for i in range(5):
        x = 73 + i * 7
        draw.rectangle((x, 56, x + 4, 84), fill="#e0603a" if i == 3 else "#f4f1ea")
    draw.rectangle((64, 120, 1136, 122), fill="#1d1b17")
    text(130, 52, "archdb", 36)
    text(64, 560, "archdb.vladsavinov.com", 24, "#4a463d")
    if model is None:
        text(64, 174, "LLM architectures", 65)
        text(64, 258, "and pretraining recipes", 65)
        text(64, 390, f"{len(data['models'])} models, with sources for the details.", 32, "#4a463d")
        text(64, 442, "Compare models. Download the data.", 30, "#4a463d")
    else:
        title = model["name"]
        size = 58
        while draw.textlength(title, font=ImageFont.load_default(size=size)) > 1070 and size > 26:
            size -= 1
        text(64, 155, title, size)
        text(64, 235, model["org"], 28, "#4a463d")
        for i, (key, label) in enumerate(
            (
                ("params.total", "TOTAL PARAMS"),
                ("params.active", "ACTIVE / TOKEN"),
                ("shape.layers", "LAYERS"),
                ("shape.d_model", "HIDDEN SIZE"),
            )
        ):
            x = 64 + i * 275
            value = model["facts"].get(key, {}).get("value")
            if key == "params.active" and "params.active_prefill" in model["facts"]:
                label = "ACTIVE (DECODE)"
            text(x, 340, label, 19, "#4a463d")
            shown = (
                number(value)
                if isinstance(value, (int, float))
                else str(value)
                if value is not None
                else "Not disclosed"
            )
            if key.startswith("shape.") and isinstance(value, int):
                shown = f"{value:,}"
            text(x, 382, shown, 37 if value else 25)
        kinds = str(model["facts"].get("attn.kind", {}).get("value", "Architecture and pretraining"))
        if len(kinds) > 94:
            kinds = kinds[:91] + "..."
        text(64, 480, kinds, 27, "#4a463d")
        prefill = model["facts"].get("params.active_prefill", {}).get("value")
        if isinstance(prefill, (int, float)):
            text(64, 522, number(prefill) + " active parameters during prefill", 22, "#4a463d")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def write(data: dict, public: Path, generated: Path) -> None:
    models = public / "models"
    social = public / "social"
    for directory in (models, social):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "site-head.html").write_text(SITE_HEAD)
    (generated / "header.html").write_text(site_header(data, home=True))
    (generated / "footer.html").write_text(site_footer())
    intro = "<h2>Models</h2><p>LLM architectures and pretraining recipes, with links to their sources.</p>"
    intro += '<p><a href="/models/">Browse model details</a> · <a href="/models.csv">Download CSV</a> · <a href="/data.json">Download JSON</a></p>'
    (generated / "home.html").write_text('<main class="static-home">' + intro + catalog(data) + "</main>")
    (generated / "structured-data.html").write_text(
        '<script type="application/ld+json">' + json_script(dataset_metadata(data)) + "</script>"
    )
    listing = "<h1>Models</h1><p>Open a model to see its architecture, pretraining details and sources.</p>"
    (models / "index.html").write_text(
        page(
            "LLM architectures and pretraining recipes · archdb",
            DESCRIPTION,
            "/models/",
            listing + catalog(data),
            data=data,
        )
    )
    for model in data["models"]:
        path = "/models/" + model["id"] + "/"
        title = model["name"] + " architecture and pretraining · archdb"
        structured = {
            "@context": "https://schema.org",
            "@type": "TechArticle",
            "headline": title,
            "url": ORIGIN + path,
            "description": model_description(model),
            "isPartOf": {"@id": ORIGIN + "/#website"},
            "citation": [source["url"] for source in model["sources"].values()],
        }
        directory = models / model["id"]
        directory.mkdir()
        (directory / "index.html").write_text(
            page(
                title,
                model_description(model),
                path,
                model_body(model, data),
                image=f"/social/{model['id']}.png",
                structured=structured,
                data=data,
            )
        )
        card(social / (model["id"] + ".png"), data, model)
    card(public / "og.png", data)
    sitemap = Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    for path in ["/", "/models/", *("/models/" + m["id"] + "/" for m in data["models"])]:
        SubElement(SubElement(sitemap, "url"), "loc").text = ORIGIN + path
    (public / "sitemap.xml").write_bytes(tostring(sitemap, encoding="utf-8", xml_declaration=True))
    (public / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {ORIGIN}/sitemap.xml\n")
