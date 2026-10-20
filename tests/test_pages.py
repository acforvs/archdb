import copy
import json
from html.parser import HTMLParser
from xml.etree import ElementTree

import pytest
from PIL import Image

from archdb.data import load_fields, load_models, load_orgs
from archdb.export import dataset
from archdb.pages import ORIGIN, dataset_metadata, model_body, page, write


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.meta, self.links, self.ids = {}, [], []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta":
            self.meta[attrs.get("name") or attrs.get("property")] = attrs.get("content")
        if tag == "a":
            self.links.append(attrs["href"])
        if "id" in attrs:
            self.ids.append(attrs["id"])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    groups, fields = load_fields()
    data = dataset(load_models(), groups, fields, load_orgs())
    root = tmp_path_factory.mktemp("pages")
    public = root / "public"
    write(data, public, root / "generated")
    return data, public, root / "generated"


def test_every_model_has_crawlable_facts_sources_and_a_social_preview(built):
    data, public, _ = built
    for model in data["models"]:
        text = (public / "models" / model["id"] / "index.html").read_text()
        doc = Document(text)
        assert doc.meta["og:url"] == ORIGIN + f"/models/{model['id']}/"
        assert doc.meta["twitter:image"] == doc.meta["og:image"]
        assert doc.meta["og:image"].startswith(ORIGIN + "/social/")
        assert set(model["facts"]) <= set(doc.ids)
        assert set(model["not_disclosed"]) <= set(doc.ids)
        for fact in model["facts"].values():
            source = model["sources"][fact["source"]]["url"].replace("/resolve/", "/blob/")
            assert source in doc.links
        with Image.open(public / "social" / (model["id"] + ".png")) as image:
            assert image.format == "PNG"
            assert image.size == (1200, 630)


def test_sitemap_and_homepage_link_to_all_models(built):
    data, public, generated = built
    expected = {ORIGIN + "/", ORIGIN + "/models/"}
    expected |= {ORIGIN + f"/models/{m['id']}/" for m in data["models"]}
    xml = ElementTree.fromstring((public / "sitemap.xml").read_text())
    urls = [n.text for n in xml.findall("{*}url/{*}loc")]
    assert set(urls) == expected
    assert len(urls) == len(expected)
    assert "Sitemap: " + ORIGIN + "/sitemap.xml" in (public / "robots.txt").read_text()
    for file in (public / "models/index.html", generated / "home.html"):
        links = Document(file.read_text()).links
        assert all(f"/models/{m['id']}/" in links for m in data["models"])
    structured = dataset_metadata(data)["@graph"][1]
    assert structured["@type"] == "Dataset"
    assert {x["encodingFormat"] for x in structured["distribution"]} == {"text/csv", "application/json"}


def test_model_text_and_structured_data_escape_html(built):
    data, _, _ = built
    model = copy.deepcopy(data["models"][0])
    model["name"] = '<script>alert("model")</script>'
    key = next(k for k, f in model["facts"].items() if isinstance(f["value"], str))
    model["facts"][key]["value"] = '<img src=x onerror="alert(1)">'
    body = model_body(model, data)
    assert "<script>" not in body
    assert "<img src=x" not in body
    structured = {"@type": "Dataset", "description": "</script><script>alert(1)</script>"}
    rendered = page("test", "test", "/", body, structured=structured)
    inline = rendered.split('<script type="application/ld+json">')[1].split("</script>")[0]
    assert json.loads(inline) == structured
    assert "<script>" not in inline
