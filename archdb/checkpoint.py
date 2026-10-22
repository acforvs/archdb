"""Count parameters from the tensor headers of a released safetensors checkpoint.

A safetensors file starts with an 8-byte length and a JSON header listing every tensor's name, dtype
and shape, so the counts below need a few small range requests per shard, not the weights.
Tensors are classified by name; routed-expert parameters count towards active parameters in
proportion top_k / experts.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import time
import urllib.error
import urllib.request
from math import prod

from .data import ROOT, Model
from .sources import fetch

CACHE = ROOT / ".cache" / "headers"
HF_FILE = re.compile(r"^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<commit>[0-9a-f]{40})/")

# Quantization side tensors that store scales or shapes, not parameters.
SIDE = re.compile(r"(_scale_inv|_scales?|\.scales?|_scale_2|weight_shape|zero_point|input_scale|_amax)$")
# Elements stored per entry when weights are packed into integer containers.
PACKED = {("U8", "_blocks"): 2, ("I32", "weight_packed"): 8, ("U8", "weight_packed"): 2}

PARTS = ("embedding", "head", "lookup", "routed", "dense", "mtp", "other_modality")
MULTIMODAL = re.compile(
    r"(vision|visual|\bvit\b|audio|mm_projector|multi_modal|image_newline|siglip|patch_embed|"
    r"perceiver|merger|\.mlp1\.|speech|video)"
)
EMBEDDING = re.compile(r"(^|\.)(embed_tokens|tok_embeddings|wte|word_embeddings|embeddings|embed)\.weight$")
LOOKUP = re.compile(r"(engram|ngram|n_gram|oe_embed|embed_tokens_per_layer|per_layer_embed)")
# Experts stored as one fused tensor per projection with a leading expert dimension (Step).
FUSED_MOE = re.compile(r"\.moe\.(gate_proj|up_proj|down_proj|gate_up_proj|w1|w2|w3)\.weight$")
HEAD = re.compile(r"(^|\.)(lm_head|embed_out|unembed)\.weight$|^output\.weight$")
TEXT_PREFIX = re.compile(r"^((model|language_model|llm|text_model)\.)+")
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    """GET with an optional byte range, retrying when Hugging Face rate-limits."""
    headers = {"User-Agent": "archdb-source-check"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == 7:
                raise
        time.sleep(15 * (attempt + 1))
    raise AssertionError("unreachable")


def header(url: str) -> dict:
    """JSON header of one safetensors file, cached under .cache/headers."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / hashlib.sha256(url.encode()).hexdigest()[:24]
    if not path.exists():
        n = struct.unpack("<Q", _get(url, 0, 7))[0]
        path.write_bytes(_get(url, 8, 7 + n))
    return json.loads(path.read_bytes())


def shards(repo: str, commit: str) -> list[str]:
    """Safetensors files of a repository at a commit, from the Hugging Face tree API."""
    url = f"https://huggingface.co/api/models/{repo}/tree/{commit}?recursive=1"
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / hashlib.sha256(url.encode()).hexdigest()[:24]
    if not path.exists():
        path.write_bytes(_get(url))
    tree = json.loads(path.read_bytes())
    files = [f["path"] for f in tree if f["path"].endswith(".safetensors")]
    # Copies in subfolders (original/, metal/) and consolidated single files duplicate the main shards.
    top = [f for f in files if "/" not in f]
    sharded = [f for f in top if not f.startswith("consolidated")]
    return sorted(sharded or top or files)


def elements(name: str, spec: dict, mx_scaled: bool = False) -> int:
    """Parameters stored in a tensor. Byte tensors with a microscaling scale hold two FP4 values per byte."""
    for (dtype, suffix), factor in PACKED.items():
        if spec["dtype"] == dtype and name.endswith(suffix):
            return prod(spec["shape"]) * factor
    if spec["dtype"] in ("I8", "U8") and mx_scaled:
        return prod(spec["shape"]) * 2
    return prod(spec["shape"])


def classify(name: str, layers: int | None) -> str:
    """Part of the model a tensor belongs to. `layers` is the config's num_hidden_layers: tensors in
    layers numbered at or beyond it are MTP modules appended to the stack (DeepSeek layers.61)."""
    low = name.lower()
    if MULTIMODAL.search(low):
        return "other_modality"
    m = LAYER.search(low)
    if "mtp" in low or "nextn" in low or (m and layers is not None and int(m.group(1)) >= layers):
        return "mtp"
    if LOOKUP.search(low):
        return "lookup"
    if EMBEDDING.search(low):
        return "embedding"
    if HEAD.search(low):
        return "head"
    if (".experts." in low and "shared" not in low) or FUSED_MOE.search(low):
        return "routed"
    return "dense"


def count(m: Model) -> dict | None:
    """Parameter counts by part for the checkpoint whose config the record cites, or None."""
    config = next(
        (s["url"] for s in m.sources.values() if s["kind"] == "config" and s["url"].endswith("config.json")),
        None,
    )
    match = HF_FILE.match(config or "")
    if not match:
        return None
    repo, commit = match["repo"], match["commit"]
    cfg = json.loads(fetch(config).read_text())
    layers = cfg.get("num_hidden_layers") or cfg.get("text_config", {}).get("num_hidden_layers")
    parts = dict.fromkeys(PARTS, 0)
    per_layer: dict[str, int] = {}
    for f in shards(repo, commit):
        tensors = header(f"https://huggingface.co/{repo}/resolve/{commit}/{f}")
        # FP4 weights packed two per byte carry a microscaling scale: an E8M0 `.scale` (DeepSeek) or
        # a byte-typed `weight_scale` (MiMo).
        mx = {
            re.sub(r"(\.scale|_scale)$", "", n)
            for n, t in tensors.items()
            if n != "__metadata__"
            and t["dtype"] in ("F8_E8M0", "U8")
            and re.search(r"(\.scale|weight_scale)$", n)
        }
        mx |= {n + ".weight" for n in mx}
        for name, spec in tensors.items():
            if name == "__metadata__" or SIDE.search(name):
                continue
            # Multimodal checkpoints prefix the text model (language_model., model.llm., ...).
            text = TEXT_PREFIX.sub("", name)
            part, n = classify(text, layers), elements(name, spec, name in mx)
            parts[part] += n
            layer = LAYER.search(text.lower())
            if layer and part in ("dense", "routed"):
                block = text[:layer.end(1)]
                per_layer[block] = per_layer.get(block, 0) + n
    return {"repo": repo, "commit": commit, **parts, "max_layer": max(per_layer.values(), default=0)}
