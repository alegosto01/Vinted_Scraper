"""Ask a vision model whether two listings show the same product.

DINOv2 gives a number with no meaning attached: 0.588 could be the same phone on a
different desk or a different phone on the same desk, and the threshold that separates
them has to be invented. A vision model answers the actual question and says why, so the
two are kept side by side - the cosine ranks, the model decides.

One call per candidate, cheapest vision model, cached on disk.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

from config.project_config import settings  # noqa: E402,F401  (imports load_dotenv)

LOG = logging.getLogger("image_compare")
MODEL = "gpt-4.1-mini"  # nano contradicts itself on these pairs; mini stays coherent at ~$0.0003
PROMPT_VERSION = 4
MAX_IMAGES = 5
IMAGE_PX = 512

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["same_product", "same_colour", "confidence", "note"],
    "properties": {
        "same_product": {"type": "boolean"},
        "same_colour": {"type": "boolean", "description": "whether the colourway matches"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "note": {"type": "string", "description": "at most 10 words on what the photos show"},
    },
}

PROMPT = """You compare photos from two second-hand listings.

You are given every photo of the TARGET listing, then every photo of the CANDIDATE
listing, each announced by a line saying which listing it belongs to and its number.

Use all of them together. Listing photos are uneven: one may be a close-up of a label,
another a full view, another a detail of a defect. Judge from whichever photos actually
show the article, and do not conclude from a single unhelpful photo. A close-up of a brand
label tells you the brand, not the type of garment - do not read "POLO RALPH LAUREN" on a
label as meaning the article is a polo shirt.

same_product: whether they show the same kind of article in the same model and material -
what a buyer paying for the target would accept. Judge the object, not the
photo: ignore lighting, background, angle, crop, wrinkles and photo quality. Colour and fit are NOT part of this answer: a navy and a white shirt of one model are the
same product, and so are a slim fit and a custom fit of it. Do not reject on the wording of
a fit label.
Say false when the article itself differs - a polo shirt instead of a linen shirt, a
different model or line, a different material, or a different format.

same_colour: whether the colourway matches. Report it separately; it never changes
same_product.

Use confidence "low" when the photos are too poor or too partial to tell."""


def _cache_path(cache_dir: Path, target_id: str, candidate_id: str) -> Path:
    digest = hashlib.sha256(f"v{PROMPT_VERSION}:{target_id}:{candidate_id}".encode()).hexdigest()[:20]
    return cache_dir / f"{target_id}_{candidate_id}_{digest}.json"


def _encode(url: str) -> str | None:
    from PIL import Image

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image.thumbnail((IMAGE_PX, IMAGE_PX))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=80)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as exc:
        LOG.warning("Could not read %s: %s", url[:60], exc)
        return None


def compare_photos(target: dict, candidate: dict, cache_dir: Path, client=None,
                   max_images: int = MAX_IMAGES) -> dict:
    """{'same_product', 'confidence', 'note'}; empty dict when the model cannot answer."""
    target_id, candidate_id = str(target["item_id"]), str(candidate["item_id"])
    path = _cache_path(Path(cache_dir), target_id, candidate_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOG.warning("Bad image cache entry %s", path)

    target_urls = (target.get("photo_urls") or [])[:max_images]
    candidate_urls = (candidate.get("photo_urls") or [])[:max_images]
    if not target_urls or not candidate_urls:
        return {}

    if client is None:
        import openai

        if not os.getenv("OPENAI_API_KEY"):
            return {}
        client = openai.OpenAI()

    content: list[dict] = []
    for label, urls in (("TARGET", target_urls), ("CANDIDATE", candidate_urls)):
        for index, url in enumerate(urls, 1):
            encoded = _encode(url)
            if not encoded:
                continue
            content.append({"type": "text", "text": f"{label} photo {index} of {len(urls)}:"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    if sum(1 for part in content if part["type"] == "image_url") < 2:
        return {}

    try:
        options = {} if MODEL.startswith("gpt-5") else {"temperature": 0}
        response = client.chat.completions.create(
            model=MODEL,
            **options,
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": content}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "photo_match", "strict": True, "schema": SCHEMA}},
        )
        parsed = json.loads(response.choices[0].message.content)
    except Exception:
        LOG.exception("Photo comparison failed for %s vs %s", target_id, candidate_id)
        return {}

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return parsed
