"""Turn a seller's listing title into a Vinted search query.

Sellers write titles for humans: emoji, shouting, sizes, condition, and three
languages in one line. Feeding that back into Vinted's catalog search either
returns noise (too generic) or nothing (too specific). This asks gpt-4.1-nano
for the product identity behind the title and builds a query from it.

Every distinct title is parsed once and cached on disk.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

from config.project_config import settings  # noqa: E402,F401  (imports load_dotenv)

LOG = logging.getLogger("title_query")
MODEL = "gpt-5-nano"  # reasoning tokens are what recognise "stich" as Stitch; ~€0.07/day at current volume
PROMPT_VERSION = 5  # bump to invalidate cached rewrites when the instructions change

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["brand", "model", "product_type", "query", "query_local", "confidence"],
    "properties": {
        "brand": {"type": ["string", "null"]},
        "model": {"type": ["string", "null"], "description": "specific model or line name, null if the title names none"},
        "product_type": {"type": ["string", "null"], "description": "what the object is, in English"},
        "query": {"type": "string", "description": "Vinted search text in English: brand + model + product type"},
        "query_local": {"type": "string", "description": "the same query written in Italian"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}

PROMPT = """You normalise second-hand marketplace listings into search queries.

Given a listing title, and often its description, identify the product and return a search
query that would find the SAME product listed by other sellers on Vinted Italy.

Sellers frequently write a lazy title ("Anello Pandora", "The one") and name the actual
product in the description. Read the description whenever it is given: take the model name,
reference or line from there when the title lacks it, and prefer it over guessing.

Rules for the query:
- brand + model + product type, in that order
- ALWAYS keep the brand if the title or the hints name one. Never return a query without the
  brand when a brand is known - "Parfum prada" must stay "Prada profumo", never just "profumo"
- use the canonical brand and model spelling ("Téléphone s22 ultra" -> "Samsung Galaxy S22 Ultra")
- when no model name exists, keep the most distinctive descriptive word from the title
  ("Tomorrowland botanic ring" -> "Tomorrowland botanic ring")
- keep the material or finish whenever it separates this item from the brand's ordinary
  stock, because it sets the price: linen vs cotton, silk, leather, cashmere, 18k vs
  silver plated. "Camisa LINO Polo Ralph Lauren" must keep the linen: "Polo Ralph Lauren
  linen shirt", never just "Polo Ralph Lauren shirt"
- never repeat a word already present in the brand name ("Polo Ralph Lauren Polo camicia"
  is wrong)
- write the whole query in ENGLISH, translating the product type and any descriptive word
  ("Camisa" -> "shirt", "Parfum" -> "perfume", "anello" -> "ring", "lino" -> "linen").
  Brand and model names keep their own spelling and are never translated
- never include: size, condition, price, emoji, shipping or seller phrases, "NUOVA", "vintage" unless it is part of the model name
- 2 to 5 words; shorter is better than more specific
- brand goes in "brand" alone, never merged with the model ("Samsung", not "Samsung Galaxy")
- confidence "low" when the title does not identify a specific product well enough to find the
  same item (e.g. "Parfum prada" names no specific fragrance) - low confidence still keeps the
  brand in the query, it only marks the result as weak

Also return the query you would use in Italian, as query_local, following the same rules."""


def _cache_path(cache_dir: Path, title: str) -> Path:
    digest = hashlib.sha256(f"v{PROMPT_VERSION}:{title}".encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{digest}.json"


def rewrite_title(
    title: str,
    cache_dir: Path,
    client=None,
    brand: str | None = None,
    category: str | None = None,
    description: str | None = None,
) -> dict:
    """Return the parsed identity for `title`; falls back to the raw title on any failure."""
    fallback = {"brand": brand, "model": None, "product_type": None, "query": title,
                "query_local": title, "confidence": "low", "source": "fallback"}
    title = (title or "").strip()
    if not title:
        return fallback

    description = (description or "").strip()[:500]
    path = _cache_path(cache_dir, f"{title}\n{description}")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOG.warning("Bad title cache entry %s", path)

    if client is None:
        import openai

        if not os.getenv("OPENAI_API_KEY"):
            return fallback
        client = openai.OpenAI()

    hints = []
    if brand:
        hints.append(f"catalog brand field: {brand}")
    if category:
        hints.append(f"category: {category}")
    user = f"title: {title}"
    if hints:
        user += f"\n({'; '.join(hints)})"
    if description:
        user += f"\ndescription: {description}"

    # gpt-5 models reject any temperature but their default, so only pin it where allowed.
    options = {} if MODEL.startswith("gpt-5") else {"temperature": 0}
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "listing_identity", "strict": True, "schema": SCHEMA}},
            **options,
        )
        parsed = json.loads(response.choices[0].message.content)
    except Exception:
        LOG.exception("Title rewrite failed for %r", title)
        return fallback

    parsed["source"] = MODEL
    parsed["query"] = (parsed.get("query") or title).strip() or title
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return parsed
