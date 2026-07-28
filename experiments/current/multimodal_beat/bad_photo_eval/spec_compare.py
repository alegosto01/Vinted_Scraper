"""Ask gpt-5-nano whether shortlisted candidates are the same product, and at what spec.

Embeddings answer "does this look similar"; they cannot answer "is this the same
model with the same storage, and is it a bundle, a replica, or an empty box".
Those are what poison a price median: one 128GB phone or one empty box sets the
comparable price for a 512GB target.

Only the shortlist reaches this module, and the whole shortlist goes in one call,
so a comparison costs one request rather than one per candidate.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

from config.project_config import settings  # noqa: E402,F401  (imports load_dotenv)

LOG = logging.getLogger("spec_compare")
MODEL = "gpt-5-nano"
PROMPT_VERSION = 2
MAX_DESCRIPTION = 600
BATCH = 12

RELATIONS = ["same", "lower", "higher", "unknown"]
DISQUALIFIERS = ["none", "for_parts", "replica", "bundle", "empty_box_or_accessory",
                 "wrong_variant", "miniature_or_sample"]

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "same_product", "size_or_capacity", "full_size_equivalent",
                             "spec_relation", "disqualifier", "note"],
                "properties": {
                    "id": {"type": "string"},
                    "same_product": {"type": "boolean"},
                    "size_or_capacity": {
                        "type": ["string", "null"],
                        "description": "the size/capacity as written, e.g. '90 ml', '256GB', '18k', 'taglia M'",
                    },
                    "full_size_equivalent": {
                        "type": "boolean",
                        "description": "false for miniatures, samples, travel or refill formats",
                    },
                    "spec_relation": {"type": "string", "enum": RELATIONS},
                    "disqualifier": {"type": "string", "enum": DISQUALIFIERS},
                    "note": {"type": "string", "description": "at most 12 words, what differs"},
                },
            },
        }
    },
}

PROMPT = """You compare second-hand listings to decide which ones are valid price comparables.

You get one TARGET listing and several CANDIDATE listings (title, brand, price, description).
For each candidate return:

- same_product: true only if a buyer wanting the target would accept this listing instead.
  A different generation, a different fragrance or line, or a different format is NOT the
  same product, even when the brand matches.
- size_or_capacity: the size exactly as the listing writes it ("90 ml", "256GB", "18k",
  "taglia M"), or null when the listing never states one.
- full_size_equivalent: false whenever the listing is a miniature, sample, tester, travel
  format, refill, or any reduced-size version of the product. A 7 ml miniature of the right
  fragrance is NOT full size. Only true for the normal retail article.
- spec_relation: how that size compares to the target ("same", "lower", "higher", "unknown").
  Use "unknown" only when neither listing states a size.
- disqualifier: why this listing must not set a comparable price, if any:
  "for_parts" (broken/spares), "replica" (fake/inspired/dupe), "bundle" or "lot" (several items
  in one listing), "empty_box_or_accessory" (box, case, strap, receipt only - not the product),
  "miniature_or_sample" (miniature, sample, tester, travel size, refill),
  "wrong_variant" (different variant, e.g. kids size, EDT instead of parfum), otherwise "none".
- note: at most 12 words naming the actual difference, empty string if identical.

A listing far cheaper than the target is usually a miniature, a sample, an empty box or a
fake - say so through the disqualifier rather than treating it as a cheap comparable.

Judge from the title and the description only. Do not guess beyond what they say."""


def _clip(value: object, limit: int = MAX_DESCRIPTION) -> str:
    text = "" if value is None else str(value).strip().replace("\n", " ")
    return text[:limit]


def _listing_block(item: dict, prefix: str = "") -> str:
    return (f"{prefix}id: {item.get('item_id')}\n"
            f"{prefix}title: {_clip(item.get('title'), 160)}\n"
            f"{prefix}brand: {item.get('brand')} | price: EUR {item.get('price')}\n"
            f"{prefix}description: {_clip(item.get('description'))}")


def _cache_path(cache_dir: Path, target_id: str, ids: list[str]) -> Path:
    digest = hashlib.sha256(f"v{PROMPT_VERSION}:{target_id}:{','.join(sorted(ids))}".encode()).hexdigest()[:24]
    return cache_dir / f"{target_id}_{digest}.json"


def _ask(client, target: dict, batch: list[dict]) -> list[dict]:
    user = (f"TARGET\n{_listing_block(target)}\n\nCANDIDATES\n"
            + "\n\n".join(_listing_block(item) for item in batch))
    options = {} if MODEL.startswith("gpt-5") else {"temperature": 0}
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": user}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "comparables", "strict": True, "schema": SCHEMA}},
        **options,
    )
    return json.loads(response.choices[0].message.content)["verdicts"]


def compare_specs(target: dict, candidates: list[dict], cache_dir: Path, client=None) -> pd.DataFrame:
    """One row per candidate: same_product, spec_relation, disqualifier, note.

    Returns an empty frame when the model is unavailable, so callers can fall back
    to the embedding-only decision.
    """
    columns = ["candidate_item_id", "same_product", "size_or_capacity", "full_size_equivalent",
               "spec_relation", "disqualifier", "note"]
    if not candidates:
        return pd.DataFrame(columns=columns)

    ids = [str(item["item_id"]) for item in candidates]
    path = _cache_path(Path(cache_dir), str(target["item_id"]), ids)
    if path.exists():
        try:
            return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))[columns]
        except Exception:
            LOG.warning("Bad spec cache entry %s", path)

    if client is None:
        import openai

        if not os.getenv("OPENAI_API_KEY"):
            LOG.warning("No OPENAI_API_KEY; skipping spec comparison")
            return pd.DataFrame(columns=columns)
        client = openai.OpenAI()

    verdicts: list[dict] = []
    for start in range(0, len(candidates), BATCH):
        batch = candidates[start:start + BATCH]
        try:
            verdicts.extend(_ask(client, target, batch))
        except Exception:
            LOG.exception("Spec batch failed; retrying candidates individually")
            for item in batch:
                try:
                    verdicts.extend(_ask(client, target, [item]))
                except Exception:
                    LOG.exception("Spec comparison failed for %s", item.get("item_id"))

    if not verdicts:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(verdicts).rename(columns={"id": "candidate_item_id"})
    frame["candidate_item_id"] = frame["candidate_item_id"].astype(str)
    frame = frame[frame["candidate_item_id"].isin(ids)].drop_duplicates("candidate_item_id")
    for column in columns:
        if column not in frame:
            frame[column] = None
    frame = frame[columns]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(frame.to_json(orient="records", force_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return frame


def usable_comparables(rows: pd.DataFrame) -> pd.DataFrame:
    """Same product, full-size, same-or-unstated spec, no disqualifier."""
    if rows.empty or "same_product" not in rows:
        return rows.iloc[0:0]
    return rows[
        rows["same_product"].fillna(False).astype(bool)
        & rows["full_size_equivalent"].fillna(True).astype(bool)
        & rows["spec_relation"].fillna("unknown").isin(["same", "unknown"])
        & rows["disqualifier"].fillna("none").eq("none")
    ]


def comparable_prices(rows: pd.DataFrame) -> pd.Series:
    usable = usable_comparables(rows)
    return pd.to_numeric(usable["price"], errors="coerce") if "price" in usable else pd.Series(dtype=float)
