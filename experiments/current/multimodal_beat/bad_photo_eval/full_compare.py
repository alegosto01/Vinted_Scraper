"""Full-data comparison (approach B) for the bad-photo deal monitor.

Approach A (match_vinted_products) compares a target against catalog rows: title
text plus the single catalog thumbnail. This module compares the *full* item
pages instead - every photo of every item, plus description, brand and colour -
and renders a report showing where the two approaches disagree.
"""
from __future__ import annotations

import html
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from match_vinted_products import (
    ImageResolver,
    _embed_images,
    _embed_titles,
    deterministic_ranks,
)

LOG = logging.getLogger("full_compare")


@dataclass(frozen=True)
class FullSplitConfig:
    photo_min: float = 0.60
    title_min: float = 0.75
    combined_min: float = 0.68
    photo_weight: float = 0.45
    title_weight: float = 0.35
    desc_weight: float = 0.20
    max_photos: int = 8


def _text(value: object) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()


def _photo_similarities(
    target: dict,
    candidates: list[dict],
    encoder,
    model_name: str,
    cache_dir: Path,
    download_dir: Path,
    max_photos: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    """Best and mean-best cosine over every target photo x candidate photo pair."""
    resolver = ImageResolver(download_dir)
    ids: list[str] = []
    urls: list[str] = []
    owners: list[str] = []
    for item in [target, *candidates]:
        item_id = str(item["item_id"])
        for index, url in enumerate((item.get("photo_urls") or [])[:max_photos]):
            ids.append(f"{item_id}#{index}")
            urls.append(url)
            owners.append(item_id)
    vectors, _, _, computed = _embed_images(
        ids, urls, encoder, model_name, cache_dir, batch_size=8, resolver=resolver
    )
    LOG.info("Embedded %d photos (%d newly computed)", len(urls), computed)

    grouped: dict[str, list[np.ndarray]] = {}
    for owner, vector in zip(owners, vectors):
        if vector is not None:
            grouped.setdefault(owner, []).append(vector)

    target_vectors = grouped.get(str(target["item_id"]), [])
    best: dict[str, float] = {}
    mean_best: dict[str, float] = {}
    counts: dict[str, int] = {}
    if not target_vectors:
        LOG.warning("No usable target photos for item %s", target["item_id"])
        return best, mean_best, counts
    target_matrix = np.vstack(target_vectors)
    for candidate in candidates:
        item_id = str(candidate["item_id"])
        candidate_vectors = grouped.get(item_id, [])
        counts[item_id] = len(candidate_vectors)
        if not candidate_vectors:
            continue
        similarity = target_matrix @ np.vstack(candidate_vectors).T
        best[item_id] = float(similarity.max())
        mean_best[item_id] = float(similarity.max(axis=1).mean())
    return best, mean_best, counts


def _text_similarities(
    target: dict,
    candidates: list[dict],
    field: str,
    encoder,
    model_name: str,
    cache_dir: Path,
) -> dict[str, float]:
    target_text = _text(target.get(field))
    if not target_text:
        return {}
    usable = [c for c in candidates if _text(c.get(field))]
    if not usable:
        return {}
    target_vector = _embed_titles(
        [f"{target['item_id']}:{field}"], [target_text], "query", encoder, model_name, cache_dir, 16
    )[0][0]
    vectors, _ = _embed_titles(
        [f"{c['item_id']}:{field}" for c in usable],
        [_text(c.get(field)) for c in usable],
        "passage",
        encoder,
        model_name,
        cache_dir,
        16,
    )
    return {
        str(candidate["item_id"]): float(np.dot(target_vector, vector))
        for candidate, vector in zip(usable, vectors)
    }


def score_full(
    target: dict,
    candidates: list[dict],
    title_encoder,
    image_encoder,
    title_model: str,
    image_model: str,
    cache_dir: Path,
    download_dir: Path,
    config: FullSplitConfig = FullSplitConfig(),
) -> pd.DataFrame:
    """Score candidates against the target using every field of the full item pages."""
    if not candidates:
        return pd.DataFrame()
    photo_best, photo_mean, photo_counts = _photo_similarities(
        target, candidates, image_encoder, image_model, cache_dir, download_dir, config.max_photos
    )
    title_sims = _text_similarities(target, candidates, "title", title_encoder, title_model, cache_dir)
    desc_sims = _text_similarities(target, candidates, "description", title_encoder, title_model, cache_dir)

    target_brand = _text(target.get("brand")).casefold()
    target_price = float(target.get("price") or 0) or np.nan

    records = []
    for candidate in candidates:
        item_id = str(candidate["item_id"])
        photo = photo_best.get(item_id, np.nan)
        title = title_sims.get(item_id, np.nan)
        description = desc_sims.get(item_id, np.nan)
        brand = _text(candidate.get("brand")).casefold()
        brand_match = None if not brand or not target_brand else brand == target_brand
        price = float(candidate.get("price") or 0) or np.nan

        weights = {"photo": config.photo_weight, "title": config.title_weight, "desc": config.desc_weight}
        parts = {"photo": photo, "title": title, "desc": description}
        usable = {key: value for key, value in parts.items() if np.isfinite(value)}
        total_weight = sum(weights[key] for key in usable)
        combined = (
            sum(weights[key] * value for key, value in usable.items()) / total_weight
            if total_weight
            else np.nan
        )

        failures = []
        if not np.isfinite(photo) or photo < config.photo_min:
            failures.append(f"photo<{config.photo_min}")
        if not np.isfinite(title) or title < config.title_min:
            failures.append(f"title<{config.title_min}")
        if not np.isfinite(combined) or combined < config.combined_min:
            failures.append(f"combined<{config.combined_min}")
        if brand_match is False:
            failures.append("brand mismatch")
        records.append(
            {
                "candidate_item_id": item_id,
                "candidate_title": _text(candidate.get("title")),
                "listing_url": f"https://www.vinted.it/items/{item_id}",
                "photo_similarity": photo,
                "photo_mean_similarity": photo_mean.get(item_id, np.nan),
                "photo_count": photo_counts.get(item_id, 0),
                "title_similarity": title,
                "description_similarity": description,
                "brand": _text(candidate.get("brand")),
                "brand_match": brand_match,
                "price": price,
                "price_ratio": price / target_price if np.isfinite(price) and np.isfinite(target_price) else np.nan,
                "condition": _text(candidate.get("condition")),
                "primary_image": (candidate.get("photo_urls") or [""])[0],
                "combined_score": combined,
                "decision": "non_kept" if failures else "kept",
                "reason": "; ".join(failures) if failures else "passed every full-data floor",
            }
        )

    frame = pd.DataFrame(records)
    frame["combined_rank"] = deterministic_ranks(frame["combined_score"], frame["candidate_item_id"])
    shutil.rmtree(download_dir, ignore_errors=True)
    return frame.sort_values("combined_rank", na_position="last").reset_index(drop=True)


def _fmt(value: object, digits: int = 3) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def _verdict_block(analysis: dict) -> str:
    median = "—" if analysis["median"] is None else f"€{analysis['median']:.2f}"
    discount = "—" if analysis["discount_pct"] is None else f"{analysis['discount_pct']:+.1f}%"
    return (f"<div class='verdict'><h3>{html.escape(analysis['verdict'])}</h3>"
            f"<p>{analysis['count']} comparable listings · median {median} · "
            f"target vs median {discount}</p></div>")


def write_full_report(
    path: Path,
    target: dict,
    musiq: float,
    rows: pd.DataFrame,
    analysis: dict,
    max_rejected: int = 40,
) -> None:
    """One report: the comparables that count, then everything rejected and why."""
    rows = rows.copy()
    if "decision" not in rows:
        rows["decision"] = "non_kept"
    kept = rows[rows["decision"].eq("kept")]
    rejected = rows[~rows["decision"].eq("kept")]

    def card(row) -> str:
        photos = int(row.get("photo_count", 0) or 0)
        price = row.get("price")
        price_text = "—" if price is None or pd.isna(price) else f"€{float(price):.2f}"
        size = row.get("size_or_capacity")
        size_text = "" if size is None or pd.isna(size) else f" · {html.escape(str(size))}"
        scores = (f"photos {photos} · best {_fmt(row.get('photo_similarity'))}"
                  f" · title {_fmt(row.get('title_similarity'))}"
                  f" · desc {_fmt(row.get('description_similarity'))}")
        return f"""<article>
<img src="{html.escape(str(row.get('primary_image') or ''))}" loading="lazy">
<h3>{html.escape(str(row.get('candidate_title') or ''))}</h3>
<p><b>{price_text}</b>{size_text}</p>
<p class="reason">{html.escape(str(row.get('reason') or ''))}</p>
<code>{scores}</code>
<a href="https://www.vinted.it/items/{html.escape(str(row['candidate_item_id']))}" target="_blank" rel="noopener">Open on Vinted</a>
</article>"""

    def grid(frame, limit=None) -> str:
        chosen = frame.head(limit) if limit else frame
        return f"<div class='grid'>{''.join(card(row) for _, row in chosen.iterrows())}</div>"

    document = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(target['Title']))} comparison</title>
<style>
body{{font:15px system-ui;margin:14px;background:#f4f4f4;color:#222}}a{{color:#0645ad}}
.target,article,.verdict{{background:white;border:1px solid #ddd;border-radius:10px;padding:12px}}
.target img,article img{{width:100%;height:240px;object-fit:contain;background:#eee;border-radius:7px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}}
.verdict{{margin:10px 0}}.verdict h3{{margin:0 0 6px;font-size:15px}}
.reason{{color:#666;font-size:13px}}
code{{display:block;font-size:12px;overflow-wrap:anywhere;color:#555}}
h2{{font-size:17px;margin-top:22px}}h2 small{{font-weight:400;color:#666}}
</style>
<h1>{html.escape(str(target['Title']))}</h1>
<section class="target"><img src="{html.escape(str(target['Images']))}">
<p><b>Target €{float(target['Price']):.2f}</b> · {html.escape(str(target['Condition']))}</p>
<p>MUSIQ {musiq:.1f} (0-100, lower is worse)</p>
<p><a href="{html.escape(str(target['Link']))}" target="_blank" rel="noopener">Open target on Vinted</a></p>
</section>
{_verdict_block(analysis)}
<p class="reason">Candidates come from a search built from this listing's own title and
description. Every photo and the description of both sides are compared, then each candidate
is judged as a price comparable. Asking prices, not sold prices.</p>
<h2>Comparable <small>({len(kept)})</small></h2>{grid(kept)}
<h2>Rejected <small>({len(rejected)}, showing {min(len(rejected), max_rejected)})</small></h2>{grid(rejected, max_rejected)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
