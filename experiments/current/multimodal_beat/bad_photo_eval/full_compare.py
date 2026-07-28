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


def _verdict_block(name: str, analysis: dict) -> str:
    median = "—" if analysis["median"] is None else f"€{analysis['median']:.2f}"
    discount = "—" if analysis["discount_pct"] is None else f"{analysis['discount_pct']:+.1f}%"
    return (
        f"<div class='verdict'><h3>{html.escape(name)}</h3>"
        f"<p><b>{html.escape(analysis['verdict'])}</b></p>"
        f"<p>kept {analysis['count']} · median {median} · discount {discount}</p></div>"
    )


def write_dual_report(
    path: Path,
    target: dict,
    musiq: float,
    rows_a: pd.DataFrame,
    rows_b: pd.DataFrame,
    analysis_a: dict,
    analysis_b: dict,
    max_rejected: int = 24,
) -> None:
    """Rewrite the mobile report so both approaches' kept/non-kept sets are visible."""
    a = rows_a.set_index(rows_a["candidate_item_id"].astype(str))
    b = rows_b.set_index(rows_b["candidate_item_id"].astype(str))
    item_ids = list(dict.fromkeys([*a.index, *b.index]))

    def card(item_id: str) -> str:
        row_a = a.loc[item_id] if item_id in a.index else None
        row_b = b.loc[item_id] if item_id in b.index else None
        source = row_b if row_b is not None else row_a
        title = str(source.get("candidate_title", ""))
        # Only the full pass carries photos and price; catalog-only rows fall back to blanks.
        image = str(source.get("primary_image") or source.get("Images") or "")
        price = source.get("price", source.get("Price", None))
        price_text = "—" if price is None or pd.isna(price) else f"€{float(price):.2f}"
        photos = 0 if row_b is None else int(row_b.get("photo_count", 0) or 0)
        line_a = (
            "not fetched by catalog pass"
            if row_a is None
            else f"title {_fmt(row_a['title_similarity'])} · image {_fmt(row_a['image_similarity'])} · combined {_fmt(row_a['combined_score'])}"
        )
        line_b = (
            "full page not fetched"
            if row_b is None
            else (
                f"photos {photos} · best {_fmt(row_b['photo_similarity'])} · mean {_fmt(row_b['photo_mean_similarity'])}"
                f" · title {_fmt(row_b['title_similarity'])} · desc {_fmt(row_b['description_similarity'])}"
                f" · combined {_fmt(row_b['combined_score'])}"
            )
        )
        brand = "" if row_b is None or not row_b.get("brand") else f" · {html.escape(str(row_b['brand']))}"
        return f"""<article>
<img src="{html.escape(image)}" loading="lazy">
<h3>{html.escape(title)}</h3>
<p><b>{price_text}</b>{brand}</p>
<code>catalog: {line_a}</code>
<code>full: {line_b}</code>
<a href="https://www.vinted.it/items/{html.escape(item_id)}" target="_blank" rel="noopener">Open on Vinted</a>
</article>"""

    buckets: dict[str, list[str]] = {"both": [], "a_only": [], "b_only": [], "neither": []}
    for item_id in item_ids:
        kept_a = item_id in a.index and str(a.loc[item_id, "decision"]) == "kept"
        kept_b = item_id in b.index and str(b.loc[item_id, "decision"]) == "kept"
        key = "both" if kept_a and kept_b else "a_only" if kept_a else "b_only" if kept_b else "neither"
        buckets[key].append(item_id)

    def grid(key: str, limit: int | None = None) -> str:
        chosen = buckets[key][:limit] if limit else buckets[key]
        return f"<div class='grid'>{''.join(card(item_id) for item_id in chosen)}</div>" or ""

    document = f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(target['Title']))} comparison</title>
<style>
body{{font:15px system-ui;margin:14px;background:#f4f4f4;color:#222}}a{{color:#0645ad}}
.target,article,.verdict{{background:white;border:1px solid #ddd;border-radius:10px;padding:12px}}
.target img,article img{{width:100%;height:240px;object-fit:contain;background:#eee;border-radius:7px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}}
.verdicts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin:10px 0}}
.verdict h3{{margin:0 0 6px;font-size:15px}}
code{{display:block;font-size:12px;overflow-wrap:anywhere}}
.warning{{background:#fff3cd;padding:12px;border-radius:8px}}
h2{{font-size:17px;margin-top:22px}}h2 small{{font-weight:400;color:#666}}
</style>
<h1>{html.escape(str(target['Title']))}</h1>
<section class="target"><img src="{html.escape(str(target['Images']))}">
<p><b>Target €{float(target['Price']):.2f}</b> · {html.escape(str(target['Condition']))}</p>
<p>MUSIQ {musiq:.1f} (0-100, lower is worse)</p>
<p><a href="{html.escape(str(target['Link']))}" target="_blank" rel="noopener">Open target on Vinted</a></p>
</section>
<div class="verdicts">{_verdict_block('Catalog data (title + 1 photo)', analysis_a)}{_verdict_block('Full data (all photos + description)', analysis_b)}</div>
<p class="warning">Catalog pass compares the listing title and the single catalog thumbnail.
Full pass compares every photo of both items, the description, and the brand.
Thresholds are provisional, not probabilities. Asking prices, not sold prices.</p>
<h2>Kept by both <small>({len(buckets['both'])})</small></h2>{grid('both')}
<h2>Kept only by catalog data <small>({len(buckets['a_only'])})</small></h2>{grid('a_only')}
<h2>Kept only by full data <small>({len(buckets['b_only'])})</small></h2>{grid('b_only')}
<h2>Rejected by both <small>({len(buckets['neither'])}, showing {min(len(buckets['neither']), max_rejected)})</small></h2>{grid('neither', max_rejected)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
