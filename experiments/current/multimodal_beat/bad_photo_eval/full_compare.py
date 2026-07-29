"""Photo comparison over the full item pages, plus the report.

Text is judged by the model in spec_compare, not by embeddings: a cosine over
two listing descriptions sat in a 0.04 band for every candidate and separated
nothing. What embeddings are good for here is images - a continuous score that
orders candidates - so this module compares every photo of the target against
every photo of the candidates that already passed the text verdict.
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
    """Photo similarity ranks candidates; the spec verdict decides them."""
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


def score_images(
    target: dict,
    candidates: list[dict],
    image_encoder,
    image_model: str,
    cache_dir: Path,
    download_dir: Path,
    config: FullSplitConfig = FullSplitConfig(),
) -> pd.DataFrame:
    """Best and mean-best photo similarity for candidates that already passed the text verdict."""
    if not candidates:
        return pd.DataFrame()
    photo_best, photo_mean, photo_counts = _photo_similarities(
        target, candidates, image_encoder, image_model, cache_dir, download_dir, config.max_photos
    )
    records = []
    for candidate in candidates:
        item_id = str(candidate["item_id"])
        records.append({
            "candidate_item_id": item_id,
            "photo_similarity": photo_best.get(item_id, np.nan),
            "photo_mean_similarity": photo_mean.get(item_id, np.nan),
            "photo_count": photo_counts.get(item_id, 0),
        })
    frame = pd.DataFrame(records)
    frame["photo_rank"] = deterministic_ranks(frame["photo_similarity"], frame["candidate_item_id"])
    shutil.rmtree(download_dir, ignore_errors=True)
    return frame


def candidate_frame(target: dict, candidates: list[dict]) -> pd.DataFrame:
    """The listing facts every candidate carries, before any judgement."""
    target_price = float(target.get("price") or 0) or np.nan
    rows = []
    for candidate in candidates:
        price = float(candidate.get("price") or 0) or np.nan
        rows.append({
            "candidate_item_id": str(candidate["item_id"]),
            "candidate_title": _text(candidate.get("title")),
            "brand": _text(candidate.get("brand")),
            "price": price,
            "price_ratio": price / target_price if np.isfinite(price) and np.isfinite(target_price) else np.nan,
            "condition": _text(candidate.get("condition")),
            "primary_image": (candidate.get("photo_urls") or [""])[0],
            "listing_url": f"https://www.vinted.it/items/{candidate['item_id']}",
        })
    return pd.DataFrame(rows)


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
        # Rejected candidates never reach the photo pass, so this column is empty for them.
        photo_count = row.get("photo_count")
        photos = 0 if photo_count is None or pd.isna(photo_count) else int(photo_count)
        price = row.get("price")
        price_text = "—" if price is None or pd.isna(price) else f"€{float(price):.2f}"
        size = row.get("size_or_capacity")
        size_text = "" if size is None or pd.isna(size) else f" · {html.escape(str(size))}"
        scores = (f"photos {photos} · best {_fmt(row.get('photo_similarity'))}"
                  f" · mean {_fmt(row.get('photo_mean_similarity'))}")
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
