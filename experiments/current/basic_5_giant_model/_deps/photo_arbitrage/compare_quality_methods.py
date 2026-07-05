#!/usr/bin/env python3
from __future__ import annotations

import argparse
from html import escape
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.current.basic_5_giant_model._deps.photo_arbitrage.dataset import build_candidate_dataset
from experiments.current.basic_5_giant_model._deps.photo_arbitrage.features import add_photo_features
from experiments.current.basic_5_giant_model._deps.photo_arbitrage.paths import CANDIDATES_DIR, FEATURES_DIR, REPORTS_DIR, assert_photo_path, ensure_experiment_dirs, run_id, write_csv, write_manifest
from experiments.current.basic_5_giant_model._deps.photo_arbitrage.quality_methods import (
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_DINO_MODEL,
    DEFAULT_FASHIONCLIP_MODEL,
    DEFAULT_PYIQA_MODEL,
    MethodConfig,
    add_quality_method_scores,
    normalize_methods,
)


FRONT_COLUMNS = [
    "SearchName",
    "Title",
    "Price",
    "Brand",
    "Link",
    "Dataid",
    "LocalPrimaryImagePath",
    "SimpleBadPhotoScore",
    "PyiqaQualityScore",
    "PyiqaBadPhotoScore",
    "PyiqaStatus",
    "AestheticGoodScore",
    "AestheticBadPhotoScore",
    "AestheticLabel",
    "AestheticStatus",
    "FashionClipGoodScore",
    "FashionClipBadScore",
    "FashionClipBadPhotoScore",
    "FashionClipScoreMargin",
    "FashionClipPromptSet",
    "FashionClipStatus",
    "DinoEmbedding",
    "DinoEmbeddingDim",
    "DinoOutlierScore",
    "DinoStatus",
    "CombinedBadPhotoScore",
    "QualityMethodStatus",
    "manual_label",
    "manual_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare several local/pretrained photo-quality scoring methods.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all-searches", action="store_true")
    group.add_argument("--search", action="append")
    parser.add_argument("--candidates", default=str(CANDIDATES_DIR / "latest_candidates.csv"))
    parser.add_argument("--limit-per-search", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--require-local-image", action="store_true", help="Keep only rows with an existing local primary image.")
    parser.add_argument("--methods", default="all", help="Comma-separated methods: simple,pyiqa,aesthetic,fashionclip,dino,all")
    parser.add_argument("--pyiqa-model", default=DEFAULT_PYIQA_MODEL)
    parser.add_argument("--aesthetic-model", default=DEFAULT_AESTHETIC_MODEL)
    parser.add_argument("--fashionclip-model", default=DEFAULT_FASHIONCLIP_MODEL)
    parser.add_argument(
        "--allow-fashionclip-downloads",
        action="store_true",
        help="Allow Transformers to fetch the FashionCLIP model if it is not already cached locally.",
    )
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--max-images-per-item", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--fashionclip-pseudo-threshold", type=float, default=0.85)
    parser.add_argument("--fashionclip-pseudo-top-n", type=int, default=500)
    return parser.parse_args()


def load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    if args.all_searches or args.search:
        candidates = build_candidate_dataset(
            all_searches=bool(args.all_searches),
            searches=args.search,
            limit_per_search=args.limit_per_search,
        )
        write_csv(candidates, CANDIDATES_DIR / "latest_candidates.csv")
    else:
        path = Path(args.candidates)
        if not path.exists():
            raise FileNotFoundError(f"Candidate file not found: {path}")
        candidates = pd.read_csv(path, low_memory=False)
    if args.require_local_image and "LocalPrimaryImagePath" in candidates.columns:
        candidates = candidates[
            candidates["LocalPrimaryImagePath"].fillna("").astype(str).map(lambda value: bool(value and Path(value).exists()))
        ].copy()
    if args.max_items is not None:
        candidates = candidates.head(int(args.max_items)).copy()
    return candidates


def reorder_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = [col for col in FRONT_COLUMNS if col in frame.columns]
    rest = [col for col in frame.columns if col not in ordered]
    return frame[ordered + rest]


def summarize_quality_status_counts(scored: pd.DataFrame) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for column in ("PyiqaStatus", "AestheticStatus", "FashionClipStatus", "DinoStatus"):
        if column not in scored.columns:
            continue
        values = scored[column].fillna("").astype(str)
        values = values[values != ""]
        if values.empty:
            continue
        counts = values.value_counts(dropna=False)
        summary[column] = {str(label): int(count) for label, count in counts.items()}
    return summary


def build_fashionclip_pseudo_label_review(scored: pd.DataFrame, *, threshold: float, top_n: int) -> pd.DataFrame:
    required = {"FashionClipGoodScore", "FashionClipBadScore", "FashionClipStatus"}
    if not required.issubset(scored.columns):
        return pd.DataFrame()
    work = scored.copy()
    good = pd.to_numeric(work["FashionClipGoodScore"], errors="coerce")
    bad = pd.to_numeric(work["FashionClipBadScore"], errors="coerce")
    ok = work["FashionClipStatus"].fillna("").astype(str).str.startswith("ok")
    confident = ok & ((good >= threshold) | (bad >= threshold))
    review = work[confident].copy()
    if review.empty:
        return review
    review_good = pd.to_numeric(review["FashionClipGoodScore"], errors="coerce").fillna(-1.0)
    review_bad = pd.to_numeric(review["FashionClipBadScore"], errors="coerce").fillna(-1.0)
    review["FashionClipPseudoLabel"] = [
        "photo_quality_bad" if bad_score >= good_score else "photo_quality_good"
        for good_score, bad_score in zip(review_good, review_bad)
    ]
    review["FashionClipPseudoConfidence"] = np.maximum(review_good.to_numpy(), review_bad.to_numpy())
    for column in ("FashionClipBadPhotoScore", "CombinedBadPhotoScore"):
        if column not in review.columns:
            review[column] = np.nan
    if "manual_label" not in review.columns:
        review["manual_label"] = ""
    if "manual_notes" not in review.columns:
        review["manual_notes"] = ""
    review = review.sort_values(
        ["FashionClipPseudoConfidence", "FashionClipBadPhotoScore", "CombinedBadPhotoScore"],
        ascending=False,
        kind="stable",
    )
    if top_n:
        review = review.head(int(top_n)).copy()
    front = [
        "FashionClipPseudoLabel",
        "FashionClipPseudoConfidence",
        "manual_label",
        "manual_notes",
        "SearchName",
        "Title",
        "Price",
        "Brand",
        "Link",
        "Dataid",
        "LocalPrimaryImagePath",
        "FashionClipGoodScore",
        "FashionClipBadScore",
        "FashionClipBadPhotoScore",
        "FashionClipScoreMargin",
        "FashionClipPromptSet",
        "FashionClipStatus",
        "CombinedBadPhotoScore",
        "QualityMethodStatus",
    ]
    ordered = [col for col in front if col in review.columns]
    rest = [col for col in review.columns if col not in ordered]
    return review[ordered + rest]


def review_key_part(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value or "").strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def fashionclip_review_key(row: pd.Series | dict) -> str:
    search = review_key_part(row.get("SearchName", "")).lower()
    dataid = review_key_part(row.get("Dataid", ""))
    if search and dataid:
        return f"item:{search}:{dataid}"
    link = review_key_part(row.get("Link", ""))
    if link:
        return f"link:{link}"
    image_path = review_key_part(row.get("LocalPrimaryImagePath", ""))
    if image_path:
        return f"image:{image_path}"
    title = review_key_part(row.get("Title", ""))
    price = review_key_part(row.get("Price", ""))
    if title or price:
        return f"title_price:{title}:{price}"
    return ""


def restore_existing_review_annotations(review: pd.DataFrame, existing_path: Path) -> tuple[pd.DataFrame, int]:
    out = review.copy()
    if out.empty or not existing_path.exists():
        return out, 0
    try:
        existing = pd.read_csv(existing_path, low_memory=False)
    except Exception:
        return out, 0
    if existing.empty:
        return out, 0
    out["_ReviewKey"] = out.apply(fashionclip_review_key, axis=1)
    existing["_ReviewKey"] = existing.apply(fashionclip_review_key, axis=1)
    restored = pd.Series(False, index=out.index)
    for column in ("manual_label", "manual_notes"):
        if column not in out.columns:
            out[column] = ""
        if column not in existing.columns:
            continue
        mapping: dict[str, str] = {}
        for key, value in zip(existing["_ReviewKey"], existing[column]):
            text = review_key_part(value)
            if key and text:
                mapping[str(key)] = text
        if not mapping:
            continue
        current_blank = out[column].fillna("").astype(str).str.strip() == ""
        mapped = out["_ReviewKey"].map(mapping)
        mapped_has_value = mapped.fillna("").astype(str).str.strip() != ""
        fill_mask = current_blank & mapped_has_value
        out.loc[fill_mask, column] = mapped[fill_mask]
        restored = restored | fill_mask
    out = out.drop(columns=["_ReviewKey"])
    return out, int(restored.sum())


def html_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return escape(str(value))


def html_score(value: object) -> str:
    try:
        score = float(value)
    except Exception:
        return ""
    if not np.isfinite(score):
        return ""
    return f"{score:.3f}"


def local_image_uri(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    path = Path(text)
    if not path.exists():
        return ""
    return path.resolve().as_uri()


def render_fashionclip_review_html(review: pd.DataFrame, *, title: str = "FashionCLIP Pseudo-Label Review") -> str:
    label_counts = review.get("FashionClipPseudoLabel", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict()
    rows = []
    for idx, row in review.reset_index(drop=True).iterrows():
        image_src = local_image_uri(row.get("LocalPrimaryImagePath", ""))
        if image_src:
            image_html = f'<img src="{escape(image_src)}" alt="">'
        else:
            image_html = '<div class="missing-image">No local image</div>'
        link = str(row.get("Link", "") or "").strip()
        link_html = f'<a href="{escape(link)}">Open listing</a>' if link else ""
        rows.append(
            f"""
            <article class="card">
              <div class="image">{image_html}</div>
              <div class="body">
                <div class="row"><span class="idx">#{idx + 1}</span><span class="label">{html_value(row.get("FashionClipPseudoLabel", ""))}</span></div>
                <h2>{html_value(row.get("Title", ""))}</h2>
                <dl>
                  <dt>Confidence</dt><dd>{html_score(row.get("FashionClipPseudoConfidence", ""))}</dd>
                  <dt>Good</dt><dd>{html_score(row.get("FashionClipGoodScore", ""))}</dd>
                  <dt>Bad</dt><dd>{html_score(row.get("FashionClipBadScore", ""))}</dd>
                  <dt>Search</dt><dd>{html_value(row.get("SearchName", ""))}</dd>
                  <dt>Price</dt><dd>{html_value(row.get("Price", ""))}</dd>
                </dl>
                <p class="path">{html_value(row.get("LocalPrimaryImagePath", ""))}</p>
                <p class="link">{link_html}</p>
              </div>
            </article>
            """
        )
    count_text = ", ".join(f"{html_value(label)}: {count}" for label, count in label_counts.items()) or "no rows"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #1f2933; }}
    header {{ position: sticky; top: 0; z-index: 1; padding: 16px 24px; background: #fff; border-bottom: 1px solid #d8dee8; }}
    h1 {{ margin: 0 0 4px; font-size: 22px; }}
    header p {{ margin: 0; color: #52606d; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; padding: 16px; }}
    .card {{ overflow: hidden; background: #fff; border: 1px solid #d8dee8; border-radius: 8px; }}
    .image {{ display: flex; align-items: center; justify-content: center; height: 260px; background: #111827; }}
    .image img {{ max-width: 100%; max-height: 260px; object-fit: contain; }}
    .missing-image {{ color: #fff; font-weight: 600; }}
    .body {{ padding: 12px; }}
    .row {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
    .idx {{ color: #627d98; font-size: 13px; }}
    .label {{ padding: 3px 8px; border-radius: 999px; background: #e0f2fe; color: #075985; font-size: 12px; font-weight: 700; }}
    h2 {{ min-height: 42px; margin: 10px 0; font-size: 16px; line-height: 1.3; }}
    dl {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; margin: 0; font-size: 13px; }}
    dt {{ color: #627d98; }}
    dd {{ margin: 0; font-weight: 600; }}
    .path {{ overflow-wrap: anywhere; color: #829ab1; font-size: 11px; }}
    .link a {{ color: #0f766e; font-weight: 700; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(title)}</h1>
    <p>{len(review)} rows. {count_text}. Edit <code>fashionclip_pseudo_label_review_queue.csv</code> and fill <code>manual_label</code> after review.</p>
  </header>
  <main>
    {''.join(rows)}
  </main>
</body>
</html>
"""


def write_fashionclip_review_html(review: pd.DataFrame, path: Path) -> Path:
    path = assert_photo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_fashionclip_review_html(review), encoding="utf-8")
    tmp.replace(path)
    return path


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    methods = normalize_methods(args.methods)
    candidates = load_candidates(args)
    featured = add_photo_features(candidates)
    config = MethodConfig(
        methods=methods,
        pyiqa_model=args.pyiqa_model,
        aesthetic_model=args.aesthetic_model,
        fashionclip_model=args.fashionclip_model,
        fashionclip_local_files_only=not bool(args.allow_fashionclip_downloads),
        dino_model=args.dino_model,
        max_images_per_item=args.max_images_per_item,
        device=args.device,
    )
    scored = add_quality_method_scores(featured, config=config)
    if "manual_label" not in scored.columns:
        scored["manual_label"] = ""
    if "manual_notes" not in scored.columns:
        scored["manual_notes"] = ""
    scored = scored.sort_values(["CombinedBadPhotoScore", "SimpleBadPhotoScore"], ascending=False, kind="stable").reset_index(drop=True)
    scored = reorder_columns(scored)
    quality_status_counts = summarize_quality_status_counts(scored)

    stem = run_id("photo_quality_comparison")
    comparison_path = write_csv(scored, FEATURES_DIR / f"{stem}.csv")
    latest_path = write_csv(scored, FEATURES_DIR / "latest_photo_quality_comparison.csv")
    review = scored.head(int(args.top_n)).copy() if args.top_n else scored.copy()
    queue_path = write_csv(review, REPORTS_DIR / "photo_quality_comparison_review_queue.csv")
    timestamped_queue_path = write_csv(review, REPORTS_DIR / f"photo_quality_comparison_review_queue_{stem}.csv")
    fashionclip_review = build_fashionclip_pseudo_label_review(
        scored,
        threshold=float(args.fashionclip_pseudo_threshold),
        top_n=int(args.fashionclip_pseudo_top_n),
    )
    fashionclip_latest_queue = REPORTS_DIR / "fashionclip_pseudo_label_review_queue.csv"
    fashionclip_review, restored_manual_rows = restore_existing_review_annotations(fashionclip_review, fashionclip_latest_queue)
    fashionclip_queue_path = write_csv(fashionclip_review, fashionclip_latest_queue)
    timestamped_fashionclip_queue_path = write_csv(
        fashionclip_review,
        REPORTS_DIR / f"fashionclip_pseudo_label_review_queue_{stem}.csv",
    )
    fashionclip_html_path = write_fashionclip_review_html(fashionclip_review, REPORTS_DIR / "fashionclip_pseudo_label_review_queue.html")
    timestamped_fashionclip_html_path = write_fashionclip_review_html(
        fashionclip_review,
        REPORTS_DIR / f"fashionclip_pseudo_label_review_queue_{stem}.html",
    )
    manifest_path = write_manifest(
        FEATURES_DIR / f"{stem}_manifest.json",
        command=" ".join(sys.argv),
        extra={
            "candidate_rows": int(len(candidates)),
            "scored_rows": int(len(scored)),
            "require_local_image": bool(args.require_local_image),
            "methods": list(methods),
            "pyiqa_model": args.pyiqa_model,
            "aesthetic_model": args.aesthetic_model,
            "fashionclip_model": args.fashionclip_model,
            "fashionclip_local_files_only": not bool(args.allow_fashionclip_downloads),
            "fashionclip_pseudo_threshold": float(args.fashionclip_pseudo_threshold),
            "dino_model": args.dino_model,
            "quality_status_counts": quality_status_counts,
            "comparison_path": str(comparison_path),
            "latest_path": str(latest_path),
            "queue_path": str(queue_path),
            "timestamped_queue_path": str(timestamped_queue_path),
            "fashionclip_pseudo_rows": int(len(fashionclip_review)),
            "fashionclip_restored_manual_rows": int(restored_manual_rows),
            "fashionclip_queue_path": str(fashionclip_queue_path),
            "timestamped_fashionclip_queue_path": str(timestamped_fashionclip_queue_path),
            "fashionclip_html_path": str(fashionclip_html_path),
            "timestamped_fashionclip_html_path": str(timestamped_fashionclip_html_path),
        },
    )
    print(f"Compared {len(scored)} candidates with methods: {', '.join(methods)}")
    print(f"Comparison table written to {comparison_path}")
    print(f"Latest comparison table written to {latest_path}")
    print(f"Review queue written to {queue_path}")
    print(f"FashionCLIP pseudo-label review queue written to {fashionclip_queue_path} ({len(fashionclip_review)} rows)")
    if restored_manual_rows:
        print(f"Restored manual labels/notes for {restored_manual_rows} FashionCLIP review rows")
    print(f"FashionCLIP pseudo-label HTML review written to {fashionclip_html_path}")
    if "FashionClipStatus" in quality_status_counts:
        print(f"FashionCLIP status counts: {quality_status_counts['FashionClipStatus']}")
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
