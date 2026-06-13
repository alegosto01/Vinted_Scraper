#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.photo_arbitrage.dataset import (
    CANDIDATE_COLUMNS,
    dedupe_candidates,
    find_cached_image_paths,
    first_nonempty,
    identity_series,
    load_csv_or_empty,
    merge_full_enrichment,
    normalize_id,
    normalize_sources,
    numeric_max,
)
from experiments.photo_arbitrage.features import add_photo_features
from experiments.photo_arbitrage.paths import (
    CANDIDATES_DIR,
    FEATURES_DIR,
    REPORTS_DIR,
    SIMPLE_SCRAPE_DIR,
    ensure_experiment_dirs,
    run_id,
    utc_now_iso,
    write_csv,
    write_manifest,
)
from experiments.photo_arbitrage.quality_methods import (
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_DINO_MODEL,
    DEFAULT_PYIQA_MODEL,
    MethodConfig,
    add_quality_method_scores,
    combine_bad_photo_scores,
    normalize_methods,
)


TARGET_SEARCHES = (
    "griffati_donna_all",
    "griffati_uomo_all",
    "gucci",
    "nike",
    "prada",
    "ps4",
)

FRONT_COLUMNS = [
    "SearchName",
    "PhotoOutcomeLabel",
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
    "DinoEmbedding",
    "DinoEmbeddingDim",
    "DinoEmbeddingNorm",
    "DinoOutlierScore",
    "DinoStatus",
    "CombinedBadPhotoScore",
    "QualityMethodStatus",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score balanced sold/unsold item sets with all photo-quality methods."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--target-searches",
        action="store_true",
        help="Use the six established searches, excluding Borse_Griffate and Scarpe_Griffate.",
    )
    group.add_argument("--search", action="append", help="Specific search folder to process. Repeatable.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for unsold sampling.")
    parser.add_argument("--max-sold-per-search", type=int, default=None, help="Smoke-test limit before unsold balancing.")
    parser.add_argument("--methods", default="all", help="Comma-separated methods: simple,pyiqa,aesthetic,dino,all")
    parser.add_argument("--pyiqa-model", default=DEFAULT_PYIQA_MODEL)
    parser.add_argument("--aesthetic-model", default=DEFAULT_AESTHETIC_MODEL)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--max-images-per-item", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--run-name", default=None, help="Optional stable run folder name.")
    parser.add_argument("--resume", action="store_true", help="Skip searches whose scored CSV already exists.")
    parser.add_argument("--batch-size", type=int, default=250, help="Rows to score before checkpointing a chunk CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Build counts only; do not score or write method outputs.")
    parser.add_argument("--top-n", type=int, default=1000)
    return parser.parse_args()


def selected_searches(args: argparse.Namespace) -> list[str]:
    if args.search:
        return list(dict.fromkeys(args.search))
    return list(TARGET_SEARCHES)


def image_urls_from_row(row: pd.Series) -> list[str]:
    urls: list[str] = []
    for value in (row.get("FullImageUrls"), row.get("PrimaryImageUrl"), row.get("Images")):
        for source in normalize_sources(value):
            if source not in urls:
                urls.append(source)
    return urls


def candidates_from_frame(search_dir: Path, frame: pd.DataFrame, *, label: str, source_file: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*CANDIDATE_COLUMNS, "PhotoOutcomeLabel", "PhotoOutcomeSourceFile"])
    base = dedupe_candidates(frame, keep="last")
    base["SearchName"] = search_dir.name
    full = load_csv_or_empty(search_dir / "full_scrape" / "items_enriched.csv")
    merged = merge_full_enrichment(base, full)
    built_at = utc_now_iso()
    rows = []
    for _, row in merged.iterrows():
        local_paths = normalize_sources(first_nonempty(row.get("LocalImagePaths"), ""))
        if not local_paths:
            local_paths = find_cached_image_paths(search_dir, row.get("Dataid"))
        primary_path = first_nonempty(row.get("LocalPrimaryImagePath"), local_paths[0] if local_paths else "")
        image_urls = image_urls_from_row(row)
        visible_hidden_total = numeric_max(row.get("VisiblePictureCount"), 0) + numeric_max(row.get("HiddenPictureCount"), 0)
        picture_count = numeric_max(row.get("PictureCount"), visible_hidden_total, len(local_paths), len(image_urls))
        item_id = normalize_id(row.get("Dataid")) or str(row.get("Link") or "").strip()
        rows.append(
            {
                "item_id": item_id,
                "SearchName": search_dir.name,
                "Title": row.get("Title", ""),
                "Brand": row.get("Brand", ""),
                "Price": row.get("Price", pd.NA),
                "Size": row.get("Size", ""),
                "Likes": row.get("Likes", pd.NA),
                "Page": row.get("Page", pd.NA),
                "SearchDate": row.get("SearchDate", ""),
                "MarketStatus": row.get("MarketStatus", ""),
                "Dataid": row.get("Dataid", ""),
                "Link": row.get("Link", ""),
                "Images": row.get("Images", ""),
                "ImageUrls": json.dumps(image_urls, ensure_ascii=True),
                "LocalPrimaryImagePath": str(primary_path or ""),
                "LocalImagePaths": json.dumps(local_paths, ensure_ascii=True),
                "PictureCount": picture_count,
                "VisiblePictureCount": row.get("VisiblePictureCount", pd.NA),
                "HiddenPictureCount": row.get("HiddenPictureCount", pd.NA),
                "Description": row.get("Description", ""),
                "Condition": row.get("Condition", ""),
                "Upload_date": row.get("Upload_date", ""),
                "Interested_count": row.get("Interested_count", pd.NA),
                "View_count": row.get("View_count", pd.NA),
                "SellerName": row.get("SellerName", ""),
                "Location": row.get("Location", ""),
                "ReviewsCount": row.get("ReviewsCount", pd.NA),
                "Stars": row.get("Stars", pd.NA),
                "CandidateBuiltAt": built_at,
                "PhotoOutcomeLabel": label,
                "PhotoOutcomeSourceFile": source_file,
            }
        )
    return pd.DataFrame(rows)


def balanced_search_candidates(search_name: str, *, seed: int = 42, max_sold: int | None = None) -> tuple[pd.DataFrame, dict]:
    search_dir = SIMPLE_SCRAPE_DIR / search_name
    sold_path = search_dir / "sold_df.csv"
    unsold_path = search_dir / "unsold_df.csv"
    sold_raw = load_csv_or_empty(sold_path)
    unsold_raw = load_csv_or_empty(unsold_path)
    sold = dedupe_candidates(sold_raw, keep="last")
    unsold = dedupe_candidates(unsold_raw, keep="last")
    if max_sold is not None:
        sold = sold.head(int(max_sold)).copy()
    sold_ids = set(identity_series(sold).astype(str))
    if not unsold.empty:
        unsold = unsold.loc[~identity_series(unsold).astype(str).isin(sold_ids)].copy()
    unsold_n = min(len(sold), len(unsold))
    if unsold_n and len(unsold) > unsold_n:
        unsold = unsold.sample(n=unsold_n, random_state=seed).reset_index(drop=True)
    else:
        unsold = unsold.head(unsold_n).copy()
    sold_candidates = candidates_from_frame(search_dir, sold, label="sold", source_file=str(sold_path))
    unsold_candidates = candidates_from_frame(search_dir, unsold, label="unsold", source_file=str(unsold_path))
    combined = pd.concat([sold_candidates, unsold_candidates], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True) if len(combined) else combined
    counts = {
        "search": search_name,
        "sold_raw_rows": int(len(sold_raw)),
        "unsold_raw_rows": int(len(unsold_raw)),
        "sold_dedup_rows": int(len(dedupe_candidates(sold_raw, keep="last"))),
        "unsold_dedup_rows": int(len(dedupe_candidates(unsold_raw, keep="last"))),
        "sold_selected_rows": int(len(sold_candidates)),
        "unsold_selected_rows": int(len(unsold_candidates)),
        "total_selected_rows": int(len(combined)),
        "local_image_rows": int(
            combined["LocalPrimaryImagePath"].fillna("").astype(str).map(lambda path: bool(path and Path(path).exists())).sum()
        )
        if "LocalPrimaryImagePath" in combined.columns
        else 0,
    }
    return combined, counts


def reorder_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = [col for col in FRONT_COLUMNS if col in frame.columns]
    rest = [col for col in frame.columns if col not in ordered]
    return frame[ordered + rest]


def summarize_scored(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = scored.groupby(["SearchName", "PhotoOutcomeLabel"], dropna=False)
    for (search, label), group in groups:
        row = {
            "SearchName": search,
            "PhotoOutcomeLabel": label,
            "Rows": int(len(group)),
            "LocalImageRows": int(
                group["LocalPrimaryImagePath"].fillna("").astype(str).map(lambda path: bool(path and Path(path).exists())).sum()
            )
            if "LocalPrimaryImagePath" in group.columns
            else 0,
            "CombinedBadPhotoScoreMean": float(pd.to_numeric(group.get("CombinedBadPhotoScore"), errors="coerce").mean()),
            "SimpleBadPhotoScoreMean": float(pd.to_numeric(group.get("SimpleBadPhotoScore"), errors="coerce").mean()),
            "PyiqaBadPhotoScoreMean": float(pd.to_numeric(group.get("PyiqaBadPhotoScore"), errors="coerce").mean()),
            "AestheticBadPhotoScoreMean": float(pd.to_numeric(group.get("AestheticBadPhotoScore"), errors="coerce").mean()),
            "DinoOutlierScoreMean": float(pd.to_numeric(group.get("DinoOutlierScore"), errors="coerce").mean()),
        }
        for status_col in ("PyiqaStatus", "AestheticStatus", "DinoStatus"):
            if status_col in group.columns:
                row[f"{status_col}Counts"] = group[status_col].fillna("").astype(str).value_counts().to_dict()
        rows.append(row)
    return pd.DataFrame(rows)


def count_existing_candidates(search_name: str, candidates: pd.DataFrame) -> dict:
    labels = candidates.get("PhotoOutcomeLabel", pd.Series(dtype=object)).fillna("").astype(str).value_counts()
    local_rows = int(
        candidates.get("LocalPrimaryImagePath", pd.Series(dtype=object))
        .fillna("")
        .astype(str)
        .map(lambda path: bool(path and Path(path).exists()))
        .sum()
    )
    return {
        "search": search_name,
        "sold_raw_rows": pd.NA,
        "unsold_raw_rows": pd.NA,
        "sold_dedup_rows": pd.NA,
        "unsold_dedup_rows": pd.NA,
        "sold_selected_rows": int(labels.get("sold", 0)),
        "unsold_selected_rows": int(labels.get("unsold", 0)),
        "total_selected_rows": int(len(candidates)),
        "local_image_rows": local_rows,
        "loaded_existing_candidates": True,
    }


def chunk_output_path(chunks_dir: Path, search: str, start: int, end: int) -> Path:
    return chunks_dir / search / f"{search}_chunk_{start:06d}_{end:06d}.csv"


def parse_embedding(value: object) -> list[float] | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    try:
        vector = [float(item) for item in parsed]
    except Exception:
        return None
    return vector


def recompute_dino_outliers(scored: pd.DataFrame) -> pd.DataFrame:
    if "DinoEmbedding" not in scored.columns:
        return scored
    out = scored.copy()
    embeddings: dict[int, list[float]] = {}
    for idx, value in out["DinoEmbedding"].items():
        vector = parse_embedding(value)
        if vector:
            embeddings[idx] = vector
    if not embeddings:
        return out
    import numpy as np

    matrix = np.asarray([embeddings[idx] for idx in embeddings], dtype=np.float32)
    center = np.median(matrix, axis=0)
    distances = np.linalg.norm(matrix - center, axis=1)
    if len(distances) > 1 and float(distances.max() - distances.min()) > 1e-9:
        outlier_scores = (distances - distances.min()) / (distances.max() - distances.min())
    else:
        outlier_scores = np.zeros_like(distances)
    for idx, score in zip(embeddings, outlier_scores):
        out.at[idx, "DinoOutlierScore"] = float(np.clip(score, 0.0, 1.0))
    out["CombinedBadPhotoScore"] = combine_bad_photo_scores(out)
    return out


def combine_chunk_outputs(search: str, chunk_paths: list[Path], scored_path: Path) -> pd.DataFrame:
    if not chunk_paths:
        scored = pd.DataFrame()
    else:
        frames = [pd.read_csv(path, low_memory=False) for path in chunk_paths if path.exists()]
        scored = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not scored.empty:
        scored = recompute_dino_outliers(scored)
        scored = scored.sort_values(
            ["CombinedBadPhotoScore", "SimpleBadPhotoScore"],
            ascending=False,
            kind="stable",
        ).reset_index(drop=True)
        scored = reorder_columns(scored)
    write_csv(scored, scored_path)
    print(f"{search}: combined {len(scored)} scored rows into {scored_path}")
    return scored


def score_in_batches(
    search: str,
    candidates: pd.DataFrame,
    *,
    config: MethodConfig,
    batch_size: int,
    chunks_dir: Path,
    scored_path: Path,
    resume: bool,
) -> tuple[pd.DataFrame, list[str]]:
    batch_size = max(int(batch_size or 0), 1)
    chunk_paths: list[Path] = []
    total = len(candidates)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        chunk_path = chunk_output_path(chunks_dir, search, start, end)
        chunk_paths.append(chunk_path)
        if resume and chunk_path.exists():
            print(f"{search}: keeping existing chunk {start}-{end}: {chunk_path}")
            continue
        chunk = candidates.iloc[start:end].copy()
        print(f"{search}: scoring rows {start + 1}-{end} of {total}")
        featured = add_photo_features(chunk)
        scored_chunk = add_quality_method_scores(featured, config=config)
        scored_chunk = reorder_columns(scored_chunk)
        write_csv(scored_chunk, chunk_path)
        print(f"{search}: wrote checkpoint chunk {chunk_path}")
    scored = combine_chunk_outputs(search, chunk_paths, scored_path)
    return scored, [str(path) for path in chunk_paths]


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    methods = normalize_methods(args.methods)
    searches = selected_searches(args)
    stem = args.run_name or run_id("sold_unsold_visuals")
    run_features_dir = FEATURES_DIR / stem
    run_candidates_dir = CANDIDATES_DIR / stem
    run_reports_dir = REPORTS_DIR / stem
    chunks_dir = run_features_dir / "chunks"
    run_features_dir.mkdir(parents=True, exist_ok=True)
    run_candidates_dir.mkdir(parents=True, exist_ok=True)
    run_reports_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    counts = []
    scored_paths = []
    candidate_paths = []
    chunk_output_paths = {}
    all_summaries = []
    config = MethodConfig(
        methods=methods,
        pyiqa_model=args.pyiqa_model,
        aesthetic_model=args.aesthetic_model,
        dino_model=args.dino_model,
        max_images_per_item=args.max_images_per_item,
        device=args.device,
    )

    for search in searches:
        scored_path = run_features_dir / f"{search}_scored.csv"
        candidate_path = run_candidates_dir / f"{search}_candidates.csv"
        if args.resume and scored_path.exists():
            scored = pd.read_csv(scored_path, low_memory=False)
            scored_paths.append(str(scored_path))
            all_summaries.append(summarize_scored(scored))
            print(f"Skipping {search}; scored output already exists: {scored_path}")
            continue

        if args.resume and candidate_path.exists():
            candidates = pd.read_csv(candidate_path, low_memory=False)
            count_row = count_existing_candidates(search, candidates)
            print(f"{search}: loaded existing candidates from {candidate_path}")
        else:
            candidates, count_row = balanced_search_candidates(search, seed=args.seed, max_sold=args.max_sold_per_search)
            write_csv(candidates, candidate_path)
        counts.append(count_row)
        candidate_paths.append(str(candidate_path))
        print(
            f"{search}: selected {count_row['sold_selected_rows']} sold + "
            f"{count_row['unsold_selected_rows']} unsold "
            f"({count_row['local_image_rows']} rows with local primary images)"
        )
        if args.dry_run:
            continue

        scored, chunk_paths = score_in_batches(
            search,
            candidates,
            config=config,
            batch_size=args.batch_size,
            chunks_dir=chunks_dir,
            scored_path=scored_path,
            resume=args.resume,
        )
        chunk_output_paths[search] = chunk_paths
        scored_paths.append(str(scored_path))
        all_summaries.append(summarize_scored(scored))
        print(f"{search}: scored output written to {scored_path}")

    counts_frame = pd.DataFrame(counts)
    counts_path = write_csv(counts_frame, run_reports_dir / "balanced_counts.csv")
    if all_summaries:
        summary = pd.concat(all_summaries, ignore_index=True)
    else:
        summary = pd.DataFrame()
    summary_path = write_csv(summary, run_reports_dir / "method_summary_by_search_and_label.csv")

    if scored_paths:
        scored_frames = [pd.read_csv(path, low_memory=False) for path in scored_paths]
        combined = pd.concat(scored_frames, ignore_index=True)
        combined_path = write_csv(combined, run_features_dir / "combined_scored.csv")
        review_path = write_csv(
            combined.sort_values(["CombinedBadPhotoScore"], ascending=False, kind="stable").head(int(args.top_n)),
            run_reports_dir / "top_bad_photo_review_queue.csv",
        )
    else:
        combined_path = ""
        review_path = ""

    manifest_path = write_manifest(
        run_reports_dir / "manifest.json",
        command=" ".join(sys.argv),
        extra={
            "run_name": stem,
            "searches": searches,
            "excluded_searches": ["Borse_Griffate", "Scarpe_Griffate"],
            "methods": list(methods),
            "seed": args.seed,
            "max_sold_per_search": args.max_sold_per_search,
            "batch_size": args.batch_size,
            "dry_run": bool(args.dry_run),
            "candidate_paths": candidate_paths,
            "scored_paths": scored_paths,
            "chunk_output_paths": chunk_output_paths,
            "counts_path": str(counts_path),
            "summary_path": str(summary_path),
            "combined_path": str(combined_path),
            "review_path": str(review_path),
        },
    )
    print(f"Counts written to {counts_path}")
    print(f"Summary written to {summary_path}")
    if combined_path:
        print(f"Combined scored table written to {combined_path}")
        print(f"Review queue written to {review_path}")
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
