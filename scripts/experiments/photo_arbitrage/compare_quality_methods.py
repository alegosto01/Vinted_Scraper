#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.photo_arbitrage.dataset import build_candidate_dataset
from experiments.photo_arbitrage.features import add_photo_features
from experiments.photo_arbitrage.paths import CANDIDATES_DIR, FEATURES_DIR, REPORTS_DIR, ensure_experiment_dirs, run_id, write_csv, write_manifest
from experiments.photo_arbitrage.quality_methods import (
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_DINO_MODEL,
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
    parser.add_argument("--methods", default="all", help="Comma-separated methods: simple,pyiqa,aesthetic,dino,all")
    parser.add_argument("--pyiqa-model", default=DEFAULT_PYIQA_MODEL)
    parser.add_argument("--aesthetic-model", default=DEFAULT_AESTHETIC_MODEL)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--max-images-per-item", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--top-n", type=int, default=1000)
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

    stem = run_id("photo_quality_comparison")
    comparison_path = write_csv(scored, FEATURES_DIR / f"{stem}.csv")
    latest_path = write_csv(scored, FEATURES_DIR / "latest_photo_quality_comparison.csv")
    review = scored.head(int(args.top_n)).copy() if args.top_n else scored.copy()
    queue_path = write_csv(review, REPORTS_DIR / "photo_quality_comparison_review_queue.csv")
    timestamped_queue_path = write_csv(review, REPORTS_DIR / f"photo_quality_comparison_review_queue_{stem}.csv")
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
            "dino_model": args.dino_model,
            "comparison_path": str(comparison_path),
            "latest_path": str(latest_path),
            "queue_path": str(queue_path),
            "timestamped_queue_path": str(timestamped_queue_path),
        },
    )
    print(f"Compared {len(scored)} candidates with methods: {', '.join(methods)}")
    print(f"Comparison table written to {comparison_path}")
    print(f"Latest comparison table written to {latest_path}")
    print(f"Review queue written to {queue_path}")
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
