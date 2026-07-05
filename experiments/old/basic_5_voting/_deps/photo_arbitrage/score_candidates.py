#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.old.basic_5_voting._deps.photo_arbitrage.dataset import build_candidate_dataset
from experiments.old.basic_5_voting._deps.photo_arbitrage.features import add_business_scores, add_photo_features
from experiments.old.basic_5_voting._deps.photo_arbitrage.modeling import load_latest_model, score_bad_photo_probability
from experiments.old.basic_5_voting._deps.photo_arbitrage.paths import (
    CANDIDATES_DIR,
    FEATURES_DIR,
    REPORTS_DIR,
    ensure_experiment_dirs,
    ensure_project_imports,
    run_id,
    write_csv,
    write_manifest,
)

REVIEW_COLUMNS = [
    "SearchName",
    "Title",
    "Price",
    "Brand",
    "Link",
    "Dataid",
    "LocalPrimaryImagePath",
    "BadPhotoProbability",
    "PhotoOpportunityScore",
    "PhotoOpportunityNotes",
    "PhotoValueSignal",
    "PhotoRiskPenalty",
    "PictureCount",
    "Condition",
    "SellerName",
    "ReviewsCount",
    "Stars",
    "PhotoModelVersion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score local candidates for photo-improvement review.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all-searches", action="store_true")
    group.add_argument("--search", action="append")
    parser.add_argument("--candidates", default=str(CANDIDATES_DIR / "latest_candidates.csv"))
    parser.add_argument("--limit-per-search", type=int, default=None)
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
        return candidates
    path = Path(args.candidates)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    candidates = load_candidates(args)
    features = add_photo_features(candidates)
    model, metadata = load_latest_model()
    probabilities, model_version = score_bad_photo_probability(features, model=model, metadata=metadata)
    scored = features.copy()
    scored["BadPhotoProbability"] = probabilities
    scored["PhotoModelVersion"] = metadata.get("model_version", model_version) if model is not None else model_version
    scored = add_business_scores(scored)
    scored = scored.sort_values(["PhotoOpportunityScore", "BadPhotoProbability"], ascending=False, kind="stable").reset_index(drop=True)

    stem = run_id("photo_scores")
    features_path = write_csv(scored, FEATURES_DIR / f"{stem}.csv")
    write_csv(scored, FEATURES_DIR / "latest_scored_candidates.csv")

    review = scored.copy()
    if args.top_n:
        review = review.head(int(args.top_n)).copy()
    review = review[[col for col in REVIEW_COLUMNS if col in review.columns] + [col for col in review.columns if col not in REVIEW_COLUMNS]]
    queue_path = write_csv(review, REPORTS_DIR / "photo_review_queue.csv")
    timestamped_queue_path = write_csv(review, REPORTS_DIR / f"photo_review_queue_{stem}.csv")
    write_manifest(
        FEATURES_DIR / f"{stem}_manifest.json",
        command=" ".join(sys.argv),
        extra={
            "candidate_rows": int(len(candidates)),
            "scored_rows": int(len(scored)),
            "model_status": metadata.get("status", "heuristic"),
            "model_version": scored["PhotoModelVersion"].iloc[0] if len(scored) else model_version,
            "features_path": str(features_path),
            "queue_path": str(queue_path),
            "timestamped_queue_path": str(timestamped_queue_path),
        },
    )
    print(f"Scored {len(scored)} candidates")
    print(f"Review queue written to {queue_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
