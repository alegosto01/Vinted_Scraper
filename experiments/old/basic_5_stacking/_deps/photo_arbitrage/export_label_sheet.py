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

from experiments.old.basic_5_stacking._deps.photo_arbitrage.dataset import build_candidate_dataset
from experiments.old.basic_5_stacking._deps.photo_arbitrage.features import add_business_scores, add_photo_features, heuristic_bad_photo_probability
from experiments.old.basic_5_stacking._deps.photo_arbitrage.paths import (
    CANDIDATES_DIR,
    LABELS_DIR,
    ensure_experiment_dirs,
    ensure_project_imports,
    write_csv,
    write_manifest,
)

LABEL_COLUMNS = [
    "manual_label",
    "review_tags",
    "review_notes",
    "SearchName",
    "Title",
    "Brand",
    "Price",
    "Link",
    "Dataid",
    "item_id",
    "LocalPrimaryImagePath",
    "LocalImagePaths",
    "PictureCount",
    "BadPhotoHeuristicProbability",
    "PhotoOpportunityScore",
    "PhotoOpportunityNotes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a manual label sheet for photo quality.")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--candidates", default=str(CANDIDATES_DIR / "latest_candidates.csv"))
    parser.add_argument("--all-searches", action="store_true", help="Build candidates first if latest candidates are missing.")
    parser.add_argument("--search", action="append", help="Build these searches first if latest candidates are missing.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_or_build_candidates(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.candidates)
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    if args.all_searches or args.search:
        return build_candidate_dataset(all_searches=args.all_searches, searches=args.search)
    raise FileNotFoundError(f"Candidate file not found: {path}")


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    candidates = load_or_build_candidates(args)
    work = add_photo_features(candidates)
    work["BadPhotoHeuristicProbability"] = heuristic_bad_photo_probability(work)
    work = add_business_scores(work, bad_probability_col="BadPhotoHeuristicProbability")
    work = work.sort_values(["PhotoOpportunityScore", "BadPhotoHeuristicProbability"], ascending=False, kind="stable")
    if args.limit:
        work = work.head(int(args.limit)).copy()
    work.insert(0, "manual_label", "")
    work.insert(1, "review_tags", "")
    work.insert(2, "review_notes", "")

    out_path = LABELS_DIR / "photo_quality_label_sheet.csv"
    if out_path.exists() and not args.overwrite:
        existing = pd.read_csv(out_path, low_memory=False)
        key_cols = ["item_id"] if "item_id" in existing.columns and "item_id" in work.columns else ["Link"]
        preserved = existing[[*key_cols, "manual_label", "review_tags", "review_notes"]].copy()
        work = work.drop(columns=["manual_label", "review_tags", "review_notes"], errors="ignore").merge(
            preserved,
            how="left",
            on=key_cols,
            suffixes=("", "_old"),
        )
        for col in ("manual_label", "review_tags", "review_notes"):
            if col not in work.columns:
                work[col] = ""
            work[col] = work[col].fillna("")
        work = work[LABEL_COLUMNS + [col for col in work.columns if col not in LABEL_COLUMNS]]
    else:
        work = work[LABEL_COLUMNS + [col for col in work.columns if col not in LABEL_COLUMNS]]

    write_csv(work, out_path)
    write_manifest(
        LABELS_DIR / "photo_quality_label_sheet_manifest.json",
        command=" ".join(sys.argv),
        extra={
            "rows": int(len(work)),
            "output": str(out_path),
            "valid_labels": ["photo_quality_bad", "photo_quality_good", "unclear", "not_item_photo"],
        },
    )
    print(f"Wrote label sheet with {len(work)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
