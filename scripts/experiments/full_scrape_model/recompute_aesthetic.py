"""Recompute aesthetic scores for rows where AestheticStatus=load_failed:OSError."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.photo_arbitrage.quality_methods import (
    add_aesthetic_scores,
    combine_bad_photo_scores,
)


def recompute_aesthetic_for_csv(csv_path: Path, searches: list[str]) -> None:
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    mask = df["AestheticStatus"].astype(str) == "load_failed:OSError"
    if searches:
        search_mask = df["SearchName"].isin(searches)
        mask = mask & search_mask
    
    to_fix = df[mask].copy()
    print(f"Rows to recompute: {len(to_fix)}")
    
    if to_fix.empty:
        print("Nothing to recompute.")
        return
    
    # Recompute aesthetic scores
    fixed = add_aesthetic_scores(to_fix)
    
    # Recompute CombinedBadPhotoScore for fixed rows
    fixed["CombinedBadPhotoScore"] = combine_bad_photo_scores(fixed)
    
    # Update original dataframe
    update_cols = [
        "AestheticGoodScore", "AestheticBadPhotoScore", "AestheticLabel",
        "AestheticStatus", "AestheticModelName", "CombinedBadPhotoScore",
    ]
    for col in update_cols:
        if col in fixed.columns:
            df.loc[mask, col] = fixed[col].values
    
    # Save back
    print(f"Saving {csv_path} ...")
    df.to_csv(csv_path, index=False)
    
    # Verify
    after = df[df["SearchName"].isin(searches)] if searches else df
    print(f"After fix: AestheticGoodScore non-null = {after['AestheticGoodScore'].notna().sum()}/{len(after)}")
    print(f"AestheticStatus: {after['AestheticStatus'].value_counts().to_dict()}")


def regenerate_dataset(visual_path: Path, dataset_dir: Path, search: str) -> None:
    """Regenerate a single dataset CSV from the updated visual features."""
    from experiments.full_scrape_model.compare_feature_modalities import (
        read_visual_features,
        add_dino_embedding_columns,
        VISUAL_NUMERIC,
    )
    
    dataset_path = dataset_dir / f"{search}.csv"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return
    
    print(f"Regenerating dataset for {search} ...")
    frame = pd.read_csv(dataset_path, low_memory=False)
    
    visual = read_visual_features(visual_path, include_dino_embedding=True)
    search_visual = visual[visual["SearchName"].astype(str) == str(search)].copy()
    
    # Merge
    merged = frame.merge(
        search_visual.drop(columns=["SearchName"], errors="ignore"),
        on="item_id",
        how="left",
    )
    merged, embedding_cols = add_dino_embedding_columns(merged, max_dims=None)
    
    merged.to_csv(dataset_path, index=False)
    print(f"Saved {dataset_path}")


def main() -> int:
    visual_path = ROOT / "data" / "experiments" / "photo_arbitrage" / "features" / "sold_unsold_visuals_20260514_full" / "combined_scored.csv"
    dataset_dir = ROOT / "data" / "experiments" / "full_scrape_model" / "offline_runs" / "sold_status_feature_modalities_20260518_082505" / "datasets"
    searches = ["gucci", "prada", "ps4"]
    
    recompute_aesthetic_for_csv(visual_path, searches)
    
    for search in searches:
        regenerate_dataset(visual_path, dataset_dir, search)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
