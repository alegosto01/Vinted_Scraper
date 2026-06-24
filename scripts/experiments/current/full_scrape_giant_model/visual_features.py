#!/usr/bin/env python3
"""Shared visual/photo-quality feature loaders for full_scrape_giant_model.

Two sources share the same VISUAL_NUMERIC column names:
- offline: data/experiments/photo_arbitrage/features/sold_unsold_visuals_20260514_full/combined_scored.csv
  (backfill dataset, keyed by SearchName+item_id)
- live: <live_dir>/visual_features/*.csv (per-snapshot, keyed by item_id)

A model trained with one source's columns can therefore be scored against the
other source's frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.full_scrape_model.compare_feature_modalities import VISUAL_NUMERIC  # noqa: E402
from experiments.current.full_scrape_giant_model.paths import ROOT  # noqa: E402

OFFLINE_VISUAL_SOURCE = (
    ROOT / "data" / "experiments" / "photo_arbitrage" / "features"
    / "sold_unsold_visuals_20260514_full" / "combined_scored.csv"
)


def merge_offline_visual_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Left-merge offline visual/photo-quality scores onto ``frame`` by (SearchName, item_id).

    Returns the merged frame and the subset of ``VISUAL_NUMERIC`` columns that
    are present and have at least one non-null value after the merge.
    """
    head = pd.read_csv(OFFLINE_VISUAL_SOURCE, nrows=0).columns
    visual_cols = [c for c in VISUAL_NUMERIC if c in head]
    usecols = ["SearchName", "item_id", *visual_cols]
    visual = pd.read_csv(OFFLINE_VISUAL_SOURCE, usecols=usecols, low_memory=False)
    visual["SearchName"] = visual["SearchName"].fillna("").astype(str)
    visual["item_id"] = visual["item_id"].astype(str).str.strip()
    visual = visual.drop_duplicates(subset=["SearchName", "item_id"], keep="last")

    out = frame.copy()
    out["item_id"] = out["item_id"].astype(str).str.strip()
    merged = out.merge(visual, on=["SearchName", "item_id"], how="left", suffixes=("", "_visual"))
    present = [c for c in visual_cols if c in merged.columns and merged[c].notna().any()]
    return merged, present


def load_live_visual_features(live_dir: Path) -> pd.DataFrame:
    """Concatenate per-snapshot visual feature CSVs, keep item_id + VISUAL_NUMERIC."""
    vdir = live_dir / "visual_features"
    files = sorted(vdir.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["item_id"])
    keep = ["item_id"] + list(VISUAL_NUMERIC)
    frames = []
    for f in files:
        head = pd.read_csv(f, nrows=0).columns
        usecols = [c for c in keep if c in head]
        if "item_id" not in usecols:
            continue
        frames.append(pd.read_csv(f, usecols=usecols, low_memory=False))
    if not frames:
        return pd.DataFrame(columns=["item_id"])
    visual = pd.concat(frames, ignore_index=True)
    visual["item_id"] = visual["item_id"].astype(str).str.strip()
    visual = visual[visual["item_id"].str.len() > 0].drop_duplicates("item_id", keep="last")
    return visual.reset_index(drop=True)
