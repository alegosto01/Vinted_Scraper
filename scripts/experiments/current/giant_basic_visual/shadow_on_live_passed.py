"""Apply main_image_scores model to live items that passed basic5, measure precision."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

from experiments.current.giant_basic_visual.features import (
    MAIN_IMAGE_FEATURES,
    MAIN_IMAGE_SCORE_FEATURES,
    main_image_features_for_path,
    resolve_main_image_path,
)

LIVE_SCORED = (
    ROOT
    / "experiments/current/basic_5_giant_model/data/live_scoring"
    / "live_scoring_20260613_152451"
    / "live_scored_items.csv"
)

MODEL_PKL = (
    ROOT
    / "data/experiments/giant_basic_visual/models"
    / "giant_basic_visual_20260613_153550_main_image_scores_hist_gradient_basic_numeric_v1_seed42.pkl"
)

THRESHOLD = 0.9946

SEARCH_ONEHOTS = [
    "search__griffati_donna_all",
    "search__griffati_uomo_all",
    "search__gucci",
    "search__nike",
    "search__prada",
    "search__ps4",
    "search__telefoni",
]

SCORE_COLS = MAIN_IMAGE_SCORE_FEATURES  # SimpleBadPhotoScore etc.
ALL_FEATURE_COLS = ["Price", "Likes"] + SEARCH_ONEHOTS + MAIN_IMAGE_FEATURES + SCORE_COLS


def load_visual_source_for_items(items: pd.DataFrame) -> pd.DataFrame:
    """Load quality scores from cascade visual feature files for these items."""
    vfp_col = "collector_visual_features_path"
    unique_files = items[vfp_col].dropna().unique()
    chunks = []
    for fpath in unique_files:
        p = Path(fpath)
        if not p.exists():
            continue
        try:
            chunk = pd.read_csv(p, usecols=lambda c: c in {"SearchName", "item_id", "LocalPrimaryImagePath"} | set(SCORE_COLS), low_memory=False)
            chunks.append(chunk)
        except Exception:
            continue
    if not chunks:
        return pd.DataFrame(columns=["SearchName", "item_id", "LocalPrimaryImagePath"] + SCORE_COLS)
    combined = pd.concat(chunks, ignore_index=True)
    combined["item_id"] = combined["item_id"].astype(str)
    combined = combined.drop_duplicates(subset=["SearchName", "item_id"], keep="last")
    return combined.reset_index(drop=True)


def build_search_onehots(search_name: str) -> dict[str, float]:
    return {col: float(col == f"search__{search_name}") for col in SEARCH_ONEHOTS}


def main() -> None:
    print("Loading live scored items...")
    df = pd.read_csv(LIVE_SCORED, low_memory=False)
    df["item_id"] = df["item_id"].astype(str)

    pass_cols = [c for c in df.columns if c.startswith("pass__") and "rules" not in c]
    any_pass = df[pass_cols].any(axis=1)
    passed = df[any_pass].copy()
    print(f"Items passing any basic5 model: {len(passed)}")

    print("\nPer-model pass counts:")
    for col in pass_cols:
        print(f"  {col}: {(passed[col] == True).sum()}")

    print("\nLoading visual feature scores...")
    visual = load_visual_source_for_items(passed)
    print(f"  Loaded {len(visual)} visual score rows from cascade files")

    # Merge quality scores onto passed items
    passed = passed.merge(
        visual[["SearchName", "item_id"] + SCORE_COLS + ["LocalPrimaryImagePath"]],
        on=["SearchName", "item_id"],
        how="left",
        suffixes=("", "__vis"),
    )
    # Prefer visual-source paths for LocalPrimaryImagePath if not in passed
    if "LocalPrimaryImagePath" not in passed.columns and "LocalPrimaryImagePath__vis" in passed.columns:
        passed["LocalPrimaryImagePath"] = passed["LocalPrimaryImagePath__vis"]
    elif "LocalPrimaryImagePath__vis" in passed.columns:
        passed["LocalPrimaryImagePath"] = passed["LocalPrimaryImagePath"].fillna(passed["LocalPrimaryImagePath__vis"])

    score_matched = passed[SCORE_COLS[0]].notna().sum()
    print(f"  Items with quality scores: {score_matched}/{len(passed)}")

    print("\nComputing MainImage* features from local image paths...")
    main_image_rows = []
    for _, row in passed.iterrows():
        path, reason = resolve_main_image_path(row.get("LocalPrimaryImagePath"))
        if reason or path is None:
            main_image_rows.append({col: np.nan for col in MAIN_IMAGE_FEATURES})
            continue
        try:
            feats = main_image_features_for_path(path)
            main_image_rows.append(feats)
        except Exception:
            main_image_rows.append({col: np.nan for col in MAIN_IMAGE_FEATURES})

    mif = pd.DataFrame(main_image_rows)
    passed = pd.concat([passed.reset_index(drop=True), mif.reset_index(drop=True)], axis=1)
    has_img = passed["MainImageWidth"].notna().sum()
    print(f"  Items with MainImage features: {has_img}/{len(passed)}")

    # Build search one-hots
    for col in SEARCH_ONEHOTS:
        search_key = col.replace("search__", "")
        passed[col] = (passed["SearchName"] == search_key).astype(float)

    # Restrict to items with full features
    has_all = passed[ALL_FEATURE_COLS].notna().all(axis=1)
    scoreable = passed[has_all].copy()
    print(f"\nItems scoreable (all features present): {len(scoreable)}")

    # Load model
    print(f"\nLoading model: {MODEL_PKL.name}")
    model = joblib.load(MODEL_PKL)

    X = scoreable[ALL_FEATURE_COLS].astype(float)
    scoreable = scoreable.copy()
    scoreable["img_score"] = model.predict_proba(X)[:, 1]
    scoreable["img_pass"] = scoreable["img_score"] >= THRESHOLD

    n_pass = scoreable["img_pass"].sum()
    print(f"Threshold: {THRESHOLD}")
    print(f"Items above threshold: {n_pass}/{len(scoreable)}")

    # Precision at 24h
    for h in [6, 12, 24, 48]:
        sold_col = f"sold_within_{h}h"
        ev_col = f"evaluated_at_{h}h"
        if sold_col not in scoreable.columns:
            continue
        eval_mask = scoreable[ev_col].notna()
        above = scoreable[scoreable["img_pass"] & eval_mask]
        below = scoreable[~scoreable["img_pass"] & eval_mask]
        if len(above) == 0:
            continue
        prec_above = above[sold_col].sum() / len(above)
        prec_below = below[sold_col].sum() / len(below) if len(below) > 0 else float("nan")
        prec_all = scoreable.loc[eval_mask, sold_col].sum() / eval_mask.sum()
        print(f"\n--- {h}h sold label ---")
        print(f"  All scoreable (evaluated): {eval_mask.sum()}, precision={prec_all:.1%}")
        print(f"  Above threshold ({len(above)} evaluated): precision={prec_above:.1%}")
        print(f"  Below threshold ({len(below)} evaluated): precision={prec_below:.1%}")

    # Per-search breakdown at 24h
    print("\n--- Per-search 24h precision, above-threshold items ---")
    sold_col = "sold_within_24h"
    ev_col = "evaluated_at_24h"
    above_all = scoreable[scoreable["img_pass"] & scoreable[ev_col].notna()]
    for search in above_all["SearchName"].unique():
        sub = above_all[above_all["SearchName"] == search]
        prec = sub[sold_col].sum() / len(sub)
        print(f"  {search}: {sub[sold_col].sum()}/{len(sub)} sold = {prec:.1%}")

    # Show score distribution for passed items
    print("\n--- Score distribution (scoreable items, model score) ---")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        v = scoreable["img_score"].quantile(q)
        print(f"  p{int(q*100):3d}: {v:.4f}")


if __name__ == "__main__":
    main()
