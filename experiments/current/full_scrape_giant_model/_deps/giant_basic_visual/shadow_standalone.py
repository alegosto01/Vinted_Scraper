"""Apply main_image_scores model to ALL live items (no basic5 pre-filter)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))

from experiments.current.full_scrape_giant_model._deps.giant_basic_visual.features import (
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
    / "experiments/current/giant_basic_visual/data/models"
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

SCORE_COLS = MAIN_IMAGE_SCORE_FEATURES
ALL_FEATURE_COLS = ["Price", "Likes"] + SEARCH_ONEHOTS + MAIN_IMAGE_FEATURES + SCORE_COLS


def load_visual_source_for_items(items: pd.DataFrame) -> pd.DataFrame:
    vfp_col = "collector_visual_features_path"
    unique_files = items[vfp_col].dropna().unique()
    chunks = []
    for fpath in unique_files:
        p = Path(fpath)
        if not p.exists():
            continue
        try:
            chunk = pd.read_csv(
                p,
                usecols=lambda c: c in {"SearchName", "item_id", "LocalPrimaryImagePath"} | set(SCORE_COLS),
                low_memory=False,
            )
            chunks.append(chunk)
        except Exception:
            continue
    if not chunks:
        return pd.DataFrame(columns=["SearchName", "item_id", "LocalPrimaryImagePath"] + SCORE_COLS)
    combined = pd.concat(chunks, ignore_index=True)
    combined["item_id"] = combined["item_id"].astype(str)
    combined = combined.drop_duplicates(subset=["SearchName", "item_id"], keep="last")
    return combined.reset_index(drop=True)


def main() -> None:
    print("Loading live scored items (ALL items, no basic5 filter)...")
    df = pd.read_csv(LIVE_SCORED, low_memory=False)
    df["item_id"] = df["item_id"].astype(str)
    print(f"Total live items: {len(df)}")

    pass_cols = [c for c in df.columns if c.startswith("pass__") and "rules" not in c]
    any_pass = df[pass_cols].any(axis=1)
    print(f"Items passing any basic5: {any_pass.sum()}")

    # Work on all items
    all_items = df.copy()

    print("\nLoading visual feature scores...")
    visual = load_visual_source_for_items(all_items)
    print(f"  Loaded {len(visual)} visual score rows from cascade files")

    all_items = all_items.merge(
        visual[["SearchName", "item_id"] + SCORE_COLS + ["LocalPrimaryImagePath"]],
        on=["SearchName", "item_id"],
        how="left",
        suffixes=("", "__vis"),
    )
    if "LocalPrimaryImagePath" not in all_items.columns and "LocalPrimaryImagePath__vis" in all_items.columns:
        all_items["LocalPrimaryImagePath"] = all_items["LocalPrimaryImagePath__vis"]
    elif "LocalPrimaryImagePath__vis" in all_items.columns:
        all_items["LocalPrimaryImagePath"] = all_items["LocalPrimaryImagePath"].fillna(
            all_items["LocalPrimaryImagePath__vis"]
        )

    score_matched = all_items[SCORE_COLS[0]].notna().sum()
    print(f"  Items with quality scores: {score_matched}/{len(all_items)}")

    print("\nComputing MainImage* features from local image paths...")
    main_image_rows = []
    for _, row in all_items.iterrows():
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
    all_items = pd.concat([all_items.reset_index(drop=True), mif.reset_index(drop=True)], axis=1)
    has_img = all_items["MainImageWidth"].notna().sum()
    print(f"  Items with MainImage features: {has_img}/{len(all_items)}")

    for col in SEARCH_ONEHOTS:
        search_key = col.replace("search__", "")
        all_items[col] = (all_items["SearchName"] == search_key).astype(float)

    has_all = all_items[ALL_FEATURE_COLS].notna().all(axis=1)
    scoreable = all_items[has_all].copy()
    print(f"\nItems scoreable (all features present): {len(scoreable)}")

    # How many scoreable also passed basic5?
    scoreable_pass = scoreable[pass_cols].any(axis=1)
    print(f"  Of scoreable, passed basic5: {scoreable_pass.sum()}")
    print(f"  Of scoreable, did NOT pass basic5: {(~scoreable_pass).sum()}")

    print(f"\nLoading model: {MODEL_PKL.name}")
    model = joblib.load(MODEL_PKL)

    X = scoreable[ALL_FEATURE_COLS].astype(float)
    scoreable = scoreable.copy()
    scoreable["img_score"] = model.predict_proba(X)[:, 1]
    scoreable["img_pass"] = scoreable["img_score"] >= THRESHOLD

    also_basic5 = scoreable[pass_cols].any(axis=1)

    n_pass = scoreable["img_pass"].sum()
    print(f"Threshold: {THRESHOLD}")
    print(f"Items above threshold: {n_pass}/{len(scoreable)}")
    print(f"  Of above-threshold, also passed basic5: {(scoreable['img_pass'] & also_basic5).sum()}")
    print(f"  Of above-threshold, did NOT pass basic5: {(scoreable['img_pass'] & ~also_basic5).sum()}")

    # Precision at each horizon
    for h in [6, 12, 24, 48]:
        sold_col = f"sold_within_{h}h"
        ev_col = f"evaluated_at_{h}h"
        if sold_col not in scoreable.columns:
            continue
        eval_mask = scoreable[ev_col].notna()
        above = scoreable[scoreable["img_pass"] & eval_mask]
        below = scoreable[~scoreable["img_pass"] & eval_mask]
        basic5_above = scoreable[also_basic5 & eval_mask]
        if len(above) == 0:
            continue
        prec_above = above[sold_col].sum() / len(above)
        prec_below = below[sold_col].sum() / len(below) if len(below) > 0 else float("nan")
        prec_all = scoreable.loc[eval_mask, sold_col].sum() / eval_mask.sum()
        prec_basic5 = basic5_above[sold_col].sum() / len(basic5_above) if len(basic5_above) > 0 else float("nan")
        print(f"\n--- {h}h sold label ---")
        print(f"  All scoreable (evaluated): {eval_mask.sum()}, precision={prec_all:.1%}")
        print(f"  basic5 passed (evaluated): {len(basic5_above)}, precision={prec_basic5:.1%}")
        print(f"  img_model above threshold ({len(above)} evaluated): precision={prec_above:.1%}")
        print(f"  img_model below threshold ({len(below)} evaluated): precision={prec_below:.1%}")

    # Per-search 24h breakdown
    print("\n--- Per-search 24h, above-threshold items ---")
    sold_col = "sold_within_24h"
    ev_col = "evaluated_at_24h"
    above_eval = scoreable[scoreable["img_pass"] & scoreable[ev_col].notna()]
    for search in sorted(above_eval["SearchName"].unique()):
        sub = above_eval[above_eval["SearchName"] == search]
        prec = sub[sold_col].sum() / len(sub)
        also_b5 = sub[pass_cols].any(axis=1).sum()
        print(f"  {search}: {sub[sold_col].sum()}/{len(sub)} sold = {prec:.1%}  (also_basic5={also_b5})")

    # Score distribution
    print("\n--- Score distribution (ALL scoreable items) ---")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        v = scoreable["img_score"].quantile(q)
        print(f"  p{int(q*100):3d}: {v:.4f}")

    print("\n--- Score distribution (basic5-passed scoreable only) ---")
    b5s = scoreable[also_basic5]
    for q in [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        v = b5s["img_score"].quantile(q)
        print(f"  p{int(q*100):3d}: {v:.4f}")

    print("\n--- Score distribution (basic5-FAILED scoreable only) ---")
    b5f = scoreable[~also_basic5]
    for q in [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        v = b5f["img_score"].quantile(q)
        print(f"  p{int(q*100):3d}: {v:.4f}")


if __name__ == "__main__":
    main()
