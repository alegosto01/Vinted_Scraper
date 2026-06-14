"""Train basic_5_control vs main_image_scores models on LIVE collector data.

Offline models were trained on a curated photo_arbitrage dataset with a much
higher positive base rate (~47%) than the live collector (~15% sold_within_24h).
This retrains both feature modes on the live population/labels so thresholds
and precision are calibrated to the live distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

from experiments.current.giant_basic_visual.features import (
    MAIN_IMAGE_FEATURES,
    MAIN_IMAGE_SCORE_FEATURES,
    main_image_features_for_path,
    resolve_main_image_path,
)
from experiments.current.giant_basic_visual.paths import (
    EXPERIMENT_ROOT,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)
from experiments.deal_finder.modeling import choose_threshold, threshold_metrics

LIVE_SCORED = (
    ROOT
    / "data/experiments/basic_5_giant_model/live_scoring"
    / "live_scoring_20260613_152451"
    / "live_scored_items.csv"
)

SEED = 42
LABEL_COL = "sold_within_24h"
EVAL_COL = "evaluated_at_24h"

SEARCH_ONEHOTS = [
    "search__griffati_donna_all",
    "search__griffati_uomo_all",
    "search__gucci",
    "search__nike",
    "search__prada",
    "search__ps4",
    "search__telefoni",
]
KNOWN_SEARCHES = {col.replace("search__", "") for col in SEARCH_ONEHOTS}

SCORE_COLS = MAIN_IMAGE_SCORE_FEATURES
BASIC_FEATURES = ["Price", "Likes"] + SEARCH_ONEHOTS
VISUAL_FEATURES = BASIC_FEATURES + MAIN_IMAGE_FEATURES + SCORE_COLS


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


def build_dataset() -> pd.DataFrame:
    print("Loading live scored items...")
    df = pd.read_csv(LIVE_SCORED, low_memory=False)
    df["item_id"] = df["item_id"].astype(str)
    print(f"  total rows: {len(df)}")

    print("Loading visual feature scores...")
    visual = load_visual_source_for_items(df)
    print(f"  loaded {len(visual)} visual score rows")

    df = df.merge(
        visual[["SearchName", "item_id"] + SCORE_COLS + ["LocalPrimaryImagePath"]],
        on=["SearchName", "item_id"],
        how="left",
        suffixes=("", "__vis"),
    )
    if "LocalPrimaryImagePath" not in df.columns and "LocalPrimaryImagePath__vis" in df.columns:
        df["LocalPrimaryImagePath"] = df["LocalPrimaryImagePath__vis"]
    elif "LocalPrimaryImagePath__vis" in df.columns:
        df["LocalPrimaryImagePath"] = df["LocalPrimaryImagePath"].fillna(df["LocalPrimaryImagePath__vis"])

    print("Computing MainImage* features from local image paths (this takes a while)...")
    rows = []
    for _, row in df.iterrows():
        path, reason = resolve_main_image_path(row.get("LocalPrimaryImagePath"))
        if reason or path is None:
            rows.append({col: np.nan for col in MAIN_IMAGE_FEATURES})
            continue
        try:
            rows.append(main_image_features_for_path(path))
        except Exception:
            rows.append({col: np.nan for col in MAIN_IMAGE_FEATURES})
    mif = pd.DataFrame(rows)
    df = pd.concat([df.reset_index(drop=True), mif.reset_index(drop=True)], axis=1)

    for col in SEARCH_ONEHOTS:
        search_key = col.replace("search__", "")
        df[col] = (df["SearchName"] == search_key).astype(float)

    return df


def evaluate_split(model, X, y, label: str) -> dict:
    scores = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, scores)
    pr_auc = average_precision_score(y, scores)
    print(f"  [{label}] n={len(y)} base_rate={y.mean():.3f} AUC={auc:.3f} PR-AUC={pr_auc:.3f}")
    return {"scores": scores, "auc": auc, "pr_auc": pr_auc}


def main() -> None:
    df = build_dataset()

    df = df[df["SearchName"].isin(KNOWN_SEARCHES)].copy()
    print(f"\nRows in 7 known searches: {len(df)}")

    has_label = df[EVAL_COL].notna() & df[LABEL_COL].notna()
    df = df[has_label].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    print(f"Rows with {LABEL_COL} evaluated: {len(df)}, base rate={df[LABEL_COL].mean():.3f}")

    has_visual = df[VISUAL_FEATURES].notna().all(axis=1)
    df = df[has_visual].copy()
    print(f"Rows with full feature set (incl. visual scores): {len(df)}, base rate={df[LABEL_COL].mean():.3f}")

    print("\nPer-search counts / base rate:")
    for search, g in df.groupby("SearchName"):
        print(f"  {search}: n={len(g)} base_rate={g[LABEL_COL].mean():.3f}")

    df["_strata"] = df["SearchName"] + "__" + df[LABEL_COL].astype(str)

    train_df, rest_df = train_test_split(df, test_size=0.4, random_state=SEED, stratify=df["_strata"])
    val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=SEED, stratify=rest_df["_strata"])
    print(f"\nSplit: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    out_dir = EXPERIMENT_ROOT / "live_trained" / run_id("live_trained")
    ensure_experiment_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for mode, feature_cols in [("basic_5_control", BASIC_FEATURES), ("main_image_scores", VISUAL_FEATURES)]:
        print(f"\n=== {mode} ({len(feature_cols)} features) ===")
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.06,
                        max_leaf_nodes=15,
                        min_samples_leaf=12,
                        l2_regularization=0.1,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        X_train = train_df[feature_cols].astype(float)
        y_train = train_df[LABEL_COL].to_numpy()
        model.fit(X_train, y_train)

        X_val = val_df[feature_cols].astype(float)
        y_val = val_df[LABEL_COL].to_numpy()
        val_eval = evaluate_split(model, X_val, y_val, "val")

        threshold_info = choose_threshold(y_val, val_eval["scores"], min_precision=0.40, min_count=10)
        threshold = threshold_info["threshold"]
        print(f"  chosen threshold={threshold:.4f} (val precision={threshold_info['precision']:.3f}, "
              f"count={threshold_info['count']}, positives={threshold_info['positive_count']})")

        X_test = test_df[feature_cols].astype(float)
        y_test = test_df[LABEL_COL].to_numpy()
        test_eval = evaluate_split(model, X_test, y_test, "test")
        test_at_threshold = threshold_metrics(y_test, test_eval["scores"], threshold)
        print(f"  test @ threshold: count={test_at_threshold['count']} "
              f"precision={test_at_threshold['precision']:.3f} "
              f"positives={test_at_threshold['positive_count']}")

        model_path = out_dir / f"{mode}_hist_gradient_seed{SEED}.pkl"
        joblib.dump(model, model_path)

        results[mode] = {
            "n_features": len(feature_cols),
            "features": feature_cols,
            "val_auc": val_eval["auc"],
            "val_pr_auc": val_eval["pr_auc"],
            "test_auc": test_eval["auc"],
            "test_pr_auc": test_eval["pr_auc"],
            "threshold": threshold,
            "threshold_val_precision": threshold_info["precision"],
            "threshold_val_count": threshold_info["count"],
            "test_at_threshold": test_at_threshold,
            "model_path": str(model_path),
        }

    print("\n=== Summary ===")
    print(f"{'mode':<20} {'test AUC':>10} {'test PR-AUC':>12} {'thresh':>8} {'test prec@thresh':>18} {'n@thresh':>10}")
    for mode, r in results.items():
        tat = r["test_at_threshold"]
        print(f"{mode:<20} {r['test_auc']:>10.3f} {r['test_pr_auc']:>12.3f} {r['threshold']:>8.4f} "
              f"{tat['precision']:>18.3f} {tat['count']:>10}")

    write_json(out_dir / "results.json", {
        "label": LABEL_COL,
        "eval_col": EVAL_COL,
        "seed": SEED,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "base_rate": float(df[LABEL_COL].mean()),
        "results": results,
    })
    write_manifest(out_dir / "manifest.json", command="giant_basic_visual.train_on_live", extra={
        "live_scored_source": str(LIVE_SCORED),
        "n_rows_used": len(df),
        "telegram_policy_changed": False,
    })
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
