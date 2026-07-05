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

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))

from experiments.current.full_scrape_giant_model._deps.giant_basic_visual.features import (
    MAIN_IMAGE_FEATURES,
    MAIN_IMAGE_SCORE_FEATURES,
    main_image_features_for_path,
    resolve_main_image_path,
)
from experiments.current.full_scrape_giant_model._deps.giant_basic_visual.paths import (
    EXPERIMENT_ROOT,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)
from experiments.current.full_scrape_giant_model._deps.deal_finder.modeling import choose_threshold, threshold_metrics

LIVE_SCORING_DIR = ROOT / "experiments/current/basic_5_giant_model/data/live_scoring"


def latest_live_scored() -> Path:
    """Most recent basic5 live-scoring snapshot (full rescoring of tracked_items.csv at that point)."""
    runs = sorted(
        (p for p in LIVE_SCORING_DIR.glob("live_scoring_*") if (p / "live_scored_items.csv").exists()),
        key=lambda p: p.name,
    )
    if not runs:
        raise FileNotFoundError(f"No live_scoring_* runs with live_scored_items.csv found under {LIVE_SCORING_DIR}")
    return runs[-1] / "live_scored_items.csv"


LIVE_SCORED = latest_live_scored()

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
    "search__donna_accessori_gioielli",
    "search__hobby_collezionismo",
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
    print(f"\nRows in {len(KNOWN_SEARCHES)} known searches: {len(df)}")

    has_label = df[EVAL_COL].notna() & df[LABEL_COL].notna()
    df = df[has_label].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    print(f"Rows with {LABEL_COL} evaluated: {len(df)}, base rate={df[LABEL_COL].mean():.3f}")

    # Dino columns are excluded from this hard gate: HistGradientBoostingClassifier tolerates
    # NaN natively, and DinoEmbeddingNorm/DinoOutlierScore can be legitimately all-NaN for a
    # while after a dino pipeline fix (no retroactive backfill of historical visual_features
    # CSVs) without that blocking training on the rest of the feature set.
    required_features = [col for col in VISUAL_FEATURES if col not in ("DinoEmbeddingNorm", "DinoOutlierScore")]
    has_visual = df[required_features].notna().all(axis=1)
    df = df[has_visual].copy()
    print(f"Rows with full feature set (incl. visual scores): {len(df)}, base rate={df[LABEL_COL].mean():.3f}")
    print(f"  rows with non-null DinoEmbeddingNorm: {int(df['DinoEmbeddingNorm'].notna().sum())}")

    print("\nPer-search counts / base rate:")
    for search, g in df.groupby("SearchName"):
        print(f"  {search}: n={len(g)} base_rate={g[LABEL_COL].mean():.3f}")

    # Stratify by label only: joint search x label strata are too fine at this row count
    # (several searches currently have 0-1 positives in this snapshot), which makes
    # sklearn's stratified split reject classes with under 2 members.
    train_df, rest_df = train_test_split(df, test_size=0.4, random_state=SEED, stratify=df[LABEL_COL])
    val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=SEED, stratify=rest_df[LABEL_COL])
    print(f"\nSplit: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    out_dir = EXPERIMENT_ROOT / "live_trained" / run_id("live_trained")
    ensure_experiment_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Small grid: enough to check the fixed defaults aren't badly off, not wide enough to
    # chase noise on the ~30 positives this live-trained set currently has.
    HPARAM_GRID = [
        {"learning_rate": 0.06, "max_leaf_nodes": 15, "min_samples_leaf": 12, "l2_regularization": 0.1},
        {"learning_rate": 0.04, "max_leaf_nodes": 15, "min_samples_leaf": 12, "l2_regularization": 0.1},
        {"learning_rate": 0.06, "max_leaf_nodes": 9, "min_samples_leaf": 12, "l2_regularization": 0.1},
        {"learning_rate": 0.06, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.1},
        {"learning_rate": 0.06, "max_leaf_nodes": 15, "min_samples_leaf": 12, "l2_regularization": 0.3},
        {"learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 12, "l2_regularization": 0.1},
    ]

    results = {}
    for mode, feature_cols in [("basic_5_control", BASIC_FEATURES), ("main_image_scores", VISUAL_FEATURES)]:
        print(f"\n=== {mode} ({len(feature_cols)} features) ===")
        X_train = train_df[feature_cols].astype(float)
        y_train = train_df[LABEL_COL].to_numpy()
        X_val = val_df[feature_cols].astype(float)
        y_val = val_df[LABEL_COL].to_numpy()

        tried = []
        best_model = None
        best_eval = None
        best_params = None
        for params in HPARAM_GRID:
            candidate = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", HistGradientBoostingClassifier(max_iter=180, random_state=SEED, **params)),
                ]
            )
            candidate.fit(X_train, y_train)
            scores = candidate.predict_proba(X_val)[:, 1]
            pr_auc = average_precision_score(y_val, scores)
            tried.append({**params, "val_pr_auc": float(pr_auc)})
            if best_eval is None or pr_auc > best_eval["pr_auc"]:
                best_model = candidate
                best_eval = {"scores": scores, "pr_auc": pr_auc}
                best_params = params
        print(f"  hyperparam sweep ({len(HPARAM_GRID)} configs), best val PR-AUC={best_eval['pr_auc']:.3f}: {best_params}")

        model = best_model
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

        per_search_test = {}
        test_scores_series = pd.Series(test_eval["scores"], index=test_df.index)
        for search in sorted(KNOWN_SEARCHES):
            mask = (test_df["SearchName"] == search).to_numpy()
            if not mask.any():
                per_search_test[search] = {"count": 0, "precision": np.nan, "positive_count": 0, "n_test": 0}
                continue
            per_search_test[search] = {
                **threshold_metrics(y_test[mask], test_scores_series.to_numpy()[mask], threshold),
                "n_test": int(mask.sum()),
            }

        model_path = out_dir / f"{mode}_hist_gradient_seed{SEED}.pkl"
        joblib.dump(model, model_path)

        results[mode] = {
            "n_features": len(feature_cols),
            "features": feature_cols,
            "hyperparams": best_params,
            "hyperparam_sweep": tried,
            "val_auc": val_eval["auc"],
            "val_pr_auc": val_eval["pr_auc"],
            "test_auc": test_eval["auc"],
            "test_pr_auc": test_eval["pr_auc"],
            "threshold": threshold,
            "threshold_val_precision": threshold_info["precision"],
            "threshold_val_count": threshold_info["count"],
            "test_at_threshold": test_at_threshold,
            "per_search_test": per_search_test,
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
