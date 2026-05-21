"""Train a basic+visual → sold/not-sold classifier per search."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.basic_plus_visual.paths import (
    ensure_experiment_dirs,
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    REPORTS_DIR,
    run_id,
    utc_now_iso,
    write_json,
    write_manifest,
)

SEED = 42
DATASET_DIR = ROOT / "data" / "experiments" / "full_scrape_model" / "offline_runs" / "sold_status_feature_modalities_20260518_082505" / "datasets"

TEXT_FEATURES = ["Title", "Brand", "Size"]
ALL_NUMERIC = [
    "Price", "Likes", "Page", "SearchCount",
    "SimpleBadPhotoScore",
    "AestheticGoodScore", "AestheticBadPhotoScore",
    "CombinedBadPhotoScore",
    "DinoEmbeddingDim", "DinoEmbeddingNorm", "DinoOutlierScore",
    "PhotoImageCount", "PhotoUsableImageCount",
    "PhotoPrimaryWidth", "PhotoPrimaryHeight", "PhotoPrimaryAspectRatio",
    "PhotoPrimaryBrightness", "PhotoPrimaryContrast", "PhotoPrimarySaturation",
    "PhotoPrimarySharpness", "PhotoPrimaryEdgeDensity",
    "PhotoAvgBrightness", "PhotoAvgContrast", "PhotoAvgSaturation",
    "PhotoAvgSharpness", "PhotoAvgEdgeDensity",
    "PhotoDuplicateImageCount", "PhotoLowQualityFraction",
    "PhotoScreenshotRiskFraction", "PhotoOnlyOneImage", "PhotoMissingImages",
    "PictureCountNum",
] + [f"DinoEmbedding_{i:04d}" for i in range(384)]


def to_numeric_frame(X):
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")


def text_values_to_str(X):
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0].fillna("").astype(str)
    return pd.Series(X).fillna("").astype(str)


def make_model() -> Pipeline:
    transformers = []
    numeric_pipe = Pipeline([
        ("to_numeric", FunctionTransformer(to_numeric_frame, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])
    transformers.append(("numeric", numeric_pipe, ALL_NUMERIC))
    for col in TEXT_FEATURES:
        text_pipe = Pipeline([
            ("to_text", FunctionTransformer(text_values_to_str, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=700, ngram_range=(1, 2), min_df=1)),
        ])
        transformers.append((f"text_{col}", text_pipe, col))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
    estimator = ExtraTreesClassifier(
        n_estimators=200,
        min_samples_leaf=4,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    return Pipeline([("features", preprocessor), ("model", estimator)])


def available_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def stratified_split(df: pd.DataFrame, target_col: str, seed: int = SEED):
    """60/20/20 stratified split."""
    train, temp = train_test_split(df, test_size=0.4, stratify=df[target_col], random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[target_col], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def evaluate(y_true, y_pred_proba, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def train_search(search_name: str, df: pd.DataFrame) -> dict[str, object]:
    print(f"\n=== {search_name} ===")
    print(f"Raw rows: {len(df)}")

    target = "offline_sold_label"
    if target not in df.columns:
        print(f"  Skipping: no {target} column")
        return {}

    df = df.dropna(subset=[target]).copy()
    df[target] = df[target].astype(int)
    print(f"Rows with label: {len(df)}, sold_rate={df[target].mean():.2%}")

    if len(df) < 50:
        print("  Skipping: too few rows")
        return {}

    numeric = available_columns(df, ALL_NUMERIC)
    text = available_columns(df, TEXT_FEATURES)
    print(f"Features: {len(numeric)} numeric, {len(text)} text")

    train, val, test = stratified_split(df, target)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    model = make_model()
    t0 = time.perf_counter()
    model.fit(train[text + numeric], train[target])
    fit_sec = time.perf_counter() - t0
    print(f"Fit in {fit_sec:.1f}s")

    # Predict probabilities
    train_proba = model.predict_proba(train[text + numeric])[:, 1]
    val_proba = model.predict_proba(val[text + numeric])[:, 1]
    test_proba = model.predict_proba(test[text + numeric])[:, 1]

    # Threshold selection: target 20% pass rate on validation
    threshold = np.percentile(val_proba, 80)
    print(f"Selected threshold (80th pct): {threshold:.4f}")

    train_metrics = evaluate(train[target].values, train_proba, threshold)
    val_metrics = evaluate(val[target].values, val_proba, threshold)
    test_metrics = evaluate(test[target].values, test_proba, threshold)

    print(f"Train: ROC={train_metrics['roc_auc']:.3f} PR={train_metrics['pr_auc']:.3f} P={train_metrics['precision']:.3f} R={train_metrics['recall']:.3f}")
    print(f"Val:   ROC={val_metrics['roc_auc']:.3f} PR={val_metrics['pr_auc']:.3f} P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f}")
    print(f"Test:  ROC={test_metrics['roc_auc']:.3f} PR={test_metrics['pr_auc']:.3f} P={test_metrics['precision']:.3f} R={test_metrics['recall']:.3f}")

    # Feature importances
    feature_importances = {}
    try:
        importances = model.named_steps["model"].feature_importances_
        feature_names = model.named_steps["features"].get_feature_names_out()
        top_idx = np.argsort(importances)[-20:][::-1]
        for idx in top_idx:
            feature_importances[feature_names[idx]] = float(importances[idx])
    except Exception as exc:
        print(f"Could not extract importances: {exc}")

    # Save model
    run_name = run_id("basic_plus_visual")
    artifact_path = MODELS_DIR / f"{run_name}_{search_name}_extra_trees.pkl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(artifact_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "experiment_family": "basic_plus_visual",
        "search_name": search_name,
        "approach": "extra_trees_classifier",
        "artifact_path": str(artifact_path),
        "numeric_features": numeric,
        "text_features": text,
        "threshold": float(threshold),
        "seed": SEED,
        "fit_seconds": fit_sec,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "sold_rate": float(df[target].mean()),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "top_features": feature_importances,
    }
    meta_path = artifact_path.with_suffix(".json")
    write_json(meta_path, metadata)

    # Save scored test items
    test_scored = test.copy()
    test_scored["sold_proba"] = test_proba
    test_scored["sold_pred"] = (test_proba >= threshold).astype(int)

    return {
        "search_name": search_name,
        "artifact_path": str(artifact_path),
        "metadata_path": str(meta_path),
        "threshold": float(threshold),
        "val_roc_auc": val_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "n_test": len(test),
        "n_test_pred_sold": int((test_proba >= threshold).sum()),
        "test_scored": test_scored,
    }


def main() -> int:
    ensure_experiment_dirs()
    parser = argparse.ArgumentParser(description="Train basic+visual → sold classifier")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--search", action="append", default=[])
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Dataset dir not found: {dataset_dir}")
        return 1

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if args.search:
        wanted = {s.lower() for s in args.search}
        csv_files = [f for f in csv_files if f.stem.lower() in wanted]

    results: list[dict[str, object]] = []
    all_test_scored: list[pd.DataFrame] = []

    for csv_file in csv_files:
        search_name = csv_file.stem
        df = pd.read_csv(csv_file, low_memory=False)
        result = train_search(search_name, df)
        if result:
            results.append(result)
            all_test_scored.append(result.pop("test_scored"))

    # Summary
    summary = pd.DataFrame(results)
    run_dir = OFFLINE_RUNS_DIR / run_id("basic_plus_visual_train")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    if all_test_scored:
        test_all = pd.concat(all_test_scored, ignore_index=True)
        test_all.to_csv(run_dir / "test_scored_items.csv", index=False)

    manifest = {
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "searches": [r["search_name"] for r in results],
        "n_models": len(results),
    }
    write_manifest(run_dir / "manifest.json", command="basic_plus_visual train", extra=manifest)

    print(f"\n=== Summary ===")
    print(f"Models trained: {len(results)}")
    print(f"Run dir: {run_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
