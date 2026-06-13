#!/usr/bin/env python3
"""Train and apply full-scrape Telegram rerankers for Basic-5 Giant live scoring."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.basic_5_giant_model.paths import (  # noqa: E402
    EXPERIMENT_ROOT,
    MODELS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
)


RERANKER_ROOT = EXPERIMENT_ROOT / "full_scrape_reranker"
DEFAULT_SEARCHES = ("griffati_donna_all", "griffati_uomo_all")
DEFAULT_SEARCHES_UNIFIED = ("nike", "griffati_donna_all", "griffati_uomo_all", "gucci", "prada", "ps4")
DEFAULT_TARGET_PRECISIONS = (0.80, 0.70, 0.60)
DEFAULT_HORIZON_HOURS = 72
SCORE_PREFIX = "score__"
THRESHOLD_PREFIX = "threshold__"
RERANKER_PREFIX = "FullScrapeReranker"

BASE_NUMERIC_FEATURES = (
    "Price",
    "Likes",
    "Stage1Score",
    "Stage1Threshold",
    "Stage1Rank",
    "ReviewsCount",
    "Stars",
    "VisiblePictureCount",
    "HiddenPictureCount",
    "PictureCount",
    "description_char_len",
    "description_token_count",
    "title_char_len_full",
    "title_token_count_full",
    "upload_age_minutes",
    "seller_has_reviews",
    "stars_missing",
    "reviews_missing",
    "interested_missing",
    "views_missing",
    "picture_count_missing",
)
BASE_CATEGORICAL_FEATURES = (
    "Condition",
    "Brand",
    "LocationCountry",
)
TEXT_FEATURES = (
    "TitleText",
    "DescriptionText",
)


def latest_live_scoring_dir() -> Path:
    root = EXPERIMENT_ROOT / "live_scoring"
    runs = sorted(
        (path for path in root.glob("live_scoring_*") if (path / "live_scored_items.csv").exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No live_scoring_* folders found under {root}")
    return runs[0]


def safe_float_series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def clean_negative_missing(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values < 0)


def parse_upload_minutes(value: object) -> float:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return np.nan
    number_match = re.search(r"(\d+(?:[.,]\d+)?)", text)
    number = float(number_match.group(1).replace(",", ".")) if number_match else 1.0
    if any(token in text for token in ("min", "minute")):
        return number
    if any(token in text for token in ("ora", "ore", "hour", "heure", "h ")):
        return number * 60.0
    if any(token in text for token in ("giorn", "day", "jour", "d ")):
        return number * 1440.0
    if any(token in text for token in ("sett", "week", "semain")):
        return number * 7.0 * 1440.0
    if any(token in text for token in ("mese", "mesi", "month", "mois")):
        return number * 30.0 * 1440.0
    return np.nan


def location_country(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "unknown"
    text = re.sub(r"\s+", " ", text)
    if "," in text:
        tail = text.rsplit(",", 1)[-1].strip()
    else:
        tail = text
    tail = re.sub(r"^ultima visita.*", "unknown", tail, flags=re.IGNORECASE).strip()
    return tail or "unknown"


def text_token_count(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.findall(r"\w+").str.len()


def model_score_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith(SCORE_PREFIX))


def model_threshold_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith(THRESHOLD_PREFIX))


def add_full_scrape_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("Title", "Description", "Brand", "Condition", "Location"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    out["TitleText"] = out["Title"].fillna("").astype(str)
    out["DescriptionText"] = out["Description"].fillna("").astype(str)
    out["description_char_len"] = out["DescriptionText"].str.len()
    out["description_token_count"] = text_token_count(out["DescriptionText"])
    out["title_char_len_full"] = out["TitleText"].str.len()
    out["title_token_count_full"] = text_token_count(out["TitleText"])
    out["LocationCountry"] = out["Location"].map(location_country)
    out["upload_age_minutes"] = out.get("Upload_date", pd.Series("", index=out.index)).map(parse_upload_minutes)

    for col in ("ReviewsCount", "Stars", "Interested_count", "View_count", "PictureCount"):
        out[f"{col.lower()}_raw"] = safe_float_series(out, col)
        out[col] = clean_negative_missing(out[f"{col.lower()}_raw"])
    out["reviews_missing"] = out["ReviewsCount"].isna().astype(int)
    out["stars_missing"] = out["Stars"].isna().astype(int)
    out["interested_missing"] = out["Interested_count"].isna().astype(int)
    out["views_missing"] = out["View_count"].isna().astype(int)
    out["picture_count_missing"] = out["PictureCount"].isna().astype(int)
    out["seller_has_reviews"] = out["ReviewsCount"].fillna(0).gt(0).astype(int)

    for col in ("Price", "Likes", "Stage1Score", "Stage1Threshold", "Stage1Rank", "VisiblePictureCount", "HiddenPictureCount"):
        out[col] = safe_float_series(out, col)

    for score_col in model_score_columns(out):
        model = score_col.replace(SCORE_PREFIX, "", 1)
        threshold_col = f"{THRESHOLD_PREFIX}{model}"
        out[score_col] = safe_float_series(out, score_col)
        if threshold_col in out.columns:
            out[threshold_col] = safe_float_series(out, threshold_col)
            out[f"margin__{model}"] = out[score_col] - out[threshold_col]
    return out


def reranker_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = [col for col in BASE_NUMERIC_FEATURES if col in frame.columns]
    numeric.extend(model_score_columns(frame))
    numeric.extend(model_threshold_columns(frame))
    numeric.extend(sorted(col for col in frame.columns if col.startswith("margin__")))
    numeric = list(dict.fromkeys(numeric))
    categorical = [col for col in BASE_CATEGORICAL_FEATURES if col in frame.columns]
    text = [
        col for col in TEXT_FEATURES
        if col in frame.columns and frame[col].str.len().gt(0).any()
    ]
    return numeric, categorical, text


def build_model(numeric: list[str], categorical: list[str], text: list[str]) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers: list[tuple[str, Any, Any]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            )
        )
    for column in text:
        transformers.append(
            (
                f"text_{column}",
                TfidfVectorizer(max_features=400, min_df=3, ngram_range=(1, 2), strip_accents="unicode"),
                column,
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def score_model(model: Any, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(frame), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    return np.asarray(model.predict(frame), dtype=float)


def metric_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = scores >= threshold
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    passed = tp + fp
    total = len(y)
    positives = tp + fn
    negatives = fp + tn
    return {
        "threshold": float(threshold),
        "rows": int(total),
        "positives": int(positives),
        "passed": int(passed),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(tp / passed) if passed else np.nan,
        "recall": float(tp / positives) if positives else np.nan,
        "fpr": float(fp / negatives) if negatives else np.nan,
        "pass_rate": float(passed / total) if total else np.nan,
    }


def choose_threshold_for_precision(
    y: np.ndarray,
    scores: np.ndarray,
    target_precision: float,
) -> dict[str, float | int | str]:
    values = np.unique(scores[np.isfinite(scores)])
    candidates: list[dict[str, float | int]] = []
    for threshold in values[::-1]:
        metrics = metric_at_threshold(y, scores, float(threshold))
        precision = float(metrics["precision"]) if not pd.isna(metrics["precision"]) else -1.0
        if precision >= target_precision:
            candidates.append(metrics)
    if not candidates:
        return {
            "target_precision": float(target_precision),
            "status": "no_validation_threshold_met_target",
            **metric_at_threshold(y, scores, float("inf")),
        }
    best = sorted(
        candidates,
        key=lambda row: (
            int(row["tp"]),
            int(row["passed"]),
            float(row["recall"]) if not pd.isna(row["recall"]) else -1.0,
            -float(row["threshold"]),
        ),
        reverse=True,
    )[0]
    return {"target_precision": float(target_precision), "status": "ok", **best}


def evaluate_thresholds(
    y: np.ndarray,
    scores: np.ndarray,
    thresholds: dict[str, dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target_key, threshold_info in thresholds.items():
        threshold = float(threshold_info.get("threshold", np.inf))
        metrics = metric_at_threshold(y, scores, threshold)
        for key, value in metrics.items():
            out[f"{prefix}_{target_key}_{key}"] = value
        out[f"{prefix}_{target_key}_status"] = threshold_info.get("status", "")
    return out


def roc_auc_safe(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, scores))


def average_precision_safe(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, scores))


def split_search_frame(frame: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    y = frame["target"].astype(int)
    stratify = y if y.value_counts().min() >= 2 else None
    train, rest = train_test_split(frame, train_size=0.60, random_state=seed, shuffle=True, stratify=stratify)
    rest_y = rest["target"].astype(int)
    rest_stratify = rest_y if rest_y.value_counts().min() >= 2 else None
    validation, test = train_test_split(rest, train_size=0.50, random_state=seed, shuffle=True, stratify=rest_stratify)
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def load_scored_items(scoring_dir: Path | None) -> pd.DataFrame:
    resolved = latest_live_scoring_dir() if scoring_dir is None else scoring_dir
    path = resolved / "live_scored_items.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing live_scored_items.csv: {path}")
    return pd.read_csv(path, low_memory=False)


def load_enriched_items(live_run_dir: Path) -> pd.DataFrame:
    path = live_run_dir / "full_items" / "items_enriched.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def enrichment_columns(enriched: pd.DataFrame) -> list[str]:
    preferred = [
        "tracking_key",
        "Description",
        "Condition",
        "Upload_date",
        "Interested_count",
        "View_count",
        "SellerName",
        "SellerId",
        "Location",
        "ReviewsCount",
        "Stars",
        "PrimaryImageUrl",
        "FullImageUrls",
        "VisiblePictureCount",
        "HiddenPictureCount",
        "PictureCount",
    ]
    return [col for col in preferred if col in enriched.columns]


def merge_full_scrape_enrichment(scored: pd.DataFrame, live_run_dir: Path) -> pd.DataFrame:
    if "tracking_key" not in scored.columns:
        return scored.copy()
    enriched = load_enriched_items(live_run_dir)
    if enriched.empty:
        return scored.copy()
    cols = enrichment_columns(enriched)
    if "tracking_key" not in cols:
        return scored.copy()
    enrich = enriched[cols].drop_duplicates(subset=["tracking_key"], keep="last")
    add_cols = [col for col in enrich.columns if col == "tracking_key" or col not in scored.columns]
    return scored.merge(enrich[add_cols], on="tracking_key", how="left")


def build_training_frame(
    *,
    live_run_dir: Path,
    scoring_dir: Path | None,
    searches: tuple[str, ...],
    horizon_hours: int,
) -> pd.DataFrame:
    scored = load_scored_items(scoring_dir)
    merged = merge_full_scrape_enrichment(scored, live_run_dir)
    if "SearchName" not in merged.columns:
        raise ValueError("Scored items are missing SearchName")
    label_col = f"sold_within_{horizon_hours}h"
    if label_col not in merged.columns:
        raise ValueError(f"Scored items are missing {label_col}")
    merged = merged[merged["SearchName"].astype(str).isin(searches)].copy()
    if "Description" in merged.columns:
        merged = merged[merged["Description"].notna()].copy()
    merged["target"] = pd.to_numeric(merged[label_col], errors="coerce")
    merged = merged[merged["target"].notna()].copy()
    merged["target"] = merged["target"].astype(int)
    merged = add_full_scrape_features(merged)
    return merged.reset_index(drop=True)


def train_search_model(
    frame: pd.DataFrame,
    *,
    search: str,
    run_name: str,
    out_dir: Path,
    target_precisions: tuple[float, ...],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    search_frame = frame[frame["SearchName"].astype(str).eq(search)].copy()
    if len(search_frame) < 100 or search_frame["target"].nunique() < 2:
        raise ValueError(f"Not enough labeled full-scrape rows for {search}: rows={len(search_frame)}")

    train, validation, test = split_search_frame(search_frame, seed=seed)
    numeric, categorical, text = reranker_feature_columns(search_frame)
    model = build_model(numeric, categorical, text)
    feature_columns = numeric + categorical + text
    model.fit(train[feature_columns], train["target"].astype(int))

    val_scores = score_model(model, validation[feature_columns])
    test_scores = score_model(model, test[feature_columns])
    train_scores = score_model(model, train[feature_columns])

    thresholds: dict[str, dict[str, Any]] = {}
    for target in target_precisions:
        key = f"p{int(round(target * 100))}"
        thresholds[key] = choose_threshold_for_precision(validation["target"].to_numpy(dtype=int), val_scores, target)

    rows: list[dict[str, Any]] = []
    for split_name, split_frame, split_scores in (
        ("train", train, train_scores),
        ("validation", validation, val_scores),
        ("test", test, test_scores),
    ):
        y = split_frame["target"].to_numpy(dtype=int)
        base = {
            "search": search,
            "split": split_name,
            "rows": int(len(split_frame)),
            "positives": int(y.sum()),
            "base_rate": float(y.mean()) if len(y) else np.nan,
            "roc_auc": roc_auc_safe(y, split_scores),
            "average_precision": average_precision_safe(y, split_scores),
        }
        base.update(evaluate_thresholds(y, split_scores, thresholds, "target"))
        rows.append(base)
    metrics = pd.DataFrame(rows)

    artifact_path = MODELS_DIR / f"{run_name}_{search}_full_scrape_reranker.pkl"
    metadata_path = MODELS_DIR / f"{run_name}_{search}_full_scrape_reranker_metadata.json"
    artifact_path = assert_experiment_path(artifact_path)
    metadata_path = assert_experiment_path(metadata_path)
    with artifact_path.open("wb") as handle:
        pickle.dump(model, handle)

    metadata = {
        "run_name": run_name,
        "search": search,
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "model_family": "logistic_full_scrape_reranker_v1",
        "target_precisions": list(target_precisions),
        "thresholds": thresholds,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "text_features": text,
        "feature_columns": feature_columns,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "seed": int(seed),
    }
    write_json(metadata_path, metadata)
    return metadata, metrics


def train_unified_model(
    frame: pd.DataFrame,
    *,
    searches: tuple[str, ...],
    run_name: str,
    out_dir: Path,
    target_precisions: tuple[float, ...],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train a single model across all specified searches."""
    multi_search_frame = frame[frame["SearchName"].astype(str).isin(searches)].copy()
    if len(multi_search_frame) < 100 or multi_search_frame["target"].nunique() < 2:
        raise ValueError(f"Not enough labeled full-scrape rows: rows={len(multi_search_frame)}")

    train, validation, test = split_search_frame(multi_search_frame, seed=seed)
    numeric, categorical, text = reranker_feature_columns(multi_search_frame)
    model = build_model(numeric, categorical, text)
    feature_columns = numeric + categorical + text
    model.fit(train[feature_columns], train["target"].astype(int))

    val_scores = score_model(model, validation[feature_columns])
    test_scores = score_model(model, test[feature_columns])
    train_scores = score_model(model, train[feature_columns])

    thresholds: dict[str, dict[str, Any]] = {}
    for target in target_precisions:
        key = f"p{int(round(target * 100))}"
        thresholds[key] = choose_threshold_for_precision(validation["target"].to_numpy(dtype=int), val_scores, target)

    # Report metrics both per-split and per-search
    rows: list[dict[str, Any]] = []
    for split_name, split_frame, split_scores in (
        ("train", train, train_scores),
        ("validation", validation, val_scores),
        ("test", test, test_scores),
    ):
        y = split_frame["target"].to_numpy(dtype=int)
        # Overall metrics across all searches
        base = {
            "search": "__all__",
            "split": split_name,
            "rows": int(len(split_frame)),
            "positives": int(y.sum()),
            "base_rate": float(y.mean()) if len(y) else np.nan,
            "roc_auc": roc_auc_safe(y, split_scores),
            "average_precision": average_precision_safe(y, split_scores),
        }
        base.update(evaluate_thresholds(y, split_scores, thresholds, "target"))
        rows.append(base)

        # Per-search metrics
        for search in searches:
            search_mask = split_frame["SearchName"].astype(str).eq(search)
            if search_mask.sum() < 2:
                continue
            search_y = y[search_mask]
            search_scores = split_scores[search_mask]
            search_base = {
                "search": search,
                "split": split_name,
                "rows": int(search_mask.sum()),
                "positives": int(search_y.sum()),
                "base_rate": float(search_y.mean()) if len(search_y) else np.nan,
                "roc_auc": roc_auc_safe(search_y, search_scores),
                "average_precision": average_precision_safe(search_y, search_scores),
            }
            search_base.update(evaluate_thresholds(search_y, search_scores, thresholds, "target"))
            rows.append(search_base)

    metrics = pd.DataFrame(rows)

    artifact_path = MODELS_DIR / f"{run_name}_unified_full_scrape_reranker.pkl"
    metadata_path = MODELS_DIR / f"{run_name}_unified_full_scrape_reranker_metadata.json"
    artifact_path = assert_experiment_path(artifact_path)
    metadata_path = assert_experiment_path(metadata_path)
    with artifact_path.open("wb") as handle:
        pickle.dump(model, handle)

    metadata = {
        "run_name": run_name,
        "unified": True,
        "searches": list(searches),
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "model_family": "logistic_full_scrape_reranker_unified_v1",
        "target_precisions": list(target_precisions),
        "thresholds": thresholds,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "text_features": text,
        "feature_columns": feature_columns,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "seed": int(seed),
    }
    write_json(metadata_path, metadata)
    return metadata, metrics


def format_float(value: object, digits: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_training_report(path: Path, metrics: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "# Full-Scrape Telegram Reranker",
        "",
        f"Run: `{manifest['run_name']}`",
        f"Horizon: `{manifest['horizon_hours']}h`",
        "",
        "Thresholds are selected on validation to meet target precision, then reported on test.",
        "",
        "| search | split | rows | positives | AUC | AP | target | threshold | passed | precision | recall | FPR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in metrics.sort_values(["search", "split"]).iterrows():
        for target in manifest["target_precision_keys"]:
            prefix = f"target_{target}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["search"]),
                        str(row["split"]),
                        str(int(row["rows"])),
                        str(int(row["positives"])),
                        format_float(row["roc_auc"]),
                        format_float(row["average_precision"]),
                        target,
                        format_float(row.get(f"{prefix}_threshold")),
                        str(int(row.get(f"{prefix}_passed", 0))),
                        format_float(row.get(f"{prefix}_precision")),
                        format_float(row.get(f"{prefix}_recall")),
                        format_float(row.get(f"{prefix}_fpr")),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_rerankers(args: argparse.Namespace) -> dict[str, Any]:
    ensure_experiment_dirs()
    RERANKER_ROOT.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or run_id("full_scrape_reranker")
    out_dir = assert_experiment_path(RERANKER_ROOT / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    unified = getattr(args, "unified", False)
    if unified:
        searches = tuple(args.search or DEFAULT_SEARCHES_UNIFIED)
    else:
        searches = tuple(args.search or DEFAULT_SEARCHES)

    target_precisions = tuple(float(value) for value in (args.target_precision or DEFAULT_TARGET_PRECISIONS))

    frame = build_training_frame(
        live_run_dir=args.live_run_dir,
        scoring_dir=args.scoring_dir,
        searches=searches,
        horizon_hours=args.horizon_hours,
    )
    frame_path = out_dir / "training_frame.csv"
    frame.to_csv(frame_path, index=False)

    model_metadata: dict[str, Any] = {}
    metric_frames: list[pd.DataFrame] = []

    if unified:
        metadata, metrics = train_unified_model(
            frame,
            searches=searches,
            run_name=run_name,
            out_dir=out_dir,
            target_precisions=target_precisions,
            seed=args.seed,
        )
        model_metadata["__unified__"] = metadata
        metric_frames.append(metrics)
    else:
        for search in searches:
            metadata, metrics = train_search_model(
                frame,
                search=search,
                run_name=run_name,
                out_dir=out_dir,
                target_precisions=target_precisions,
                seed=args.seed,
            )
            model_metadata[search] = metadata
            metric_frames.append(metrics)

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics_path = out_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    manifest = {
        "run_name": run_name,
        "unified": unified,
        "created_from_live_run_dir": str(args.live_run_dir),
        "created_from_scoring_dir": str(args.scoring_dir or latest_live_scoring_dir()),
        "horizon_hours": int(args.horizon_hours),
        "searches": list(searches),
        "target_precisions": list(target_precisions),
        "target_precision_keys": [f"p{int(round(value * 100))}" for value in target_precisions],
        "models": model_metadata,
        "outputs": {
            "training_frame": str(frame_path),
            "metrics": str(metrics_path),
            "report": str(out_dir / "report.md"),
        },
    }
    write_training_report(out_dir / "report.md", metrics, manifest)
    write_json(out_dir / "manifest.json", manifest)
    write_json(RERANKER_ROOT / "latest_manifest.json", manifest)
    return manifest


def load_latest_manifest() -> dict[str, Any] | None:
    path = RERANKER_ROOT / "latest_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_reranker_model(metadata: dict[str, Any]) -> Any:
    with Path(str(metadata["artifact_path"])).open("rb") as handle:
        return pickle.load(handle)


def apply_full_scrape_reranker(
    scored: pd.DataFrame,
    *,
    live_run_dir: Path,
    policy_targets: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = load_latest_manifest()
    out = merge_full_scrape_enrichment(scored, live_run_dir)
    out[f"{RERANKER_PREFIX}Required"] = out.get("SearchName", pd.Series("", index=out.index)).astype(str).isin(policy_targets)
    out[f"{RERANKER_PREFIX}Applied"] = False
    out[f"{RERANKER_PREFIX}Score"] = np.nan
    out[f"{RERANKER_PREFIX}Threshold"] = np.nan
    out[f"{RERANKER_PREFIX}Passed"] = ~out[f"{RERANKER_PREFIX}Required"]
    out[f"{RERANKER_PREFIX}TargetPrecision"] = np.nan
    out[f"{RERANKER_PREFIX}Reason"] = np.where(out[f"{RERANKER_PREFIX}Required"], "missing_reranker", "not_required")

    if not manifest:
        return out, {"status": "missing_manifest", "policy_targets": policy_targets}

    models = manifest.get("models", {})
    applied: dict[str, Any] = {}
    enriched = add_full_scrape_features(out)
    for search, target_precision in policy_targets.items():
        metadata = models.get(search)
        search_mask = enriched.get("SearchName", pd.Series("", index=enriched.index)).astype(str).eq(search)
        if not search_mask.any():
            continue
        if not metadata:
            out.loc[search_mask, f"{RERANKER_PREFIX}Passed"] = False
            out.loc[search_mask, f"{RERANKER_PREFIX}Reason"] = "missing_search_model"
            continue
        target_key = f"p{int(round(float(target_precision) * 100))}"
        threshold_info = metadata.get("thresholds", {}).get(target_key)
        if not threshold_info:
            out.loc[search_mask, f"{RERANKER_PREFIX}Passed"] = False
            out.loc[search_mask, f"{RERANKER_PREFIX}Reason"] = f"missing_threshold_{target_key}"
            continue
        threshold = float(threshold_info.get("threshold", np.inf))
        feature_columns = list(metadata.get("feature_columns", []))
        for col in feature_columns:
            if col not in enriched.columns:
                enriched[col] = "" if col in metadata.get("text_features", []) else np.nan

        if "Description" in out.columns:
            description = out.loc[search_mask, "Description"]
            full_scrape_ok = description.notna() & description.astype(str).str.strip().ne("")
        else:
            full_scrape_ok = pd.Series(False, index=out.loc[search_mask].index)
        model = load_reranker_model(metadata)
        scored_idx = full_scrape_ok[full_scrape_ok].index
        if len(scored_idx):
            scores = np.clip(score_model(model, enriched.loc[scored_idx, feature_columns]), 0.0, 1.0)
            out.loc[scored_idx, f"{RERANKER_PREFIX}Score"] = scores
            out.loc[scored_idx, f"{RERANKER_PREFIX}Threshold"] = threshold
            out.loc[scored_idx, f"{RERANKER_PREFIX}Passed"] = scores >= threshold
            out.loc[scored_idx, f"{RERANKER_PREFIX}Applied"] = True
            out.loc[scored_idx, f"{RERANKER_PREFIX}TargetPrecision"] = float(target_precision)
            out.loc[scored_idx, f"{RERANKER_PREFIX}Reason"] = np.where(
                scores >= threshold,
                f"full_scrape_reranker_pass_{target_key}",
                f"full_scrape_reranker_below_{target_key}",
            )
        missing_idx = full_scrape_ok[~full_scrape_ok].index
        if len(missing_idx):
            out.loc[missing_idx, f"{RERANKER_PREFIX}Passed"] = False
            out.loc[missing_idx, f"{RERANKER_PREFIX}Reason"] = "missing_full_scrape_features"
        applied[search] = {
            "target_precision": float(target_precision),
            "target_key": target_key,
            "threshold": threshold,
            "rows": int(search_mask.sum()),
            "scored_rows": int(len(scored_idx)),
            "passed_rows": int(out.loc[search_mask, f"{RERANKER_PREFIX}Passed"].fillna(False).sum()),
        }

    status = {"status": "applied", "manifest_run": manifest.get("run_name"), "policy_targets": policy_targets, "applied": applied}
    return out, status


def main() -> int:
    all_available_searches = sorted(set(DEFAULT_SEARCHES + DEFAULT_SEARCHES_UNIFIED))
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-run-dir", type=Path, required=True)
    parser.add_argument("--scoring-dir", type=Path, default=None)
    parser.add_argument("--search", action="append", choices=all_available_searches)
    parser.add_argument("--target-precision", action="append", type=float)
    parser.add_argument("--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--unified", action="store_true", help="Train single model across all searches instead of per-search")
    args = parser.parse_args()

    manifest = train_rerankers(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
