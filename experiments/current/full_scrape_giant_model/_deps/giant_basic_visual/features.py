from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.full_scrape_giant_model._deps.full_scrape_model.compare_feature_modalities import add_identity  # noqa: E402
from experiments.current.full_scrape_giant_model._deps.photo_arbitrage.features import compute_one_image_features  # noqa: E402

from .paths import ROOT  # noqa: E402


FEATURE_MODES = (
    "basic_5_control",
    "main_image_simple",
    "main_image_scores",
    "main_image_dino_outlier_diagnostic",
)

MAIN_IMAGE_FEATURES = [
    "MainImageWidth",
    "MainImageHeight",
    "MainImageAspectRatio",
    "MainImageBrightness",
    "MainImageContrast",
    "MainImageSaturation",
    "MainImageSharpness",
    "MainImageEdgeDensity",
    "MainImageQualityScore",
    "MainImageScreenshotRisk",
]

MAIN_IMAGE_SCORE_FEATURES = [
    "SimpleBadPhotoScore",
    "AestheticGoodScore",
    "AestheticBadPhotoScore",
    "DinoEmbeddingNorm",
    "CombinedBadPhotoScore",
]

MAIN_IMAGE_DIAGNOSTIC_FEATURES = ["DinoOutlierScore"]
QUALITY_SCORE_COLUMNS = MAIN_IMAGE_SCORE_FEATURES + MAIN_IMAGE_DIAGNOSTIC_FEATURES
RAW_DINO_COLUMNS = {"DinoEmbedding", "DinoEmbeddingDim"}
RAW_DINO_PREFIX = "DinoEmbedding_"

SKIP_COLUMNS = ["SearchName", "item_id", "Dataid", "Title", "Link", "LocalPrimaryImagePath", "skip_reason"]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return not text or text.lower() in {"nan", "none", "null"}


def resolve_main_image_path(value: object) -> tuple[Path | None, str]:
    """Resolve only LocalPrimaryImagePath; never inspect LocalImagePaths."""
    if _is_missing(value):
        return None, "missing_main_image"
    raw = str(value).strip()
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [path, ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate, ""
    return candidates[-1], "main_image_not_found"


def main_image_features_for_path(path: str | Path) -> dict[str, float]:
    metrics = compute_one_image_features(path)
    return {
        "MainImageWidth": float(metrics.get("width", np.nan)),
        "MainImageHeight": float(metrics.get("height", np.nan)),
        "MainImageAspectRatio": float(metrics.get("aspect_ratio", np.nan)),
        "MainImageBrightness": float(metrics.get("brightness", np.nan)),
        "MainImageContrast": float(metrics.get("contrast", np.nan)),
        "MainImageSaturation": float(metrics.get("saturation", np.nan)),
        "MainImageSharpness": float(metrics.get("sharpness", np.nan)),
        "MainImageEdgeDensity": float(metrics.get("edge_density", np.nan)),
        "MainImageQualityScore": float(metrics.get("quality_score", np.nan)),
        "MainImageScreenshotRisk": float(metrics.get("screenshot_risk", np.nan)),
    }


def _skip_row(row: pd.Series, reason: str) -> dict[str, Any]:
    payload = {col: row.get(col, "") for col in SKIP_COLUMNS if col != "skip_reason"}
    payload["skip_reason"] = reason
    return payload


def filter_rows_with_main_image(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows with readable main images and append main-image-only features."""
    source = add_identity(frame)
    feature_cache: dict[str, dict[str, float]] = {}
    kept_rows: list[pd.Series] = []
    kept_features: list[dict[str, float]] = []
    skipped: list[dict[str, Any]] = []

    for _, row in source.iterrows():
        path, reason = resolve_main_image_path(row.get("LocalPrimaryImagePath"))
        if reason:
            skipped.append(_skip_row(row, reason))
            continue
        assert path is not None
        cache_key = str(path.resolve())
        if cache_key not in feature_cache:
            try:
                feature_cache[cache_key] = main_image_features_for_path(path)
            except Exception as exc:
                skipped.append(_skip_row(row, f"main_image_unreadable:{type(exc).__name__}"))
                continue
        kept_rows.append(row.copy())
        kept_features.append(feature_cache[cache_key])

    if not kept_rows:
        kept = source.iloc[0:0].copy()
        for col in MAIN_IMAGE_FEATURES:
            kept[col] = pd.Series(dtype=float)
    else:
        kept = pd.DataFrame(kept_rows).reset_index(drop=True)
        feature_df = pd.DataFrame(kept_features)
        kept = pd.concat([kept.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    skipped_df = pd.DataFrame(skipped, columns=SKIP_COLUMNS)
    return kept.reset_index(drop=True), skipped_df


def read_quality_scores(path: Path | str | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["SearchName", "item_id", *QUALITY_SCORE_COLUMNS])
    source = Path(path)
    if not source.exists():
        return pd.DataFrame(columns=["SearchName", "item_id", *QUALITY_SCORE_COLUMNS])
    header = pd.read_csv(source, nrows=0)
    usecols = [col for col in ["SearchName", "item_id", "Dataid", "Link", *QUALITY_SCORE_COLUMNS] if col in header.columns]
    visual = pd.read_csv(source, usecols=usecols, low_memory=False)
    visual = add_identity(visual)
    visual = visual[visual["item_id"].astype(str).str.len() > 0].copy()
    keep = ["SearchName", "item_id"] + [col for col in QUALITY_SCORE_COLUMNS if col in visual.columns]
    visual = visual[keep].drop_duplicates(subset=["SearchName", "item_id"], keep="last")
    return visual.reset_index(drop=True)


def merge_quality_scores(frame: pd.DataFrame, visual_source: Path | str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = add_identity(frame)
    visual = read_quality_scores(visual_source)
    if visual.empty:
        for col in QUALITY_SCORE_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
        return out, {"visual_source": str(visual_source or ""), "matched_rows": 0, "score_rows": 0}

    before_cols = set(out.columns)
    merged = out.merge(visual, on=["SearchName", "item_id"], how="left", suffixes=("", "__visual"))
    for col in QUALITY_SCORE_COLUMNS:
        visual_col = f"{col}__visual"
        if visual_col in merged.columns:
            if col in before_cols:
                merged[col] = merged[col].where(merged[col].notna(), merged[visual_col])
                merged.drop(columns=[visual_col], inplace=True)
            else:
                merged.rename(columns={visual_col: col}, inplace=True)
        if col not in merged.columns:
            merged[col] = np.nan
    matched = int(merged[QUALITY_SCORE_COLUMNS].notna().any(axis=1).sum())
    return merged.reset_index(drop=True), {
        "visual_source": str(visual_source or ""),
        "matched_rows": matched,
        "score_rows": int(len(visual)),
    }


def feature_columns_for_mode(mode: str) -> list[str]:
    if mode == "basic_5_control":
        return []
    if mode == "main_image_simple":
        return list(MAIN_IMAGE_FEATURES)
    if mode == "main_image_scores":
        return [*MAIN_IMAGE_FEATURES, *MAIN_IMAGE_SCORE_FEATURES]
    if mode == "main_image_dino_outlier_diagnostic":
        return [*MAIN_IMAGE_FEATURES, *MAIN_IMAGE_SCORE_FEATURES, *MAIN_IMAGE_DIAGNOSTIC_FEATURES]
    raise ValueError(f"Unknown feature mode: {mode}")


def mode_live_ready(mode: str) -> bool:
    return mode in {"basic_5_control", "main_image_simple", "main_image_scores"}


def is_diagnostic_mode(mode: str) -> bool:
    return mode == "main_image_dino_outlier_diagnostic"


def assert_no_raw_dino_features(features: list[str]) -> None:
    illegal = sorted(feature for feature in features if feature in RAW_DINO_COLUMNS or feature.startswith(RAW_DINO_PREFIX))
    if illegal:
        raise ValueError(f"Raw DINO embedding columns are not allowed: {', '.join(illegal)}")

