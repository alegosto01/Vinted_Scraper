from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageStat

from experiments.current.basic_5_giant_model._deps.photo_arbitrage.dataset import normalize_sources


ImageFile.LOAD_TRUNCATED_IMAGES = True

PHOTO_FEATURES = [
    "PhotoImageCount",
    "PhotoUsableImageCount",
    "PhotoPrimaryWidth",
    "PhotoPrimaryHeight",
    "PhotoPrimaryAspectRatio",
    "PhotoPrimaryBrightness",
    "PhotoPrimaryContrast",
    "PhotoPrimarySaturation",
    "PhotoPrimarySharpness",
    "PhotoPrimaryEdgeDensity",
    "PhotoAvgBrightness",
    "PhotoAvgContrast",
    "PhotoAvgSaturation",
    "PhotoAvgSharpness",
    "PhotoAvgEdgeDensity",
    "PhotoDuplicateImageCount",
    "PhotoLowQualityFraction",
    "PhotoScreenshotRiskFraction",
    "PhotoOnlyOneImage",
    "PhotoMissingImages",
    "PictureCountNum",
]

VALUE_FEATURES = [
    "PriceNum",
    "BrandPresent",
    "ConditionPresent",
    "SellerReviewsNum",
    "SellerStarsNum",
]

MODEL_FEATURES = PHOTO_FEATURES + VALUE_FEATURES


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else value


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        values = frame[column]
    else:
        values = pd.Series([default] * len(frame), index=frame.index)
    return pd.to_numeric(values, errors="coerce").fillna(default)


def average_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size, hash_size))
    arr = np.asarray(gray, dtype=np.float32)
    bits = arr >= arr.mean()
    return "".join("1" if bit else "0" for bit in bits.flatten())


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def image_quality_score(metrics: dict[str, float]) -> tuple[float, float]:
    penalty = 0.0
    if min(metrics["width"], metrics["height"]) < 220:
        penalty += 0.18
    if metrics["brightness"] < 0.18 or metrics["brightness"] > 0.92:
        penalty += 0.14
    if metrics["contrast"] < 0.10:
        penalty += 0.15
    if metrics["sharpness"] < 25.0:
        penalty += 0.18
    screenshot_risk = 0.22 if metrics["aspect_ratio"] > 1.25 or metrics["aspect_ratio"] < 0.60 else 0.0
    return clamp(1.0 - penalty - screenshot_risk), screenshot_risk


def compute_one_image_features(path: str | Path) -> dict[str, float | str]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    small = image.resize((min(width, 256), min(height, 256)))
    gray = small.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    stat = ImageStat.Stat(small)
    gy, gx = np.gradient(arr)
    metrics = {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / height) if height else np.nan,
        "brightness": float(np.mean(stat.mean) / 255.0),
        "contrast": float(np.std(arr) / 255.0),
        "saturation": float((rgb.max(axis=2) - rgb.min(axis=2)).mean()),
        "sharpness": float(np.var(gx) + np.var(gy)),
        "edge_density": float((np.hypot(gx, gy) > 20.0).mean()),
        "hash": average_hash(small),
    }
    quality, screenshot_risk = image_quality_score(metrics)
    metrics["quality_score"] = float(quality)
    metrics["screenshot_risk"] = float(screenshot_risk)
    return metrics


def duplicate_count(hashes: list[str]) -> int:
    unique: list[str] = []
    duplicates = 0
    for value in hashes:
        if any(hamming_distance(value, other) <= 4 for other in unique):
            duplicates += 1
        else:
            unique.append(value)
    return duplicates


def local_image_paths_from_row(row: pd.Series | dict) -> list[str]:
    get = row.get if isinstance(row, dict) else row.get
    paths = normalize_sources(get("LocalImagePaths"))
    primary = str(get("LocalPrimaryImagePath") or "").strip()
    if primary:
        paths = [primary] + [path for path in paths if path != primary]
    existing = []
    for path in paths:
        if path and Path(path).exists() and path not in existing:
            existing.append(path)
    return existing


def compute_row_photo_features(row: pd.Series | dict) -> dict[str, float]:
    paths = local_image_paths_from_row(row)
    picture_count = safe_float(row.get("PictureCount") if isinstance(row, dict) else row.get("PictureCount"), default=np.nan)
    if not paths:
        return {
            **{name: np.nan for name in PHOTO_FEATURES},
            "PhotoImageCount": float(picture_count) if np.isfinite(picture_count) else 0.0,
            "PhotoUsableImageCount": 0.0,
            "PhotoOnlyOneImage": 1.0 if picture_count == 1 else 0.0,
            "PhotoMissingImages": 1.0,
            "PictureCountNum": float(picture_count) if np.isfinite(picture_count) else 0.0,
        }

    metrics = []
    for path in paths:
        try:
            metrics.append(compute_one_image_features(path))
        except Exception:
            continue
    if not metrics:
        return {
            **{name: np.nan for name in PHOTO_FEATURES},
            "PhotoImageCount": float(len(paths)),
            "PhotoUsableImageCount": 0.0,
            "PhotoOnlyOneImage": 1.0 if len(paths) == 1 else 0.0,
            "PhotoMissingImages": 1.0,
            "PictureCountNum": float(picture_count) if np.isfinite(picture_count) else float(len(paths)),
        }

    primary = metrics[0]
    hashes = [str(m["hash"]) for m in metrics]
    low_quality = [float(m["quality_score"]) < 0.45 for m in metrics]
    screenshot = [float(m["screenshot_risk"]) > 0 for m in metrics]
    image_count = max(float(len(paths)), float(picture_count) if np.isfinite(picture_count) else 0.0)
    return {
        "PhotoImageCount": image_count,
        "PhotoUsableImageCount": float(len(metrics)),
        "PhotoPrimaryWidth": float(primary["width"]),
        "PhotoPrimaryHeight": float(primary["height"]),
        "PhotoPrimaryAspectRatio": float(primary["aspect_ratio"]),
        "PhotoPrimaryBrightness": float(primary["brightness"]),
        "PhotoPrimaryContrast": float(primary["contrast"]),
        "PhotoPrimarySaturation": float(primary["saturation"]),
        "PhotoPrimarySharpness": float(primary["sharpness"]),
        "PhotoPrimaryEdgeDensity": float(primary["edge_density"]),
        "PhotoAvgBrightness": float(np.mean([m["brightness"] for m in metrics])),
        "PhotoAvgContrast": float(np.mean([m["contrast"] for m in metrics])),
        "PhotoAvgSaturation": float(np.mean([m["saturation"] for m in metrics])),
        "PhotoAvgSharpness": float(np.mean([m["sharpness"] for m in metrics])),
        "PhotoAvgEdgeDensity": float(np.mean([m["edge_density"] for m in metrics])),
        "PhotoDuplicateImageCount": float(duplicate_count(hashes)),
        "PhotoLowQualityFraction": float(np.mean(low_quality)),
        "PhotoScreenshotRiskFraction": float(np.mean(screenshot)),
        "PhotoOnlyOneImage": 1.0 if image_count <= 1 else 0.0,
        "PhotoMissingImages": 0.0,
        "PictureCountNum": image_count,
    }


def compute_value_features(row: pd.Series | dict) -> dict[str, float]:
    get = row.get if isinstance(row, dict) else row.get
    brand = str(get("Brand") or "").strip().lower()
    condition = str(get("Condition") or "").strip().lower()
    return {
        "PriceNum": safe_float(get("Price"), default=np.nan),
        "BrandPresent": 1.0 if brand and brand not in {"0", "nan", "no brand", "none"} else 0.0,
        "ConditionPresent": 1.0 if condition and condition not in {"unknown", "nan", "none"} else 0.0,
        "SellerReviewsNum": safe_float(get("ReviewsCount"), default=np.nan),
        "SellerStarsNum": safe_float(get("Stars"), default=np.nan),
    }


def add_photo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rows = []
    for _, row in out.iterrows():
        feature_row = {}
        feature_row.update(compute_row_photo_features(row))
        feature_row.update(compute_value_features(row))
        rows.append(feature_row)
    features = pd.DataFrame(rows, index=out.index)
    for col in MODEL_FEATURES:
        out[col] = features[col] if col in features.columns else np.nan
    return out


def heuristic_bad_photo_probability(frame: pd.DataFrame) -> pd.Series:
    df = frame.copy()
    missing = numeric_series(df, "PhotoMissingImages", 0)
    low_quality = numeric_series(df, "PhotoLowQualityFraction", 0.5)
    one_image = numeric_series(df, "PhotoOnlyOneImage", 0)
    duplicate_count_num = numeric_series(df, "PhotoDuplicateImageCount", 0)
    duplicate_penalty = np.clip(duplicate_count_num / 3.0, 0, 1)
    contrast = numeric_series(df, "PhotoPrimaryContrast", np.nan)
    sharpness = numeric_series(df, "PhotoPrimarySharpness", np.nan)
    bad_contrast = contrast.lt(0.10).fillna(False).astype(float)
    bad_sharpness = sharpness.lt(25.0).fillna(False).astype(float)
    prob = (
        0.15
        + 0.35 * low_quality
        + 0.15 * one_image
        + 0.10 * duplicate_penalty
        + 0.10 * bad_contrast
        + 0.10 * bad_sharpness
        + 0.20 * missing
    )
    return pd.Series(np.clip(prob, 0.0, 1.0), index=frame.index)


def value_signal(frame: pd.DataFrame) -> pd.Series:
    price = numeric_series(frame, "PriceNum", np.nan)
    price_score = (1.0 - ((price - 20.0) / 180.0)).clip(0.0, 1.0).fillna(0.35)
    brand = numeric_series(frame, "BrandPresent", 0)
    condition = numeric_series(frame, "ConditionPresent", 0)
    picture_count = numeric_series(frame, "PictureCountNum", 0)
    picture_context = np.where(picture_count >= 2, 1.0, 0.25)
    return pd.Series(
        np.clip(0.25 * brand + 0.35 * price_score + 0.15 * condition + 0.10 * picture_context + 0.15, 0, 1),
        index=frame.index,
    )


def risk_penalty(frame: pd.DataFrame) -> pd.Series:
    price = numeric_series(frame, "PriceNum", np.nan)
    no_image = numeric_series(frame, "PhotoMissingImages", 0)
    no_brand = 1.0 - numeric_series(frame, "BrandPresent", 0)
    high_price = price.gt(250).fillna(False).astype(float)
    return pd.Series(np.clip(0.35 * no_image + 0.10 * no_brand + 0.20 * high_price, 0, 1), index=frame.index)


def score_reason_notes(row: pd.Series) -> str:
    notes = []
    if safe_float(row.get("PhotoMissingImages"), 0) >= 1:
        notes.append("missing_local_images")
    if safe_float(row.get("PhotoLowQualityFraction"), 0) >= 0.5:
        notes.append("many_low_quality_images")
    if safe_float(row.get("PhotoOnlyOneImage"), 0) >= 1:
        notes.append("only_one_photo")
    if safe_float(row.get("BrandPresent"), 0) < 1:
        notes.append("missing_brand")
    if safe_float(row.get("ConditionPresent"), 0) >= 1:
        notes.append("condition_available")
    if safe_float(row.get("PictureCountNum"), 0) >= 3:
        notes.append("multiple_photos_available")
    return "|".join(notes)


def add_business_scores(df: pd.DataFrame, bad_probability_col: str = "BadPhotoProbability") -> pd.DataFrame:
    out = df.copy()
    bad = numeric_series(out, bad_probability_col, np.nan).fillna(heuristic_bad_photo_probability(out))
    out["PhotoValueSignal"] = value_signal(out)
    out["PhotoRiskPenalty"] = risk_penalty(out)
    out["PhotoOpportunityScore"] = np.clip(0.55 * bad + 0.35 * out["PhotoValueSignal"] - 0.25 * out["PhotoRiskPenalty"], 0, 1)
    out["PhotoOpportunityNotes"] = [score_reason_notes(row) for _, row in out.iterrows()]
    return out
