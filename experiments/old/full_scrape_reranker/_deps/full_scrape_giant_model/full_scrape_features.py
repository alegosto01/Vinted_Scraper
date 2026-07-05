#!/usr/bin/env python3
"""Generic full-scrape feature engineering shared by full_scrape_giant_model scripts."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
for _path in (ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.old.full_scrape_reranker._deps.full_scrape_giant_model._deps.basic_5_giant_model.title_features import (  # noqa: E402
    TITLE_BINARY_FEATURES,
    TITLE_NUMERIC_FEATURES,
    build_title_feature_frame,
    raw_title_token_count,
)

SCORE_PREFIX = "score__"
THRESHOLD_PREFIX = "threshold__"


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
    return raw_title_token_count(series)


def model_score_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith(SCORE_PREFIX))


def add_full_scrape_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("Title", "Description", "Brand", "Condition", "Location"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    out["TitleText"] = out["Title"].fillna("").astype(str)
    out["DescriptionText"] = out["Description"].fillna("").astype(str)
    title_features = build_title_feature_frame(out["TitleText"])
    out = pd.concat([out, title_features], axis=1)
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


__all__ = [
    "SCORE_PREFIX",
    "THRESHOLD_PREFIX",
    "TITLE_BINARY_FEATURES",
    "TITLE_NUMERIC_FEATURES",
    "add_full_scrape_features",
    "clean_negative_missing",
    "location_country",
    "model_score_columns",
    "parse_upload_minutes",
    "safe_float_series",
    "text_token_count",
]
