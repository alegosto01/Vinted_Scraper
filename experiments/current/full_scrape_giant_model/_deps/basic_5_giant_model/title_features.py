"""Shared title normalization and feature engineering for basic_5_giant_model."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")
RAW_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
PRICE_LIKE_RE = re.compile(
    r"(?:\b\d{1,3}(?:[.,]\d{1,2})?\s*[€$£]|\d{1,3}(?:[.,]\d{1,2})?\s*(?:euro|eur)\b|"
    r"[€$£]\s*\d{1,3}(?:[.,]\d{1,2})?)",
    flags=re.IGNORECASE,
)
NEW_WORD_RE = re.compile(r"\b(?:new|nuov[aoei]?|nueva|neuf|neu|nwt|cartellino)\b")
AUTH_WORD_RE = re.compile(r"\b(?:original|authentic|autentico|autentica|vero|genuine)\b")
LIMITED_WORD_RE = re.compile(r"\b(?:limited edition|edizione limitata|raro|rara|rare)\b")
BUNDLE_WORD_RE = re.compile(r"\b(?:lotto|bundle|set|stock|blocco)\b")
DEFECT_WORD_RE = re.compile(
    r"\b(?:difett\w*|rovinat\w*|usurat\w*|macchiat\w*|strapp\w*|da riparare)\b"
)

TITLE_NUMERIC_FEATURES = [
    "title_char_len_full_norm",
    "title_token_count_full_norm",
    "title_unique_token_count_full",
    "title_digit_token_count_full",
    "title_keyword_positive_count_full",
    "title_keyword_caution_count_full",
]

TITLE_BINARY_FEATURES = [
    "title_has_digit_full",
    "title_has_new_word_full",
    "title_has_auth_word_full",
    "title_has_limited_word_full",
    "title_has_bundle_word_full",
    "title_has_defect_word_full",
    "title_has_price_like_number_full",
]


def normalize_title_text(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = URL_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
def raw_title_token_count(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.findall(RAW_TOKEN_RE).str.len()


def build_title_feature_frame(series: pd.Series) -> pd.DataFrame:
    raw = series.fillna("").astype(str)
    normalized = raw.map(normalize_title_text)
    tokens = normalized.str.findall(TITLE_TOKEN_RE)

    feature_frame = pd.DataFrame(index=series.index)
    feature_frame["TitleTextNormalized"] = normalized
    feature_frame["title_char_len_full_norm"] = normalized.str.len()
    feature_frame["title_token_count_full_norm"] = tokens.str.len()
    feature_frame["title_unique_token_count_full"] = tokens.map(lambda vals: len(set(vals)))
    feature_frame["title_digit_token_count_full"] = tokens.map(
        lambda vals: sum(any(ch.isdigit() for ch in token) for token in vals)
    )

    has_new = normalized.str.contains(NEW_WORD_RE, regex=True, na=False)
    has_auth = normalized.str.contains(AUTH_WORD_RE, regex=True, na=False)
    has_limited = normalized.str.contains(LIMITED_WORD_RE, regex=True, na=False)
    has_bundle = normalized.str.contains(BUNDLE_WORD_RE, regex=True, na=False)
    has_defect = normalized.str.contains(DEFECT_WORD_RE, regex=True, na=False)
    has_price_like = raw.str.contains(PRICE_LIKE_RE, regex=True, na=False)

    feature_frame["title_has_digit_full"] = normalized.str.contains(r"\d", regex=True, na=False).astype(int)
    feature_frame["title_has_new_word_full"] = has_new.astype(int)
    feature_frame["title_has_auth_word_full"] = has_auth.astype(int)
    feature_frame["title_has_limited_word_full"] = has_limited.astype(int)
    feature_frame["title_has_bundle_word_full"] = has_bundle.astype(int)
    feature_frame["title_has_defect_word_full"] = has_defect.astype(int)
    feature_frame["title_has_price_like_number_full"] = has_price_like.astype(int)
    feature_frame["title_keyword_positive_count_full"] = (
        has_new.astype(int) + has_auth.astype(int) + has_limited.astype(int)
    )
    feature_frame["title_keyword_caution_count_full"] = (
        has_bundle.astype(int) + has_defect.astype(int) + has_price_like.astype(int)
    )
    return feature_frame
