"""Text features used by the live time-to-sell pipeline."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


URL_RE = re.compile(r"https?://\S+|www\.\S+")
TOKEN_RE = re.compile(r"[a-z0-9]+")
PRICE_LIKE_RE = re.compile(
    r"(?:\b\d{1,3}(?:[.,]\d{1,2})?\s*[€$£]|\d{1,3}(?:[.,]\d{1,2})?\s*(?:euro|eur)\b|"
    r"[€$£]\s*\d{1,3}(?:[.,]\d{1,2})?)",
    flags=re.IGNORECASE,
)
NEW_WORD_RE = re.compile(r"\b(?:new|nuov[aoei]?|nueva|neuf|neu|nwt|cartellino)\b")
AUTH_WORD_RE = re.compile(r"\b(?:original|authentic|autentico|autentica|vero|genuine)\b")
LIMITED_WORD_RE = re.compile(r"\b(?:limited edition|edizione limitata|raro|rara|rare)\b")
BUNDLE_WORD_RE = re.compile(r"\b(?:lotto|bundle|set|stock|blocco)\b")
DEFECT_WORD_RE = re.compile(r"\b(?:difett\w*|rovinat\w*|usurat\w*|macchiat\w*|strapp\w*|da riparare)\b")

TITLE_TEXT_FEATURES = ["TitleTextNormalized"]
TITLE_NUMERIC_FEATURES = [
    "title_char_len_tts",
    "title_token_count_tts",
    "title_unique_token_count_tts",
    "title_digit_token_count_tts",
    "title_keyword_positive_count_tts",
    "title_keyword_caution_count_tts",
]
TITLE_BINARY_FEATURES = [
    "title_has_digit_tts",
    "title_has_new_word_tts",
    "title_has_auth_word_tts",
    "title_has_limited_word_tts",
    "title_has_bundle_word_tts",
    "title_has_defect_word_tts",
    "title_has_price_like_number_tts",
]
TITLE_FEATURE_COLUMNS = [*TITLE_TEXT_FEATURES, *TITLE_NUMERIC_FEATURES, *TITLE_BINARY_FEATURES]


def normalize_title_text(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = URL_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_title_feature_frame(series: pd.Series) -> pd.DataFrame:
    raw = series.fillna("").astype(str)
    normalized = raw.map(normalize_title_text)
    tokens = normalized.str.findall(TOKEN_RE)

    out = pd.DataFrame(index=series.index)
    out["TitleTextNormalized"] = normalized
    out["title_char_len_tts"] = normalized.str.len()
    out["title_token_count_tts"] = tokens.str.len()
    out["title_unique_token_count_tts"] = tokens.map(lambda values: len(set(values)))
    out["title_digit_token_count_tts"] = tokens.map(
        lambda values: sum(any(ch.isdigit() for ch in token) for token in values)
    )

    has_new = normalized.str.contains(NEW_WORD_RE, regex=True, na=False)
    has_auth = normalized.str.contains(AUTH_WORD_RE, regex=True, na=False)
    has_limited = normalized.str.contains(LIMITED_WORD_RE, regex=True, na=False)
    has_bundle = normalized.str.contains(BUNDLE_WORD_RE, regex=True, na=False)
    has_defect = normalized.str.contains(DEFECT_WORD_RE, regex=True, na=False)
    has_price_like = raw.str.contains(PRICE_LIKE_RE, regex=True, na=False)

    out["title_has_digit_tts"] = normalized.str.contains(r"\d", regex=True, na=False).astype(int)
    out["title_has_new_word_tts"] = has_new.astype(int)
    out["title_has_auth_word_tts"] = has_auth.astype(int)
    out["title_has_limited_word_tts"] = has_limited.astype(int)
    out["title_has_bundle_word_tts"] = has_bundle.astype(int)
    out["title_has_defect_word_tts"] = has_defect.astype(int)
    out["title_has_price_like_number_tts"] = has_price_like.astype(int)
    out["title_keyword_positive_count_tts"] = has_new.astype(int) + has_auth.astype(int) + has_limited.astype(int)
    out["title_keyword_caution_count_tts"] = has_bundle.astype(int) + has_defect.astype(int) + has_price_like.astype(int)
    return out


def add_title_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    title = out["Title"] if "Title" in out.columns else pd.Series([""] * len(out), index=out.index)
    features = build_title_feature_frame(title)
    for column in TITLE_FEATURE_COLUMNS:
        out[column] = features[column]
    return out
