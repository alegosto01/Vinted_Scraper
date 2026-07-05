#!/usr/bin/env python3
"""Build TF-IDF + SVD text features for the sold-vs-not-72h analysis."""

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[5]
SCRIPTS_DIR = ROOT / "scripts"
for _path in (ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Shared configuration
import common
from experiments.current.full_scrape_giant_model._deps.basic_5_giant_model.title_features import (
    TITLE_BINARY_FEATURES,
    TITLE_NUMERIC_FEATURES,
    build_title_feature_frame,
)

try:
    import nltk
except Exception:  # pragma: no cover - optional dependency fallback
    nltk = None


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = common.OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PKL_PATH = OUTPUT_DIR / "text_features.pkl"
CSV_PATH = OUTPUT_DIR / "text_features.csv"
TITLE_MODEL_PATH = OUTPUT_DIR / "title_tfidf_svd.joblib"
DESC_MODEL_PATH = OUTPUT_DIR / "desc_tfidf_svd.joblib"

REQUIRED_COLS = [
    "tracking_key",
    "item_id",
    "SearchName",
    common.OUTCOME_COL,
    "Title",
    "Description",
]

SVD_RANDOM_STATE = 42
N_SVD_COMPONENTS = 50

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TOKEN_RE = re.compile(r"\b[a-z]{2,}\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_nltk_stopwords():
    fallback = [
        "a", "ad", "al", "alla", "che", "con", "da", "del", "della",
        "di", "e", "il", "in", "la", "le", "lo", "non", "per", "un",
        "una", "su", "sono", "come", "piu", "anche", "o", "ho",
    ]
    if nltk is None:
        return fallback
    try:
        nltk.data.find("corpora/stopwords")
        return nltk.corpus.stopwords.words("italian")
    except LookupError:
        return fallback


def clean_text(series: pd.Series) -> pd.Series:
    """Fill NaNs, lowercase, collapse whitespace, remove URLs."""
    s = series.fillna("").astype(str)
    s = s.str.lower()
    s = s.str.replace(URL_RE, "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def token_set(text: str) -> set:
    return set(TOKEN_RE.findall(text))


def jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def fit_tfidf_svd(texts, stop_words):
    tfidf = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words=stop_words,
        lowercase=False,
        token_pattern=r"(?u)\b[a-z]{2,}\b",
    )
    try:
        X_tfidf = tfidf.fit_transform(texts)
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            n_docs = len(texts)
            return None, None, np.zeros((n_docs, N_SVD_COMPONENTS))
        raise

    n_components = min(N_SVD_COMPONENTS, X_tfidf.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=SVD_RANDOM_STATE)
    X_svd = svd.fit_transform(X_tfidf)
    # Pad with zeros if fewer components were available than requested.
    if X_svd.shape[1] < N_SVD_COMPONENTS:
        pad_width = N_SVD_COMPONENTS - X_svd.shape[1]
        X_svd = np.pad(
            X_svd, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
        )
    return tfidf, svd, X_svd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading source frame...")
    df = common.load_source_frame()
    print(f"  source shape: {df.shape}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    mask = common.analysis_mask(df)
    df = df.loc[mask, REQUIRED_COLS].copy()
    print(f"  valid 72h rows: {len(df)}")

    # Ensure outcome is boolean
    df[common.OUTCOME_COL] = df[common.OUTCOME_COL].astype(bool)

    print("Cleaning text...")
    title_feature_frame = build_title_feature_frame(df["Title"])
    df = pd.concat([df, title_feature_frame], axis=1)
    df["title_clean"] = df["TitleTextNormalized"]
    df["desc_clean"] = clean_text(df["Description"])

    print("Downloading/loading Italian stopwords...")
    italian_stopwords = ensure_nltk_stopwords()

    print("Fitting TF-IDF + SVD on Title...")
    title_tfidf, title_svd, title_svd_X = fit_tfidf_svd(df["title_clean"], italian_stopwords)

    print("Fitting TF-IDF + SVD on Description...")
    desc_tfidf, desc_svd, desc_svd_X = fit_tfidf_svd(df["desc_clean"], italian_stopwords)

    # Build scalar text features
    print("Building scalar text features...")
    df["title_len"] = df["title_clean"].str.len().astype(int)
    df["desc_len"] = df["desc_clean"].str.len().astype(int)

    title_tokens = df["title_clean"].apply(lambda t: TOKEN_RE.findall(t))
    desc_tokens = df["desc_clean"].apply(lambda t: TOKEN_RE.findall(t))
    df["title_tokens"] = title_tokens.apply(len).astype(int)
    df["desc_tokens"] = desc_tokens.apply(len).astype(int)

    title_token_sets = df["title_clean"].apply(token_set)
    desc_token_sets = df["desc_clean"].apply(token_set)
    df["title_desc_overlap_jaccard"] = [
        jaccard(a, b) for a, b in zip(title_token_sets, desc_token_sets)
    ]

    desc_lower = df["desc_clean"]
    df["title_has_mai_indossato_usato"] = df["title_clean"].str.contains(
        r"\bmai indossato\b|\bmai usato\b", regex=True, na=False
    ).astype(int)
    df["desc_has_spedisco"] = desc_lower.str.contains(r"\bspedisco\b", regex=True, na=False).astype(
        int
    )
    df["desc_has_affare"] = desc_lower.str.contains(r"\baffare\b", regex=True, na=False).astype(int)

    # Assemble output frame
    scalar_cols = [
        "title_len",
        "desc_len",
        "title_tokens",
        "desc_tokens",
        "title_desc_overlap_jaccard",
        "title_has_mai_indossato_usato",
        "desc_has_spedisco",
        "desc_has_affare",
    ]
    scalar_cols.extend(TITLE_NUMERIC_FEATURES)
    scalar_cols.extend(TITLE_BINARY_FEATURES)

    title_svd_cols = [f"title_svd_{i}" for i in range(title_svd_X.shape[1])]
    desc_svd_cols = [f"desc_svd_{i}" for i in range(desc_svd_X.shape[1])]

    meta = pd.DataFrame(
        {
            "tracking_key": df["tracking_key"].values,
            "item_id": df["item_id"].values,
            "SearchName": df["SearchName"].values,
            common.OUTCOME_COL: df[common.OUTCOME_COL].values,
        }
    )
    scalar_df = df[scalar_cols].reset_index(drop=True)
    title_svd_df = pd.DataFrame(title_svd_X, columns=title_svd_cols)
    desc_svd_df = pd.DataFrame(desc_svd_X, columns=desc_svd_cols)

    out = pd.concat([meta, scalar_df, title_svd_df, desc_svd_df], axis=1)

    print(f"Saving outputs to {OUTPUT_DIR}...")
    out.to_pickle(PKL_PATH)
    out.to_csv(CSV_PATH, index=False)

    joblib.dump({"tfidf": title_tfidf, "svd": title_svd}, TITLE_MODEL_PATH)
    joblib.dump({"tfidf": desc_tfidf, "svd": desc_svd}, DESC_MODEL_PATH)

    print("Done.")
    print(f"  text feature shape: {out.shape}")
    print(f"  sold_within_72h rate: {out[common.OUTCOME_COL].mean():.4f}")


if __name__ == "__main__":
    main()
