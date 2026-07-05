"""Shared configuration and helpers for the sold-within-72h analysis."""

from pathlib import Path

PROJECT_ROOT = Path("/home/ale/Desktop/vinted/Vinted_New_Version")

SOURCE_CSV = (
    PROJECT_ROOT
    / "experiments/old/full_scrape_reranker/data/full_scrape_reranker/full_scrape_reranker_20260611_192441/training_frame.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "experiments/current/basic_5_giant_model/data/analysis/sold_vs_not_72h"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The 9 Basic-5 Giant stage-1 models under comparison.
MODELS = [
    "logistic_v1_baseline",
    "logistic_snapshot_v2",
    "sgd_text_numeric_v1",
    "linear_svm_calibrated_v1",
    "numeric_tree_v1",
    "rules_price_v1",
    "random_forest_basic_v1",
    "hist_gradient_basic_numeric_v1",
    "xgboost_basic_v1",
]

PASS_COLS = [f"pass__{m}" for m in MODELS]
SCORE_COLS = [f"score__{m}" for m in MODELS]
THRESHOLD_COLS = [f"threshold__{m}" for m in MODELS]
MARGIN_COLS = [f"margin__{m}" for m in MODELS]

# Searches requested for this analysis. The source frame also contains a few
# newer searches; keep this list pinned to the six cohorts in the request.
SEARCHES = [
    "nike",
    "griffati_donna_all",
    "griffati_uomo_all",
    "gucci",
    "prada",
    "ps4",
]

OUTCOME_COL = "sold_within_72h"
EVAL_COL = "evaluated_at_72h"
STATUS_COL = "status_at_72h"

# Columns that belong to the reranker model itself and must not leak into the
# analysis of the stage-1 pass cohorts.
RERANKER_LEAK_COLS = [
    "target",
    "Stage2Passed",
    "Stage2Model",
    "Stage2Score",
    "Stage2Threshold",
    "FullScrapeRerankerPassed",
]

# Numeric columns to compare between sold-72h and not-sold-72h groups.
NUMERIC_FEATURES = [
    "Price",
    "Likes",
    "Page",
    "log_price",
    "log_likes",
    "title_char_len",
    "title_token_count",
    "title_norm_char_len",
    "title_norm_token_count",
    "title_unique_token_count",
    "title_duplicate_token_ratio",
    "title_avg_token_len",
    "title_punct_count",
    "title_upper_ratio",
    "title_char_len_full",
    "title_token_count_full",
    "title_char_len_full_norm",
    "title_token_count_full_norm",
    "title_unique_token_count_full",
    "title_digit_token_count_full",
    "title_keyword_positive_count_full",
    "title_keyword_caution_count_full",
    "description_char_len",
    "description_token_count",
    "upload_age_minutes",
    "ReviewsCount",
    "Stars",
    "reviewscount_raw",
    "stars_raw",
    "Interested_count",
    "View_count",
    "interested_count_raw",
    "view_count_raw",
    "VisiblePictureCount",
    "HiddenPictureCount",
    "PictureCount",
] + SCORE_COLS + MARGIN_COLS

# Categorical columns to compare.
CATEGORICAL_FEATURES = [
    "SearchName",
    "Brand",
    "Condition",
    "LocationCountry",
    "Size",
]

# Missingness / binary flags.
BINARY_FEATURES = [
    "reviews_missing",
    "stars_missing",
    "interested_missing",
    "views_missing",
    "picture_count_missing",
    "seller_has_reviews",
    "brand_present",
    "title_has_digit",
    "title_has_new_word",
    "title_has_auth_word",
    "title_has_digit_full",
    "title_has_new_word_full",
    "title_has_auth_word_full",
    "title_has_limited_word_full",
    "title_has_bundle_word_full",
    "title_has_defect_word_full",
    "title_has_price_like_number_full",
]

# Columns we keep from the source frame for the analysis.
# We keep SearchName as the canonical search identifier; any search__* one-hot
# columns are left in place because they are part of the existing frame.
KEEP_CORE_COLS = (
    ["tracking_key", "item_id", "Dataid", "Link", "Title", "Description",
     "SearchName", OUTCOME_COL, EVAL_COL, STATUS_COL]
    + ["FullScrapeStatus"]
    + PASS_COLS
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + BINARY_FEATURES
    + RERANKER_LEAK_COLS  # kept so we can drop them explicitly in each step
)


def coerce_bool_series(series):
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0.0)

    truthy = {"true", "1", "yes", "y", "t"}
    return series.fillna(False).map(lambda v: str(v).strip().lower() in truthy)


def load_source_frame():
    import pandas as pd

    df = pd.read_csv(SOURCE_CSV)
    # Ensure booleans are real booleans.
    for col in PASS_COLS + [OUTCOME_COL]:
        if col in df.columns:
            df[col] = coerce_bool_series(df[col])
    return df


def valid_72h_mask(df):
    """Return a boolean mask for rows with a clean 72 h outcome."""
    return (
        df[EVAL_COL].notna()
        & df[STATUS_COL].isin(["Sold", "OnSale"])
    )


def analysis_search_mask(df):
    """Return a mask for the six searches requested in the analysis plan."""
    return df["SearchName"].isin(SEARCHES)


def analysis_mask(df):
    """Return rows with a clean 72 h label in the requested searches."""
    return valid_72h_mask(df) & analysis_search_mask(df)
