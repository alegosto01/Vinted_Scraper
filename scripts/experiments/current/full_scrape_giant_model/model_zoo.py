#!/usr/bin/env python3
"""Model zoo for the full-scrape giant model family.

Trains one global sold/not-sold model across the main searches using the
*full-scrape* feature set (numeric item/seller metadata + categorical
Brand/Condition/Country/Search + Title/Description text) instead of the
basic-5 fields used by ``basic_5_giant_model``.

The zoo reuses the nine basic-5 model kinds as baselines (linear, SVM,
extra-trees, rules, random-forest, hist-gradient, xgboost) and adds five new
models that exploit the richer feature set: LightGBM, CatBoost, a tuned
ExtraTrees, a tuned HistGradientBoosting, and a stacking ensemble.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder import model_sweep as base_sweep  # noqa: E402

SEED = base_sweep.DEFAULT_SEED

# --- Full-scrape feature pools (engineered by add_full_scrape_features) -------
# Stage1* score/threshold/rank columns are live-scoring artifacts and are
# intentionally excluded: they do not exist in the offline backfill dataset.
FULL_NUMERIC = [
    "Price",
    "Likes",
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
]
FULL_CATEGORICAL = ["Condition", "Brand", "LocationCountry", "SearchName"]
FULL_TEXT = ["TitleText", "DescriptionText"]

# Engineered snapshot numerics added by base_sweep.add_engineered_snapshot_features.
ENGINEERED_EXTRA = [c for c in base_sweep.ENGINEERED_NUMERIC]

# Kinds whose estimators require a DENSE design matrix. For these we use a
# dense one-hot encoder and drop the (sparse) TF-IDF text block.
DENSE_KINDS = {"hist_gradient_boosting", "hist_gradient_tuned", "catboost"}
# Kinds that benefit from standardized numeric inputs.
LINEAR_KINDS = {"logistic", "sgd", "linear_svm_calibrated"}


@dataclass(frozen=True)
class GiantSpec:
    name: str
    kind: str
    use_text: bool = True
    use_categorical: bool = True
    use_numeric: bool = True
    use_engineered: bool = False
    origin: str = "basic5"  # "basic5" (reused baseline) or "new"
    notes: str = ""


# Nine basic-5 baselines (reused kinds) + five new full-scrape models.
ZOO: tuple[GiantSpec, ...] = (
    # --- reused basic-5 model kinds, now on full-scrape features ---
    GiantSpec("logistic_full_v1", "logistic", origin="basic5", notes="basic5 logistic_v1_baseline"),
    GiantSpec("logistic_engineered_full_v1", "logistic", use_engineered=True, origin="basic5", notes="basic5 logistic_snapshot_v2"),
    GiantSpec("sgd_full_v1", "sgd", origin="basic5", notes="basic5 sgd_text_numeric_v1"),
    GiantSpec("linear_svm_calibrated_full_v1", "linear_svm_calibrated", origin="basic5", notes="basic5 linear_svm_calibrated_v1"),
    GiantSpec("extra_trees_full_v1", "extra_trees", use_text=False, origin="basic5", notes="basic5 numeric_tree_v1"),
    GiantSpec("rules_price_v1", "rules", use_text=False, use_categorical=False, origin="basic5", notes="basic5 rules_price_v1 (price-only baseline)"),
    GiantSpec("random_forest_full_v1", "random_forest", origin="basic5", notes="basic5 random_forest_basic_v1"),
    GiantSpec("hist_gradient_full_v1", "hist_gradient_boosting", use_text=False, origin="basic5", notes="basic5 hist_gradient_basic_numeric_v1"),
    GiantSpec("xgboost_full_v1", "xgboost", origin="basic5", notes="basic5 xgboost_basic_v1"),
    # --- new models that exploit the richer feature set ---
    GiantSpec("lightgbm_full_v1", "lightgbm", origin="new", notes="LightGBM on numeric+onehot+text"),
    GiantSpec("catboost_full_v1", "catboost", use_text=False, origin="new", notes="CatBoost (dense numeric+onehot)"),
    GiantSpec("extra_trees_tuned_v1", "extra_trees_tuned", use_text=False, origin="new", notes="deeper ExtraTrees"),
    GiantSpec("hist_gradient_tuned_v1", "hist_gradient_tuned", use_text=False, origin="new", notes="tuned HistGB + class weights + early stop"),
    GiantSpec("stacking_full_v1", "stacking", origin="new", notes="logistic meta over RF/XGB/LightGBM"),
)


def zoo_by_name() -> dict[str, GiantSpec]:
    return {spec.name: spec for spec in ZOO}


def to_str_frame(values):
    """2-D-safe string coercion for the categorical block (module-level so it pickles).

    ``base_sweep.text_values_to_str`` collapses a multi-column frame to its first
    column, which breaks OneHotEncoder; this preserves all categorical columns.
    """
    frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
    return frame.fillna("unknown").astype(str)


def _has_text_signal(frame: pd.DataFrame, column: str) -> bool:
    return (
        column in frame.columns
        and frame[column].fillna("").astype(str).str.contains(base_sweep.TOKEN_RE, regex=True).any()
    )


def _has_numeric_signal(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any()


def _has_categorical_signal(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    values = frame[column].astype(str).str.strip().str.lower()
    return values.replace({"nan": "", "none": ""}).str.len().gt(0).any()


def select_full_features(frame: pd.DataFrame, spec: GiantSpec) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric, categorical, text) columns usable for ``spec`` on ``frame``."""
    if spec.kind == "rules":
        numeric = [c for c in ("price_num", "likes_num", "page_num", "Price", "Likes", "Page") if c in frame.columns]
        return numeric, [], []

    numeric: list[str] = []
    if spec.use_numeric:
        pool = list(FULL_NUMERIC)
        if spec.use_engineered:
            pool += [c for c in ENGINEERED_EXTRA if c not in pool]
        numeric = [c for c in pool if _has_numeric_signal(frame, c)]

    categorical: list[str] = []
    if spec.use_categorical:
        categorical = [c for c in FULL_CATEGORICAL if _has_categorical_signal(frame, c)]

    text: list[str] = []
    if spec.use_text and spec.kind not in DENSE_KINDS:
        text = [c for c in FULL_TEXT if _has_text_signal(frame, c)]

    # Leakage hygiene, matching basic5 / full_scrape_model conventions: drop
    # identity/raw-source/blocked-word columns from numeric, text and categorical.
    numeric = base_sweep.safe_selected_features(numeric, frame)
    text = base_sweep.safe_selected_features(text, frame)
    categorical = [c for c in categorical if not base_sweep.is_leakage_feature(c)]

    return numeric, categorical, text


def build_full_preprocessor(
    numeric: list[str],
    categorical: list[str],
    text: list[str],
    *,
    scale_numeric: bool,
    dense: bool,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    transformers: list[tuple[str, Any, Any]] = []
    if numeric:
        steps = [
            ("to_numeric", FunctionTransformer(base_sweep.to_numeric_frame, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            steps.append(("scale", StandardScaler(with_mean=False)))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("to_str", FunctionTransformer(to_str_frame, validate=False)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=not dense)),
                    ]
                ),
                categorical,
            )
        )
    for col in text:
        transformers.append(
            (
                f"text_{col}",
                Pipeline(
                    [
                        ("to_text", FunctionTransformer(base_sweep.text_values_to_str, validate=False)),
                        ("tfidf", TfidfVectorizer(max_features=400, min_df=3, ngram_range=(1, 2), strip_accents="unicode")),
                    ]
                ),
                col,
            )
        )
    sparse_threshold = 0.0 if dense else 0.3
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=sparse_threshold)


def _lightgbm_classifier():
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - exercised only without lightgbm
        raise RuntimeError("lightgbm_full_v1 requires the optional `lightgbm` package.") from exc
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def _catboost_classifier():
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover - exercised only without catboost
        raise RuntimeError("catboost_full_v1 requires the optional `catboost` package.") from exc
    return CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )


def _xgboost_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("xgboost_full_v1 requires the optional `xgboost` package.") from exc
    return XGBClassifier(
        n_estimators=320,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.85,
        min_child_weight=2.0,
        reg_lambda=2.0,
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )


def make_model(spec: GiantSpec, numeric: list[str], categorical: list[str], text: list[str]):
    """Build an unfitted sklearn-compatible estimator for ``spec``."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        StackingClassifier,
    )
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    if spec.kind == "rules":
        return base_sweep.RulePriceScorer()

    dense = spec.kind in DENSE_KINDS
    scale_numeric = spec.kind in LINEAR_KINDS
    pre = build_full_preprocessor(numeric, categorical, text, scale_numeric=scale_numeric, dense=dense)

    if spec.kind == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    elif spec.kind == "sgd":
        clf = SGDClassifier(loss="log_loss", max_iter=1500, tol=1e-3, class_weight="balanced", random_state=SEED)
    elif spec.kind == "linear_svm_calibrated":
        svc = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=3000, tol=1e-3)
        try:
            clf = CalibratedClassifierCV(estimator=svc, cv=3, method="sigmoid")
        except TypeError:  # pragma: no cover - older sklearn
            clf = CalibratedClassifierCV(base_estimator=svc, cv=3, method="sigmoid")
    elif spec.kind == "extra_trees":
        clf = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=4, class_weight="balanced", random_state=SEED, n_jobs=-1)
    elif spec.kind == "extra_trees_tuned":
        clf = ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
        )
    elif spec.kind == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=240, max_depth=24, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
        )
    elif spec.kind == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.06, max_leaf_nodes=15,
            min_samples_leaf=12, l2_regularization=0.1, random_state=SEED,
        )
    elif spec.kind == "hist_gradient_tuned":
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=0.2, early_stopping=True, validation_fraction=0.1,
            class_weight="balanced", random_state=SEED,
        )
    elif spec.kind == "xgboost":
        clf = _xgboost_classifier()
    elif spec.kind == "lightgbm":
        clf = _lightgbm_classifier()
    elif spec.kind == "catboost":
        clf = _catboost_classifier()
    elif spec.kind == "stacking":
        estimators = [
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=3, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
            ("xgb", _xgboost_classifier()),
        ]
        try:
            estimators.append(("lgbm", _lightgbm_classifier()))
        except RuntimeError:
            pass
        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
            cv=3,
            n_jobs=-1,
            passthrough=False,
        )
    else:
        raise ValueError(f"Unknown model kind: {spec.kind}")

    return Pipeline([("features", pre), ("model", clf)])
