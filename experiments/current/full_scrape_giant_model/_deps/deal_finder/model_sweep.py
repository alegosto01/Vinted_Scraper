#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TOKEN_RE = r"[A-Za-z0-9]{2,}"
TITLE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.full_scrape_giant_model._deps.deal_finder.dataset import build_datasets, safe_feature_columns
from experiments.current.full_scrape_giant_model._deps.deal_finder.modeling import (
    TEXT_CANDIDATES,
    TARGET_COL,
    ELIGIBLE_COL,
    SplitFrames,
    choose_threshold,
    precision_at_k,
    score_with_model,
    threshold_metrics,
    to_numeric_frame,
    text_values_to_str,
)
from experiments.current.full_scrape_giant_model._deps.deal_finder.paths import (
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)


PROMOTION_PRECISION = 0.80
VALIDATION_MIN_COUNT = 20
TEST_MIN_COUNT = 10
DEFAULT_SEED = 42
ROBUSTNESS_SEEDS = (42, 123, 2026)
MAX_CALIBRATED_SVM_FIT_ROWS = 5000
MAX_VISUAL_SWEEP_ROWS = 3000
BASE_NUMERIC = ["Price", "Likes", "Page"]
ENGINEERED_NUMERIC = [
    "price_num",
    "likes_num",
    "page_num",
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
    "title_has_digit",
    "title_has_new_word",
    "title_has_auth_word",
    "brand_present",
    "size_num",
]
IMAGE_FEATURES = [
    "image_width",
    "image_height",
    "image_aspect_ratio",
    "image_brightness",
    "image_contrast",
    "image_saturation",
    "image_sharpness",
    "image_edge_density",
]
BLOCKED_FEATURE_WORDS = (
    "status",
    "sold",
    "sale",
    "outcome",
    "priorityqueue",
    "recheck",
    "detected",
    "known_active",
)
IDENTITY_OR_RAW_SOURCE_COLUMNS = {"Dataid", "Link", "Images", "LocalImagePaths", "LocalPrimaryImagePath", "item_id"}


@dataclass(frozen=True)
class ApproachSpec:
    name: str
    kind: str
    feature_policy: str
    use_text: bool = True
    use_engineered: bool = True
    use_image: bool = False
    model_family: str = "linear"
    live_ready: bool = True


APPROACHES = [
    ApproachSpec("logistic_v1_baseline", "logistic", "snapshot_raw_v1", use_engineered=False),
    ApproachSpec("logistic_snapshot_v2", "logistic", "snapshot_engineered_v2"),
    ApproachSpec("sgd_text_numeric_v1", "sgd", "snapshot_engineered_v2"),
    ApproachSpec("linear_svm_calibrated_v1", "linear_svm_calibrated", "snapshot_engineered_v2"),
    ApproachSpec("numeric_tree_v1", "extra_trees", "snapshot_engineered_numeric_v1", use_text=False),
    ApproachSpec("rules_price_v1", "rules", "snapshot_rules_price_v1", use_text=False),
    ApproachSpec("visual_basic_v1", "logistic", "snapshot_visual_basic_v1", use_image=True),
]


class RulePriceScorer:
    def __init__(self) -> None:
        self.price_q10 = 0.0
        self.price_q50 = 1.0
        self.likes_q90 = 1.0

    def fit(self, frame: pd.DataFrame, y=None):
        price = pd.to_numeric(frame.get("price_num"), errors="coerce")
        likes = pd.to_numeric(frame.get("likes_num"), errors="coerce")
        self.price_q10 = float(price.quantile(0.10)) if price.notna().any() else 0.0
        self.price_q50 = float(price.quantile(0.50)) if price.notna().any() else 1.0
        self.likes_q90 = float(likes.quantile(0.90)) if likes.notna().any() else 1.0
        if self.price_q50 <= self.price_q10:
            self.price_q50 = self.price_q10 + 1.0
        if self.likes_q90 <= 0:
            self.likes_q90 = 1.0
        return self

    def _score(self, frame: pd.DataFrame) -> np.ndarray:
        price = pd.to_numeric(frame.get("price_num"), errors="coerce").fillna(self.price_q50)
        likes = pd.to_numeric(frame.get("likes_num"), errors="coerce").fillna(0.0)
        page = pd.to_numeric(frame.get("page_num"), errors="coerce").fillna(10.0)
        cheapness = 1.0 - ((price - self.price_q10) / max(self.price_q50 - self.price_q10, 1e-6))
        cheapness = cheapness.clip(0.0, 1.0)
        demand = (likes / max(self.likes_q90, 1e-6)).clip(0.0, 1.0)
        first_page = (1.0 - ((page - 1.0) / 9.0)).clip(0.0, 1.0)
        score = 0.62 * cheapness + 0.23 * demand + 0.15 * first_page
        return score.to_numpy(dtype=float)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        score = np.clip(self._score(frame), 0.0, 1.0)
        return np.column_stack([1.0 - score, score])


def parse_bool_series(series: pd.Series) -> pd.Series:
    if str(series.dtype).lower() in {"bool", "boolean"}:
        return series.fillna(False).astype(bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def prepare_sweep_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if ELIGIBLE_COL not in out.columns or TARGET_COL not in out.columns:
        return out.iloc[0:0].copy()
    out = out[parse_bool_series(out[ELIGIBLE_COL])].copy()
    if "item_id" in out.columns:
        out = out.drop_duplicates("item_id", keep="first")
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce").fillna(0).astype(int)
    return out.reset_index(drop=True)


def stratified_random_split(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    seed: int = DEFAULT_SEED,
    train_frac: float = 0.60,
    validation_frac: float = 0.20,
) -> SplitFrames:
    if df.empty:
        return SplitFrames(df.copy(), df.copy(), df.copy())
    try:
        from sklearn.model_selection import train_test_split
    except Exception:
        rng = np.random.default_rng(seed)
        train_parts = []
        validation_parts = []
        test_parts = []
        for _, part in df.groupby(target_col, sort=False):
            shuffled = part.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
            n = len(shuffled)
            train_end = int(np.floor(n * train_frac))
            validation_end = train_end + int(np.floor(n * validation_frac))
            train_parts.append(shuffled.iloc[:train_end])
            validation_parts.append(shuffled.iloc[train_end:validation_end])
            test_parts.append(shuffled.iloc[validation_end:])
        return SplitFrames(
            train=pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True),
            validation=pd.concat(validation_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True),
            test=pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True),
        )
    y = df[target_col].astype(int)
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 3 else None
    train, rest = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    rest_y = rest[target_col].astype(int)
    rest_stratify = rest_y if rest_y.nunique() > 1 and rest_y.value_counts().min() >= 2 else None
    validation_size_of_rest = validation_frac / max(1.0 - train_frac, 1e-6)
    validation, test = train_test_split(
        rest,
        train_size=validation_size_of_rest,
        random_state=seed,
        shuffle=True,
        stratify=rest_stratify,
    )
    return SplitFrames(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def coerce_numeric(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    text = series.astype(str).str.replace(",", ".", regex=False).str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(text, errors="coerce")


def normalize_title_text(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str)
    text = text.str.replace(URL_RE, " ", regex=True)
    text = text.str.lower()
    text = text.str.replace(r"[^a-z0-9]+", " ", regex=True)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def token_lists(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.findall(TITLE_TOKEN_RE)


def unique_token_count(tokens: list[str]) -> int:
    return len(set(tokens))


def duplicate_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def average_token_len(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return float(sum(len(token) for token in tokens) / len(tokens))


def uppercase_ratio(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str)
    uppercase = text.str.count(r"[A-Z]")
    letters = text.str.count(r"[A-Za-z]")
    return (uppercase / letters.replace(0, np.nan)).fillna(0.0)


def add_engineered_snapshot_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price_num"] = coerce_numeric(out["Price"]) if "Price" in out.columns else np.nan
    out["likes_num"] = coerce_numeric(out["Likes"]) if "Likes" in out.columns else np.nan
    out["page_num"] = coerce_numeric(out["Page"]) if "Page" in out.columns else np.nan
    out["log_price"] = np.log1p(out["price_num"].clip(lower=0))
    out["log_likes"] = np.log1p(out["likes_num"].clip(lower=0))
    title = out["Title"].fillna("").astype(str) if "Title" in out.columns else pd.Series([""] * len(out), index=out.index)
    brand = out["Brand"].fillna("").astype(str) if "Brand" in out.columns else pd.Series([""] * len(out), index=out.index)
    size = out["Size"].fillna("").astype(str) if "Size" in out.columns else pd.Series([""] * len(out), index=out.index)
    lowered = title.str.lower()
    normalized_title = normalize_title_text(title)
    normalized_tokens = token_lists(normalized_title)
    out["title_char_len"] = title.str.len()
    out["title_token_count"] = token_lists(title).map(len)
    out["title_norm_char_len"] = normalized_title.str.len()
    out["title_norm_token_count"] = normalized_tokens.map(len)
    out["title_unique_token_count"] = normalized_tokens.map(unique_token_count)
    out["title_duplicate_token_ratio"] = normalized_tokens.map(duplicate_token_ratio)
    out["title_avg_token_len"] = normalized_tokens.map(average_token_len)
    out["title_punct_count"] = title.str.count(r"[^\w\s]")
    out["title_upper_ratio"] = uppercase_ratio(title)
    out["title_has_digit"] = title.str.contains(r"\d", regex=True, na=False).astype(int)
    out["title_has_new_word"] = lowered.str.contains(r"\b(?:new|nuov[aoei]?|nueva|neuf|neu|nwt|cartellino)\b", regex=True, na=False).astype(int)
    out["title_has_auth_word"] = lowered.str.contains(r"\b(?:original|authentic|autentico|vero|genuine)\b", regex=True, na=False).astype(int)
    out["brand_present"] = brand.str.strip().ne("").astype(int)
    out["size_num"] = coerce_numeric(size)
    return out


def normalize_image_sources(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [text]


def primary_local_image_path(row: pd.Series) -> str:
    for column in ("LocalPrimaryImagePath", "LocalImagePaths"):
        if column not in row.index:
            continue
        for candidate in normalize_image_sources(row.get(column)):
            path = Path(candidate)
            if path.exists():
                return str(path)
    return ""


def compute_basic_image_features(path: str | Path) -> dict[str, float]:
    from PIL import Image, ImageStat

    image = Image.open(path).convert("RGB")
    width, height = image.size
    small = image.resize((min(width, 256), min(height, 256)))
    gray = small.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    stat = ImageStat.Stat(small)
    brightness = float(np.mean(stat.mean) / 255.0)
    contrast = float(np.std(arr) / 255.0)
    saturation = float((rgb.max(axis=2) - rgb.min(axis=2)).mean())
    gy, gx = np.gradient(arr)
    sharpness = float(np.var(gx) + np.var(gy))
    edge_density = float((np.hypot(gx, gy) > 20.0).mean())
    return {
        "image_width": float(width),
        "image_height": float(height),
        "image_aspect_ratio": float(width / height) if height else np.nan,
        "image_brightness": brightness,
        "image_contrast": contrast,
        "image_saturation": saturation,
        "image_sharpness": sharpness,
        "image_edge_density": edge_density,
    }


def add_image_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rows = []
    cache: dict[str, dict[str, float]] = {}
    empty = {name: np.nan for name in IMAGE_FEATURES}
    for _, row in out.iterrows():
        path = primary_local_image_path(row)
        if not path:
            rows.append(empty.copy())
            continue
        if path not in cache:
            try:
                cache[path] = compute_basic_image_features(path)
            except Exception:
                cache[path] = empty.copy()
        rows.append(cache[path])
    image_df = pd.DataFrame(rows, index=out.index)
    for col in IMAGE_FEATURES:
        out[col] = image_df[col] if col in image_df else np.nan
    out["has_basic_image_features"] = out[IMAGE_FEATURES].notna().any(axis=1).astype(int)
    return out


def safe_selected_features(features: list[str], source_df: pd.DataFrame) -> list[str]:
    safe = set(safe_feature_columns(source_df))
    safe.update(ENGINEERED_NUMERIC)
    safe.update(IMAGE_FEATURES)
    safe.add("has_basic_image_features")
    return [feature for feature in features if feature in safe and not is_leakage_feature(feature)]


def is_leakage_feature(feature: str) -> bool:
    lowered = feature.lower()
    if feature in IDENTITY_OR_RAW_SOURCE_COLUMNS:
        return True
    return any(word in lowered for word in BLOCKED_FEATURE_WORDS)


def has_leakage_features(features: list[str]) -> bool:
    return any(is_leakage_feature(feature) for feature in features)


def build_preprocessor(numeric_features: list[str], text_features: list[str], *, scale_numeric: bool = True):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    transformers = []
    if numeric_features:
        steps = [
            ("to_numeric", FunctionTransformer(to_numeric_frame, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            steps.append(("scale", StandardScaler(with_mean=False)))
        transformers.append(("numeric", Pipeline(steps), numeric_features))
    for col in text_features:
        transformers.append(
            (
                f"text_{col}",
                Pipeline(
                    [
                        ("to_text", FunctionTransformer(text_values_to_str, validate=False)),
                        ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=1)),
                    ]
                ),
                col,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)


def make_model(spec: ApproachSpec, numeric_features: list[str], text_features: list[str]):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    if spec.kind == "rules":
        return RulePriceScorer()
    if spec.kind == "extra_trees":
        pre = build_preprocessor(numeric_features, [], scale_numeric=False)
        return Pipeline(
            [
                ("features", pre),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=120,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=DEFAULT_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if spec.kind == "random_forest":
        pre = build_preprocessor(numeric_features, text_features, scale_numeric=False)
        return Pipeline(
            [
                ("features", pre),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=240,
                        max_depth=24,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=DEFAULT_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if spec.kind == "hist_gradient_boosting":
        pre = build_preprocessor(numeric_features, [], scale_numeric=False)
        return Pipeline(
            [
                ("features", pre),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.06,
                        max_leaf_nodes=15,
                        min_samples_leaf=12,
                        l2_regularization=0.1,
                        random_state=DEFAULT_SEED,
                    ),
                ),
            ]
        )
    if spec.kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("The xgboost approach requires the optional `xgboost` package.") from exc
        pre = build_preprocessor(numeric_features, text_features, scale_numeric=False)
        return Pipeline(
            [
                ("features", pre),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=220,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.85,
                        min_child_weight=2.0,
                        reg_lambda=2.0,
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=DEFAULT_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    pre = build_preprocessor(numeric_features, text_features, scale_numeric=True)
    if spec.kind == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    elif spec.kind == "sgd":
        clf = SGDClassifier(loss="log_loss", max_iter=1500, tol=1e-3, class_weight="balanced", random_state=DEFAULT_SEED)
    elif spec.kind == "linear_svm_calibrated":
        svc = LinearSVC(class_weight="balanced", random_state=DEFAULT_SEED, max_iter=3000, tol=1e-3)
        try:
            clf = CalibratedClassifierCV(estimator=svc, cv=3, method="sigmoid")
        except TypeError:
            clf = CalibratedClassifierCV(base_estimator=svc, cv=3, method="sigmoid")
    else:
        raise ValueError(f"Unknown approach kind: {spec.kind}")
    return Pipeline([("features", pre), ("model", clf)])


def select_features(df: pd.DataFrame, spec: ApproachSpec) -> tuple[list[str], list[str]]:
    if spec.use_engineered:
        numeric = [c for c in ENGINEERED_NUMERIC if c in df.columns]
    else:
        numeric = [c for c in BASE_NUMERIC if c in df.columns]
    if spec.use_image:
        numeric += [c for c in IMAGE_FEATURES if c in df.columns and df[c].notna().any()]
        if "has_basic_image_features" in df.columns:
            numeric.append("has_basic_image_features")
    text = [
        c
        for c in TEXT_CANDIDATES
        if spec.use_text and c in df.columns and df[c].fillna("").astype(str).str.contains(TOKEN_RE, regex=True).any()
    ]
    numeric = safe_selected_features(numeric, df)
    text = safe_selected_features(text, df)
    return numeric, text


def auc_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(np.unique(y_true)) < 2:
            return {"roc_auc": np.nan, "pr_auc": np.nan}
        return {
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "pr_auc": float(average_precision_score(y_true, scores)),
        }
    except Exception:
        return {"roc_auc": np.nan, "pr_auc": np.nan}


def evaluate_scores(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = frame[TARGET_COL].astype(int).to_numpy()
    threshold_row = threshold_metrics(y, scores, threshold)
    threshold_row["recall"] = float(threshold_row["positive_count"] / y.sum()) if y.sum() else np.nan
    base_rate = float(y.mean()) if len(y) else np.nan
    lift = threshold_row["precision"] / base_rate if base_rate and pd.notna(threshold_row["precision"]) else np.nan
    return {
        "rows": int(len(frame)),
        "positive_rows": int(y.sum()),
        "base_rate": base_rate,
        "threshold": threshold_row,
        "precision_at": {f"p@{k}": precision_at_k(y, scores, k) for k in (10, 25, 50)},
        "lift_over_base": float(lift) if pd.notna(lift) else np.nan,
        **auc_metrics(y, scores),
    }


def promoted_80(metrics: dict[str, Any], *, live_ready: bool, feature_leakage: bool) -> tuple[bool, list[str]]:
    reasons = []
    val = metrics.get("validation", {})
    test = metrics.get("test", {})
    val_thr = val.get("threshold", {})
    test_thr = test.get("threshold", {})
    val_p10 = val.get("precision_at", {}).get("p@10", {})
    test_p10 = test.get("precision_at", {}).get("p@10", {})
    checks = [
        (val_thr.get("precision", 0.0) >= PROMOTION_PRECISION, "validation threshold precision >= 80%"),
        (test_thr.get("precision", 0.0) >= PROMOTION_PRECISION, "test threshold precision >= 80%"),
        (val_p10.get("precision", 0.0) >= PROMOTION_PRECISION, "validation precision@10 >= 80%"),
        (test_p10.get("precision", 0.0) >= PROMOTION_PRECISION, "test precision@10 >= 80%"),
        (val_thr.get("count", 0) >= VALIDATION_MIN_COUNT, "at least 20 validation recommendations above threshold"),
        (test_thr.get("count", 0) >= TEST_MIN_COUNT, "at least 10 test recommendations above threshold"),
        (live_ready, "approach is live-ready"),
        (not feature_leakage, "no leakage columns are used"),
    ]
    for ok, reason in checks:
        if not ok:
            reasons.append(reason)
    return not reasons, reasons


def save_pickle(path: Path, obj: Any) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle)
    tmp.replace(path)


def bounded_fit_frame(frame: pd.DataFrame, spec: ApproachSpec, seed: int) -> pd.DataFrame:
    if spec.kind != "linear_svm_calibrated" or len(frame) <= MAX_CALIBRATED_SVM_FIT_ROWS:
        return frame
    rng = np.random.default_rng(seed)
    parts = []
    counts = frame[TARGET_COL].astype(int).value_counts()
    total = int(counts.sum())
    sample_sizes: dict[int, int] = {}
    for label, count in counts.items():
        share = count / max(total, 1)
        n = int(round(MAX_CALIBRATED_SVM_FIT_ROWS * share))
        n = max(3, n)
        n = min(int(count), n)
        sample_sizes[int(label)] = n
    surplus = sum(sample_sizes.values()) - MAX_CALIBRATED_SVM_FIT_ROWS
    while surplus > 0:
        largest_label = max(sample_sizes, key=lambda key: sample_sizes[key])
        reducible = max(sample_sizes[largest_label] - 3, 0)
        if reducible <= 0:
            break
        reduction = min(reducible, surplus)
        sample_sizes[largest_label] -= reduction
        surplus -= reduction
    for label, n in sample_sizes.items():
        part = frame[frame[TARGET_COL].astype(int) == int(label)]
        parts.append(part.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_sample_frame(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    rng = np.random.default_rng(seed)
    parts = []
    counts = frame[TARGET_COL].astype(int).value_counts()
    total = int(counts.sum())
    sample_sizes: dict[int, int] = {}
    for label, count in counts.items():
        share = count / max(total, 1)
        n = int(round(max_rows * share))
        n = max(1, n)
        n = min(int(count), n)
        sample_sizes[int(label)] = n
    surplus = sum(sample_sizes.values()) - max_rows
    while surplus > 0:
        largest_label = max(sample_sizes, key=lambda key: sample_sizes[key])
        reducible = max(sample_sizes[largest_label] - 1, 0)
        if reducible <= 0:
            break
        reduction = min(reducible, surplus)
        sample_sizes[largest_label] -= reduction
        surplus -= reduction
    for label, n in sample_sizes.items():
        part = frame[frame[TARGET_COL].astype(int) == int(label)]
        parts.append(part.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def flatten_metrics(row: dict[str, Any]) -> dict[str, Any]:
    validation = row.get("validation", {})
    test = row.get("test", {})
    val_thr = validation.get("threshold", {})
    test_thr = test.get("threshold", {})
    return {
        "search_name": row.get("search_name"),
        "approach": row.get("approach"),
        "status": row.get("status"),
        "seed": row.get("seed"),
        "feature_policy": row.get("feature_policy"),
        "live_ready": row.get("live_ready"),
        "qualified_for_paper_trading": row.get("qualified_for_paper_trading"),
        "sampled_for_visual": row.get("sampled_for_visual"),
        "threshold": row.get("threshold"),
        "train_rows": row.get("split_rows", {}).get("train"),
        "fit_train_rows": row.get("split_rows", {}).get("fit_train"),
        "validation_rows": validation.get("rows"),
        "test_rows": test.get("rows"),
        "validation_base_rate": validation.get("base_rate"),
        "test_base_rate": test.get("base_rate"),
        "validation_precision": val_thr.get("precision"),
        "validation_count": val_thr.get("count"),
        "validation_recall": val_thr.get("recall"),
        "test_precision": test_thr.get("precision"),
        "test_count": test_thr.get("count"),
        "test_recall": test_thr.get("recall"),
        "validation_precision_at_10": validation.get("precision_at", {}).get("p@10", {}).get("precision"),
        "test_precision_at_10": test.get("precision_at", {}).get("p@10", {}).get("precision"),
        "validation_precision_at_25": validation.get("precision_at", {}).get("p@25", {}).get("precision"),
        "test_precision_at_25": test.get("precision_at", {}).get("p@25", {}).get("precision"),
        "validation_precision_at_50": validation.get("precision_at", {}).get("p@50", {}).get("precision"),
        "test_precision_at_50": test.get("precision_at", {}).get("p@50", {}).get("precision"),
        "test_lift_over_base": test.get("lift_over_base"),
        "test_roc_auc": test.get("roc_auc"),
        "test_pr_auc": test.get("pr_auc"),
        "numeric_features": ",".join(row.get("numeric_features", [])),
        "text_features": ",".join(row.get("text_features", [])),
        "fit_seconds": row.get("fit_seconds"),
        "promotion_failures": "; ".join(row.get("promotion_failures", [])),
        "reason": row.get("reason", ""),
    }


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def train_approach(
    frame: pd.DataFrame,
    *,
    search_name: str,
    spec: ApproachSpec,
    seed: int,
    model_prefix: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_frame = frame
    sampled_for_visual = False
    if spec.use_image and len(frame) > MAX_VISUAL_SWEEP_ROWS:
        input_frame = stratified_sample_frame(frame, MAX_VISUAL_SWEEP_ROWS, seed)
        sampled_for_visual = True
    work = add_engineered_snapshot_features(input_frame)
    if spec.use_image:
        work = add_image_features(work)
    splits = stratified_random_split(work, seed=seed)
    if min(len(splits.train), len(splits.validation), len(splits.test)) == 0:
        return {"search_name": search_name, "approach": spec.name, "seed": seed, "status": "skipped", "reason": "empty split"}
    if splits.train[TARGET_COL].nunique() < 2:
        return {"search_name": search_name, "approach": spec.name, "seed": seed, "status": "skipped", "reason": "training split has only one label class"}
    numeric, text = select_features(splits.train, spec)
    if not numeric and not text:
        return {"search_name": search_name, "approach": spec.name, "seed": seed, "status": "skipped", "reason": "no usable features"}
    if spec.kind == "linear_svm_calibrated" and splits.train[TARGET_COL].value_counts().min() < 3:
        return {
            "search_name": search_name,
            "approach": spec.name,
            "seed": seed,
            "status": "skipped",
            "reason": "calibrated SVM needs at least 3 training rows per class",
        }
    feature_leakage = has_leakage_features(numeric + text)
    fit_frame = bounded_fit_frame(splits.train, spec, seed)
    model = make_model(spec, numeric, text)
    model.fit(fit_frame, fit_frame[TARGET_COL].astype(int))
    val_scores = score_with_model(model, splits.validation)
    test_scores = score_with_model(model, splits.test)
    threshold = choose_threshold(
        splits.validation[TARGET_COL].astype(int).to_numpy(),
        val_scores,
        min_precision=PROMOTION_PRECISION,
        min_count=VALIDATION_MIN_COUNT,
    )["threshold"]
    artifact_path = MODELS_DIR / f"{model_prefix}_{search_name}_{spec.name}_seed{seed}.pkl"
    metadata_path = MODELS_DIR / f"{model_prefix}_{search_name}_{spec.name}_seed{seed}_metadata.json"
    save_pickle(artifact_path, model)
    metadata = {
        "search_name": search_name,
        "approach": spec.name,
        "model_kind": spec.kind,
        "artifact_path": str(artifact_path),
        "numeric_features": numeric,
        "text_features": text,
        "feature_policy": spec.feature_policy,
        "threshold": float(threshold),
        "label": TARGET_COL,
        "live_success_label": "sold_within_2d",
        "seed": int(seed),
        "split": "stratified_random_60_20_20",
        "input_rows": int(len(frame)),
        "sweep_rows": int(len(input_frame)),
        "train_rows": int(len(splits.train)),
        "fit_train_rows": int(len(fit_frame)),
        "live_ready": bool(spec.live_ready),
        "requires_images": bool(spec.use_image),
        "sampled_for_visual": bool(sampled_for_visual),
    }
    write_json(metadata_path, to_builtin(metadata))
    metrics = {
        "search_name": search_name,
        "approach": spec.name,
        "status": "trained",
        "seed": int(seed),
        "model_metadata": metadata,
        "feature_policy": spec.feature_policy,
        "live_ready": bool(spec.live_ready),
        "requires_images": bool(spec.use_image),
        "sampled_for_visual": bool(sampled_for_visual),
        "numeric_features": numeric,
        "text_features": text,
        "feature_leakage": bool(feature_leakage),
        "threshold": float(threshold),
        "split_rows": {
            "train": int(len(splits.train)),
            "fit_train": int(len(fit_frame)),
            "validation": int(len(splits.validation)),
            "test": int(len(splits.test)),
        },
        "validation": evaluate_scores(splits.validation, val_scores, float(threshold)),
        "test": evaluate_scores(splits.test, test_scores, float(threshold)),
    }
    ok, reasons = promoted_80(metrics, live_ready=spec.live_ready, feature_leakage=feature_leakage)
    metrics["qualified_for_paper_trading"] = bool(ok)
    metrics["promotion_failures"] = reasons
    metrics["fit_seconds"] = float(time.perf_counter() - started)
    return metrics


def best_tables(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trained = metrics_df[metrics_df["status"] == "trained"].copy()
    if trained.empty:
        return pd.DataFrame(), pd.DataFrame()
    sort_cols = ["qualified_for_paper_trading", "test_precision", "test_precision_at_10", "test_count"]
    by_search = trained.sort_values(sort_cols, ascending=[False, False, False, False]).drop_duplicates("search_name", keep="first")
    by_approach = trained.sort_values(sort_cols, ascending=[False, False, False, False]).drop_duplicates("approach", keep="first")
    return by_search.reset_index(drop=True), by_approach.reset_index(drop=True)


def write_sweep_report(run_dir: Path, metrics_df: pd.DataFrame, by_search: pd.DataFrame, candidates: list[dict[str, Any]]) -> Path:
    report_path = assert_experiment_path(run_dir / "sweep_report.md")
    lines = [
        "# Deal Finder Model Sweep",
        "",
        f"Run folder: `{run_dir}`",
        "",
        "Promotion gate: validation/test precision above threshold >= 80%, validation/test precision@10 >= 80%, plus minimum recommendation counts.",
        "",
        f"Models trained: `{int((metrics_df['status'] == 'trained').sum()) if not metrics_df.empty else 0}`",
        f"Promotion candidates: `{len(candidates)}`",
        "",
    ]
    if candidates:
        lines.extend(["## Promotion Candidates", "", "| search | approach | threshold | val precision | test precision | test p@10 | test count |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for row in candidates:
            lines.append(
                f"| {row['search_name']} | {row['approach']} | {row['threshold']:.4f} | "
                f"{row['validation_precision']:.3f} | {row['test_precision']:.3f} | "
                f"{row['test_precision_at_10']:.3f} | {int(row['test_count'])} |"
            )
    else:
        lines.extend(["## Promotion Candidates", "", "No approach reached the 80% promotion gate in this sweep."])
    if not by_search.empty:
        lines.extend(["", "## Best Row Per Search", "", "| search | approach | qualified | test precision | test p@10 | reason |", "| --- | --- | --- | ---: | ---: | --- |"])
        for _, row in by_search.iterrows():
            lines.append(
                f"| {row['search_name']} | {row['approach']} | {bool(row['qualified_for_paper_trading'])} | "
                f"{row['test_precision']:.3f} | {row['test_precision_at_10']:.3f} | {row.get('promotion_failures', '')} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The sweep is offline-only and did not start or stop any paper-trading timer.",
            "- Image approaches use derived numeric image features, not raw image paths or URLs.",
            "- Pipeline/deal-score columns are intentionally excluded from promotion-eligible features.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_sweep(
    *,
    searches: list[str] | None = None,
    limit_rows: int | None = None,
    out_dir: Path | None = None,
    approach_names: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    if out_dir is None:
        out_dir = OFFLINE_RUNS_DIR / run_id("sweep")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [DEFAULT_SEED]
    selected_names = set(approach_names or [spec.name for spec in APPROACHES])
    selected_approaches = [spec for spec in APPROACHES if spec.name in selected_names]
    dataset_results = build_datasets(searches=searches, out_dir=out_dir, limit_rows=limit_rows)

    metrics: list[dict[str, Any]] = []
    for dataset_result in dataset_results:
        df = pd.read_csv(dataset_result.path, low_memory=False)
        frame = prepare_sweep_frame(df)
        print(
            f"[model_sweep] search={dataset_result.search_name} eligible_rows={len(frame)}",
            flush=True,
        )
        if len(frame) < 50 or frame[TARGET_COL].nunique() < 2:
            for spec in selected_approaches:
                metrics.append(
                    {
                        "search_name": dataset_result.search_name,
                        "approach": spec.name,
                        "status": "skipped",
                        "seed": seeds[0],
                        "reason": "not enough eligible rows or only one label class",
                    }
                )
            continue
        for seed in seeds:
            for spec in selected_approaches:
                step_started = time.perf_counter()
                print(
                    f"[model_sweep] train search={dataset_result.search_name} approach={spec.name} seed={seed}",
                    flush=True,
                )
                try:
                    row = train_approach(
                        frame,
                        search_name=dataset_result.search_name,
                        spec=spec,
                        seed=seed,
                        model_prefix=out_dir.name,
                    )
                    metrics.append(row)
                    print(
                        f"[model_sweep] done search={dataset_result.search_name} approach={spec.name} "
                        f"status={row.get('status')} seconds={time.perf_counter() - step_started:.1f}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[model_sweep] error search={dataset_result.search_name} approach={spec.name} "
                        f"seconds={time.perf_counter() - step_started:.1f}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    metrics.append(
                        {
                            "search_name": dataset_result.search_name,
                            "approach": spec.name,
                            "status": "error",
                            "seed": seed,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )

    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, to_builtin({"metrics": metrics}))
    metrics_df = pd.DataFrame([flatten_metrics(row) for row in metrics])
    metrics_long_path = assert_experiment_path(out_dir / "metrics_long.csv")
    metrics_df.to_csv(metrics_long_path, index=False)
    by_search, by_approach = best_tables(metrics_df)
    by_search_path = assert_experiment_path(out_dir / "best_by_search.csv")
    by_approach_path = assert_experiment_path(out_dir / "best_by_approach.csv")
    by_search.to_csv(by_search_path, index=False)
    by_approach.to_csv(by_approach_path, index=False)
    candidates_df = metrics_df[metrics_df.get("qualified_for_paper_trading", False) == True].copy() if not metrics_df.empty else pd.DataFrame()
    candidate_records = candidates_df.replace({np.nan: None}).to_dict("records") if not candidates_df.empty else []
    write_json(out_dir / "promotion_candidates.json", to_builtin({"promotion_candidates": candidate_records}))
    report_path = write_sweep_report(out_dir, metrics_df, by_search, candidate_records)
    summary = {
        "run_dir": str(out_dir),
        "dataset_count": len(dataset_results),
        "approaches": [spec.name for spec in selected_approaches],
        "seeds": seeds,
        "model_rows": len(metrics),
        "trained_rows": int((metrics_df["status"] == "trained").sum()) if not metrics_df.empty else 0,
        "promotion_candidate_count": len(candidate_records),
        "outputs": {
            "metrics_long": str(metrics_long_path),
            "best_by_search": str(by_search_path),
            "best_by_approach": str(by_approach_path),
            "promotion_candidates": str(out_dir / "promotion_candidates.json"),
            "sweep_report": str(report_path),
        },
        "auto_paper_trading": False,
    }
    manifest_path = out_dir / "manifest.json"
    write_manifest(manifest_path, command="model_sweep", extra=to_builtin(summary))

    # ── Experiment tracker ──────────────────────────────────────────────────
    try:
        from experiments.current.full_scrape_giant_model._deps.tracking.db import init_db, log_run, log_metrics
        init_db()
        manifest = read_json(manifest_path, {})
        git_info = manifest.get("git", {})
        log_run({
            "run_id": out_dir.name,
            "family": "deal_finder",
            "command": "model_sweep",
            "status": "completed",
            "created_at": manifest.get("created_at"),
            "finished_at": manifest.get("created_at"),
            "run_dir": str(out_dir),
            "git_branch": git_info.get("branch"),
            "git_head": git_info.get("head"),
            "git_dirty": bool(git_info.get("status_short")),
            "promotion_count": len(candidate_records),
        })
        log_metrics(out_dir.name, metrics_df)
    except Exception as exc:
        print(f"[tracking] warning: failed to log run: {exc}", flush=True)
    # ─────────────────────────────────────────────────────────────────────────

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline deal-finder model sweeps.")
    parser.add_argument("--all-searches", action="store_true")
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--approach", action="append", default=[])
    parser.add_argument("--robustness", action="store_true", help="Run seeds 42, 123, and 2026 instead of only seed 42.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all_searches and not args.search:
        raise SystemExit("Use --all-searches or at least one --search.")
    summary = run_sweep(
        searches=None if args.all_searches else args.search,
        limit_rows=args.limit_rows,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        approach_names=args.approach or None,
        seeds=list(ROBUSTNESS_SEEDS) if args.robustness else [DEFAULT_SEED],
    )
    print(json.dumps({k: summary[k] for k in ("run_dir", "trained_rows", "promotion_candidate_count")}, indent=2))


if __name__ == "__main__":
    main()
