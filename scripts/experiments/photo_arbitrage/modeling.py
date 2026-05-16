from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.photo_arbitrage.features import MODEL_FEATURES, add_photo_features, heuristic_bad_photo_probability
from experiments.photo_arbitrage.paths import MODELS_DIR, assert_photo_path, utc_now_iso, write_json
from experiments.photo_arbitrage.quality_methods import MethodConfig, add_quality_method_scores, model_feature_columns, normalize_methods


VALID_LABELS = {"photo_quality_bad", "photo_quality_good", "unclear", "not_item_photo"}
TRAINING_LABELS = {"photo_quality_bad": 1, "photo_quality_good": 0}
MODEL_VERSION = "photo_quality_v1"


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in VALID_LABELS else ""


def resolve_feature_columns(method_config: MethodConfig | None = None) -> list[str]:
    columns = list(MODEL_FEATURES)
    if method_config is not None:
        for column in model_feature_columns(method_config.methods):
            if column not in columns:
                columns.append(column)
    return columns


def ensure_model_features(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    method_config: MethodConfig | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    if not set(MODEL_FEATURES).issubset(out.columns):
        out = add_photo_features(out)
    if method_config is not None:
        extra_columns = [column for column in feature_columns if column not in MODEL_FEATURES]
        missing_extra = [column for column in extra_columns if column not in out.columns]
        if missing_extra:
            out = add_quality_method_scores(out, config=method_config)
    for column in feature_columns:
        if column not in out.columns:
            out[column] = np.nan
    return out


def prepare_labeled_frame(labels: pd.DataFrame, *, method_config: MethodConfig | None = None) -> pd.DataFrame:
    out = labels.copy()
    if "manual_label" not in out.columns:
        out["manual_label"] = ""
    out["manual_label"] = out["manual_label"].map(normalize_label)
    out = out[out["manual_label"].isin(TRAINING_LABELS)].copy()
    out["PhotoQualityTarget"] = out["manual_label"].map(TRAINING_LABELS).astype(int)
    feature_columns = resolve_feature_columns(method_config)
    out = ensure_model_features(out, feature_columns=feature_columns, method_config=method_config)
    return out.reset_index(drop=True)


def make_photo_quality_pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )


class CentroidPhotoQualityModel:
    """Small fallback classifier used when scikit-learn is unavailable."""

    def __init__(self, features: list[str]):
        self.features = features
        self.fill_values: np.ndarray | None = None
        self.bad_center: np.ndarray | None = None
        self.good_center: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "CentroidPhotoQualityModel":
        values = x.apply(pd.to_numeric, errors="coerce")
        fill = values.median(numeric_only=True).fillna(0.0)
        arr = values.fillna(fill).to_numpy(dtype=float)
        target = y.to_numpy(dtype=int)
        self.fill_values = fill.to_numpy(dtype=float)
        self.bad_center = arr[target == 1].mean(axis=0)
        self.good_center = arr[target == 0].mean(axis=0)
        return self

    def _matrix(self, x: pd.DataFrame) -> np.ndarray:
        values = x[self.features].apply(pd.to_numeric, errors="coerce")
        fill = self.fill_values if self.fill_values is not None else np.zeros(len(self.features))
        return values.fillna(pd.Series(fill, index=self.features)).to_numpy(dtype=float)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        arr = self._matrix(x)
        bad = self.bad_center if self.bad_center is not None else np.zeros(arr.shape[1])
        good = self.good_center if self.good_center is not None else np.zeros(arr.shape[1])
        dist_bad = np.linalg.norm(arr - bad, axis=1)
        dist_good = np.linalg.norm(arr - good, axis=1)
        prob_bad = dist_good / np.maximum(dist_bad + dist_good, 1e-9)
        return np.column_stack([1.0 - prob_bad, prob_bad])
def train_photo_quality_model(labels: pd.DataFrame, *, method_config: MethodConfig | None = None) -> tuple[Any | None, dict[str, Any]]:
    normalized_method_config = MethodConfig(methods=normalize_methods(method_config.methods)) if method_config is not None else None
    feature_columns = resolve_feature_columns(normalized_method_config)
    frame = prepare_labeled_frame(labels, method_config=normalized_method_config)
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at": utc_now_iso(),
        "features": feature_columns,
        "target": "photo_quality_bad_vs_good",
        "label_counts": labels.get("manual_label", pd.Series(dtype=str)).map(normalize_label).value_counts(dropna=False).to_dict(),
        "training_rows": int(len(frame)),
        "status": "not_trained",
    }
    if normalized_method_config is not None:
        metadata["quality_method_config"] = {
            "methods": list(normalized_method_config.methods),
            "pyiqa_model": normalized_method_config.pyiqa_model,
            "aesthetic_model": normalized_method_config.aesthetic_model,
            "dino_model": normalized_method_config.dino_model,
            "max_images_per_item": int(normalized_method_config.max_images_per_item),
            "device": normalized_method_config.device,
        }
    if len(frame) < 4 or frame["PhotoQualityTarget"].nunique() < 2:
        metadata["reason"] = "Need at least two usable labels in each class or a minimum of four labeled rows."
        return None, metadata
    x = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = frame["PhotoQualityTarget"].astype(int)
    if sklearn_available():
        model = make_photo_quality_pipeline()
        metadata["model_kind"] = "logistic_regression"
    else:
        model = CentroidPhotoQualityModel(feature_columns)
        metadata["model_kind"] = "centroid_fallback"
    model.fit(x, y)
    train_scores = model.predict_proba(x)[:, 1]
    metadata.update(
        {
            "status": "trained",
            "positive_rows": int(y.sum()),
            "negative_rows": int((1 - y).sum()),
            "training_mean_bad_probability": float(np.mean(train_scores)),
        }
    )
    return model, metadata


def save_model(model: Any, metadata: dict[str, Any], *, model_dir: Path = MODELS_DIR) -> tuple[Path | None, Path]:
    model_dir = assert_photo_path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now('UTC').strftime("%Y%m%d_%H%M%S")
    metadata_path = model_dir / f"{MODEL_VERSION}_{timestamp}_metadata.json"
    model_path = model_dir / f"{MODEL_VERSION}_{timestamp}.pkl"
    if model is not None:
        tmp = model_path.with_suffix(model_path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(model, handle)
        tmp.replace(model_path)
        metadata["artifact_path"] = str(model_path)
        latest_model = model_dir / f"{MODEL_VERSION}_latest.pkl"
        latest_tmp = latest_model.with_suffix(latest_model.suffix + ".tmp")
        with latest_tmp.open("wb") as handle:
            pickle.dump(model, handle)
        latest_tmp.replace(latest_model)
    else:
        model_path = None
        metadata["artifact_path"] = ""
    write_json(metadata_path, metadata)
    write_json(model_dir / f"{MODEL_VERSION}_latest_metadata.json", metadata)
    return model_path, metadata_path


def load_latest_model(model_dir: Path = MODELS_DIR) -> tuple[Any | None, dict[str, Any]]:
    metadata_path = model_dir / f"{MODEL_VERSION}_latest_metadata.json"
    if not metadata_path.exists():
        return None, {"status": "missing", "model_version": "heuristic_v0"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact = metadata.get("artifact_path") or str(model_dir / f"{MODEL_VERSION}_latest.pkl")
    path = Path(artifact)
    if not path.exists():
        latest = model_dir / f"{MODEL_VERSION}_latest.pkl"
        path = latest if latest.exists() else path
    if not path.exists():
        return None, metadata
    with path.open("rb") as handle:
        model = pickle.load(handle)
    return model, metadata


def method_config_from_metadata(metadata: dict[str, Any] | None) -> MethodConfig | None:
    payload = dict(metadata.get("quality_method_config") or {}) if metadata else {}
    methods = payload.get("methods")
    if not methods:
        return None
    return MethodConfig(
        methods=normalize_methods(methods),
        pyiqa_model=str(payload.get("pyiqa_model", MethodConfig.pyiqa_model)),
        aesthetic_model=str(payload.get("aesthetic_model", MethodConfig.aesthetic_model)),
        dino_model=str(payload.get("dino_model", MethodConfig.dino_model)),
        max_images_per_item=int(payload.get("max_images_per_item", 1)),
        device=str(payload.get("device", "auto")),
    )


def score_bad_photo_probability(
    frame: pd.DataFrame,
    model: Any | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.Series, str]:
    method_config = method_config_from_metadata(metadata)
    feature_columns = list(metadata.get("features") or MODEL_FEATURES) if metadata else list(MODEL_FEATURES)
    work = ensure_model_features(frame, feature_columns=feature_columns, method_config=method_config)
    if model is None:
        return heuristic_bad_photo_probability(work), "heuristic_v0"
    x = work[feature_columns].apply(pd.to_numeric, errors="coerce")
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)[:, 1]
    else:
        raw = model.decision_function(x)
        scores = 1.0 / (1.0 + np.exp(-raw))
    return pd.Series(np.clip(scores, 0.0, 1.0), index=frame.index), MODEL_VERSION
