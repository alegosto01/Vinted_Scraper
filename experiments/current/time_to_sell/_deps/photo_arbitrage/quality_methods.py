from __future__ import annotations

import importlib.util
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

from experiments.current.time_to_sell._deps.photo_arbitrage.features import heuristic_bad_photo_probability, local_image_paths_from_row, numeric_series
from experiments.current.time_to_sell._deps.photo_arbitrage.paths import MODEL_CACHE_DIR, ensure_experiment_dirs


QUALITY_METHOD_COLUMNS = [
    "SimpleBadPhotoScore",
    "PyiqaQualityScore",
    "PyiqaBadPhotoScore",
    "PyiqaModelName",
    "PyiqaStatus",
    "AestheticGoodScore",
    "AestheticBadPhotoScore",
    "AestheticModelName",
    "AestheticLabel",
    "AestheticStatus",
    "FashionClipGoodScore",
    "FashionClipBadScore",
    "FashionClipBadPhotoScore",
    "FashionClipScoreMargin",
    "FashionClipModelName",
    "FashionClipPromptSet",
    "FashionClipPromptCount",
    "FashionClipStatus",
    "DinoEmbedding",
    "DinoEmbeddingDim",
    "DinoEmbeddingNorm",
    "DinoOutlierScore",
    "DinoModelName",
    "DinoStatus",
    "CombinedBadPhotoScore",
    "QualityMethodStatus",
]

DEFAULT_METHODS = ("simple", "pyiqa", "aesthetic", "fashionclip", "dino")
DEFAULT_PYIQA_MODEL = "musiq"
DEFAULT_AESTHETIC_MODEL = "cafeai/cafe_aesthetic"
DEFAULT_FASHIONCLIP_MODEL = "patrickjohncyh/fashion-clip"
DEFAULT_DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINO_IMAGE_SIZE = 224
DINO_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGE_STD = (0.229, 0.224, 0.225)
FASHIONCLIP_PROMPT_SET = "marketplace_photo_quality_v1"
FASHIONCLIP_GOOD_PROMPTS = (
    "a clear well lit product photo of clothing for an online marketplace listing",
    "a sharp clean product photo where the item is easy to see",
    "a professional resale listing photo with good lighting and clear item detail",
)
FASHIONCLIP_BAD_PROMPTS = (
    "a blurry dark low quality marketplace listing photo",
    "a bad product photo where the item is hard to see",
    "a cluttered poorly cropped clothing photo with bad lighting",
)

QUALITY_METHOD_FEATURES = {
    "pyiqa": ["PyiqaBadPhotoScore"],
    "aesthetic": ["AestheticBadPhotoScore"],
    "fashionclip": ["FashionClipBadPhotoScore", "FashionClipScoreMargin"],
    "dino": ["DinoEmbeddingNorm", "DinoOutlierScore"],
}


@dataclass
class MethodConfig:
    methods: tuple[str, ...] = DEFAULT_METHODS
    pyiqa_model: str = DEFAULT_PYIQA_MODEL
    aesthetic_model: str = DEFAULT_AESTHETIC_MODEL
    fashionclip_model: str = DEFAULT_FASHIONCLIP_MODEL
    fashionclip_local_files_only: bool = True
    dino_model: str = DEFAULT_DINO_MODEL
    max_images_per_item: int = 1
    device: str = "auto"


def normalize_methods(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_METHODS
    if isinstance(raw, str):
        parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    else:
        parts = [str(part).strip().lower() for part in raw if str(part).strip()]
    if not parts or "all" in parts:
        return DEFAULT_METHODS
    aliases = {"heuristic": "simple", "clip": "aesthetic", "fclip": "fashionclip", "fashion_clip": "fashionclip", "dinov3": "dino"}
    normalized = []
    for part in parts:
        method = aliases.get(part, part)
        if method not in DEFAULT_METHODS:
            raise ValueError(f"Unknown quality method: {part}")
        if method not in normalized:
            normalized.append(method)
    return tuple(normalized)


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def resolve_device(device: str = "auto") -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def prepare_model_cache() -> None:
    ensure_experiment_dirs()
    cache = MODEL_CACHE_DIR.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    # Do not override HF_HOME: Hugging Face login tokens usually live there.
    # Keep model blobs in the experiment cache while preserving authentication.
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "huggingface" / "transformers"))


class ManualDinoImageProcessor:
    """Small DINO-compatible image preprocessor used if AutoImageProcessor is unavailable."""

    def __init__(
        self,
        *,
        size: int = DINO_IMAGE_SIZE,
        mean: tuple[float, float, float] = DINO_IMAGE_MEAN,
        std: tuple[float, float, float] = DINO_IMAGE_STD,
    ) -> None:
        self.size = int(size)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, *, images: Image.Image, return_tensors: str = "pt") -> dict[str, Any]:
        if return_tensors != "pt":
            raise ValueError("ManualDinoImageProcessor only supports return_tensors='pt'")
        import torch

        image = images.convert("RGB").resize((self.size, self.size), Image.Resampling.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return {"pixel_values": tensor}


def primary_image_path(row: pd.Series | dict) -> str:
    paths = local_image_paths_from_row(row)
    return paths[0] if paths else ""


def selected_image_paths(row: pd.Series | dict, max_images: int = 1) -> list[str]:
    paths = local_image_paths_from_row(row)
    if max_images and max_images > 0:
        return paths[: int(max_images)]
    return paths


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def normalize_quality_to_bad_probability(score: float, *, lower_better: bool | None = None) -> float:
    if not math.isfinite(score):
        return np.nan
    if lower_better:
        return float(np.clip(score / (score + 10.0), 0.0, 1.0))
    if score > 10:
        quality = score / 100.0
    elif score > 1:
        quality = score / 10.0
    else:
        quality = score
    return float(np.clip(1.0 - quality, 0.0, 1.0))


def add_simple_quality_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["SimpleBadPhotoScore"] = heuristic_bad_photo_probability(out)
    return out


def add_pyiqa_scores(frame: pd.DataFrame, *, model_name: str = DEFAULT_PYIQA_MODEL, device: str = "auto", max_images_per_item: int = 1) -> pd.DataFrame:
    out = frame.copy()
    out["PyiqaQualityScore"] = np.nan
    out["PyiqaBadPhotoScore"] = np.nan
    out["PyiqaModelName"] = model_name
    out["PyiqaStatus"] = "not_run"
    has_images = out.apply(lambda row: bool(selected_image_paths(row, max_images_per_item)), axis=1) if len(out) else pd.Series(dtype=bool)
    if len(out) and not bool(has_images.any()):
        out["PyiqaStatus"] = "missing_image"
        return out
    if not package_available("pyiqa"):
        out["PyiqaStatus"] = "pyiqa_not_installed"
        return out

    try:
        metric, lower_better = _load_pyiqa_metric(model_name, device)
    except Exception as exc:
        out["PyiqaStatus"] = f"load_failed:{type(exc).__name__}"
        return out

    for idx, row in out.iterrows():
        paths = selected_image_paths(row, max_images_per_item)
        if not paths:
            out.at[idx, "PyiqaStatus"] = "missing_image"
            continue
        scores = []
        for path in paths:
            try:
                value = metric(str(path))
                if hasattr(value, "detach"):
                    value = value.detach().cpu().flatten()[0].item()
                elif isinstance(value, (list, tuple)):
                    value = value[0]
                scores.append(float(value))
            except Exception:
                continue
        if not scores:
            out.at[idx, "PyiqaStatus"] = "score_failed"
            continue
        quality_score = float(np.mean(scores))
        out.at[idx, "PyiqaQualityScore"] = quality_score
        out.at[idx, "PyiqaBadPhotoScore"] = normalize_quality_to_bad_probability(quality_score, lower_better=lower_better)
        out.at[idx, "PyiqaStatus"] = "ok"
    return out


@lru_cache(maxsize=4)
def _load_pyiqa_metric(model_name: str, device: str):
    prepare_model_cache()
    import pyiqa

    metric = pyiqa.create_metric(model_name, device=resolve_device(device))
    lower_better = getattr(metric, "lower_better", None)
    return metric, lower_better


def _score_aesthetic_result(result: list[dict[str, Any]]) -> tuple[float, float, str]:
    good = np.nan
    label = ""
    for item in result:
        current_label = str(item.get("label", "")).lower()
        score = finite_float(item.get("score"), default=np.nan)
        if not math.isfinite(score):
            continue
        if current_label in {"aesthetic", "good", "beautiful", "high_quality"}:
            good = max(good, score) if math.isfinite(good) else score
            label = current_label
        elif current_label in {"not_aesthetic", "bad", "low_quality", "unaesthetic"}:
            bad = score
            good = 1.0 - bad
            label = current_label
        elif not label:
            label = current_label
    if not math.isfinite(good):
        best = max(result, key=lambda item: finite_float(item.get("score"), default=-1.0), default={})
        good = finite_float(best.get("score"), default=np.nan)
        label = str(best.get("label", label))
    return float(good), float(np.clip(1.0 - good, 0.0, 1.0)) if math.isfinite(good) else np.nan, label


def add_aesthetic_scores(frame: pd.DataFrame, *, model_name: str = DEFAULT_AESTHETIC_MODEL, device: str = "auto") -> pd.DataFrame:
    out = frame.copy()
    out["AestheticGoodScore"] = np.nan
    out["AestheticBadPhotoScore"] = np.nan
    out["AestheticModelName"] = model_name
    out["AestheticLabel"] = ""
    out["AestheticStatus"] = "not_run"
    has_images = out.apply(lambda row: bool(primary_image_path(row)), axis=1) if len(out) else pd.Series(dtype=bool)
    if len(out) and not bool(has_images.any()):
        out["AestheticStatus"] = "missing_image"
        return out
    if not package_available("transformers"):
        out["AestheticStatus"] = "transformers_not_installed"
        return out
    try:
        classifier = _load_aesthetic_classifier(model_name, device)
    except Exception as exc:
        out["AestheticStatus"] = f"load_failed:{type(exc).__name__}"
        return out

    for idx, row in out.iterrows():
        path = primary_image_path(row)
        if not path:
            out.at[idx, "AestheticStatus"] = "missing_image"
            continue
        try:
            result = classifier(path, top_k=None)
            if isinstance(result, dict):
                result = [result]
            good, bad, label = _score_aesthetic_result(result)
        except Exception as exc:
            out.at[idx, "AestheticStatus"] = f"score_failed:{type(exc).__name__}"
            continue
        out.at[idx, "AestheticGoodScore"] = good
        out.at[idx, "AestheticBadPhotoScore"] = bad
        out.at[idx, "AestheticLabel"] = label
        out.at[idx, "AestheticStatus"] = "ok"
    return out


@lru_cache(maxsize=4)
def _load_aesthetic_classifier(model_name: str, device: str):
    prepare_model_cache()
    from transformers import pipeline

    pipe_device = 0 if resolve_device(device) == "cuda" else -1
    return pipeline("image-classification", model=model_name, device=pipe_device)


def summarize_fashionclip_load_error(exc: Exception, model_name: str, *, local_files_only: bool) -> str:
    parts = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    text = " | ".join(parts).replace("\n", " ").strip()
    lowered = text.lower()
    if local_files_only and (
        "local_files_only" in lowered
        or "outgoing traffic has been disabled" in lowered
        or "couldn't connect" in lowered
        or "not the path to a directory" in lowered
        or "cannot find the requested files" in lowered
        or "can't load image processor" in lowered
        or "make sure you don't have a local directory" in lowered
    ):
        return f"load_failed_model_not_cached:{model_name}"
    if "403" in lowered or "forbidden" in lowered:
        return f"load_failed_forbidden:{model_name}"
    if "401" in lowered or "gated repo" in lowered or "restricted" in lowered:
        return f"load_failed_gated_repo:{model_name}"
    if "not found" in lowered or "404" in lowered:
        return f"load_failed_not_found:{model_name}"
    if "temporary failure in name resolution" in lowered or "name resolution" in lowered:
        return f"load_failed_network:{model_name}"
    return f"load_failed:{type(exc).__name__}:{text[:160]}"


@lru_cache(maxsize=4)
def _load_fashionclip_model(model_name: str, device: str, local_files_only: bool):
    prepare_model_cache()
    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only)
    model.eval()
    resolved = resolve_device(device)
    if resolved == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    return processor, model, resolved


def _fashionclip_prompt_scores(image_path: str, processor: Any, model: Any, device: str) -> tuple[float, float]:
    import torch

    prompts = list(FASHIONCLIP_GOOD_PROMPTS) + list(FASHIONCLIP_BAD_PROMPTS)
    good_count = len(FASHIONCLIP_GOOD_PROMPTS)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    if device == "cuda":
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = getattr(outputs, "logits_per_image", None)
    if logits is None:
        raise ValueError("FashionCLIP output did not include logits_per_image")
    probabilities = torch.softmax(logits[0], dim=0).detach().cpu().numpy()
    good_score = float(np.sum(probabilities[:good_count]))
    bad_score = float(np.sum(probabilities[good_count:]))
    return good_score, bad_score


def add_fashionclip_scores(
    frame: pd.DataFrame,
    *,
    model_name: str = DEFAULT_FASHIONCLIP_MODEL,
    device: str = "auto",
    local_files_only: bool = True,
) -> pd.DataFrame:
    out = frame.copy()
    out["FashionClipGoodScore"] = np.nan
    out["FashionClipBadScore"] = np.nan
    out["FashionClipBadPhotoScore"] = np.nan
    out["FashionClipScoreMargin"] = np.nan
    out["FashionClipModelName"] = model_name
    out["FashionClipPromptSet"] = FASHIONCLIP_PROMPT_SET
    out["FashionClipPromptCount"] = len(FASHIONCLIP_GOOD_PROMPTS) + len(FASHIONCLIP_BAD_PROMPTS)
    out["FashionClipStatus"] = "not_run"
    has_images = out.apply(lambda row: bool(primary_image_path(row)), axis=1) if len(out) else pd.Series(dtype=bool)
    if len(out) and not bool(has_images.any()):
        out["FashionClipStatus"] = "missing_image"
        return out
    if not package_available("transformers") or not package_available("torch"):
        out["FashionClipStatus"] = "transformers_or_torch_not_installed"
        return out

    try:
        processor, model, resolved_device = _load_fashionclip_model(model_name, device, bool(local_files_only))
    except Exception as exc:
        out["FashionClipStatus"] = summarize_fashionclip_load_error(exc, model_name, local_files_only=bool(local_files_only))
        return out

    for idx, row in out.iterrows():
        path = primary_image_path(row)
        if not path:
            out.at[idx, "FashionClipStatus"] = "missing_image"
            continue
        try:
            good_score, bad_score = _fashionclip_prompt_scores(path, processor, model, resolved_device)
        except Exception as exc:
            out.at[idx, "FashionClipStatus"] = f"score_failed:{type(exc).__name__}"
            continue
        out.at[idx, "FashionClipGoodScore"] = good_score
        out.at[idx, "FashionClipBadScore"] = bad_score
        out.at[idx, "FashionClipBadPhotoScore"] = bad_score
        out.at[idx, "FashionClipScoreMargin"] = good_score - bad_score
        out.at[idx, "FashionClipStatus"] = "ok_cached" if local_files_only else "ok"
    return out


def _preflight_dino_access(model_name: str) -> None:
    """Surface gated-token errors before Transformers hides them behind a generic processor error."""
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(model_name, "preprocessor_config.json", token=True)
    except TypeError:
        from huggingface_hub import hf_hub_download

        hf_hub_download(model_name, "preprocessor_config.json")


@lru_cache(maxsize=4)
def _load_dino_model(model_name: str, device: str):
    prepare_model_cache()
    import torch
    from transformers import AutoImageProcessor, AutoModel

    _preflight_dino_access(model_name)
    processor_status = "auto_processor"
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, token=True)
    except TypeError:
        processor = AutoImageProcessor.from_pretrained(model_name)
    except Exception:
        processor = ManualDinoImageProcessor()
        processor_status = "manual_processor"
    try:
        model = AutoModel.from_pretrained(model_name, token=True)
    except TypeError:
        model = AutoModel.from_pretrained(model_name)
    model.eval()
    resolved = resolve_device(device)
    if resolved == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    return processor, model, resolved, processor_status


def summarize_dino_load_error(exc: Exception, model_name: str) -> str:
    parts = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    text = " | ".join(parts).replace("\n", " ").strip()
    lowered = text.lower()
    if "fine-grained token" in lowered and "public gated" in lowered:
        return f"load_failed_token_scope_public_gated:{model_name}"
    if "403" in lowered or "forbidden" in lowered:
        return f"load_failed_forbidden:{model_name}"
    if "gated repo" in lowered or "restricted" in lowered or "401" in lowered:
        return f"load_failed_gated_repo:{model_name}"
    if "not found" in lowered or "404" in lowered:
        return f"load_failed_not_found:{model_name}"
    if "temporary failure in name resolution" in lowered or "name resolution" in lowered:
        return f"load_failed_network:{model_name}"
    return f"load_failed:{type(exc).__name__}:{text[:160]}"


def model_feature_columns(methods: Iterable[str] | None) -> list[str]:
    selected = normalize_methods(methods)
    columns: list[str] = []
    for method in selected:
        for column in QUALITY_METHOD_FEATURES.get(method, []):
            if column not in columns:
                columns.append(column)
    return columns


def _dino_embedding(image_path: str, processor: Any, model: Any, device: str) -> tuple[np.ndarray, float]:
    import torch

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    if device == "cuda":
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("DINO output did not include pooler_output or last_hidden_state")
        pooled = hidden[:, 0, :]
    raw_vector = pooled.detach().cpu().numpy()[0].astype(np.float32)
    norm = float(np.linalg.norm(raw_vector))
    return raw_vector / max(norm, 1e-9), norm


def embedding_to_json(vector: np.ndarray) -> str:
    # Six decimals keep the CSV much smaller while preserving enough signal for clustering/modeling.
    rounded = [round(float(value), 6) for value in vector.astype(np.float32).tolist()]
    return json.dumps(rounded, separators=(",", ":"))


def add_dino_scores(
    frame: pd.DataFrame,
    *,
    model_name: str = DEFAULT_DINO_MODEL,
    device: str = "auto",
) -> pd.DataFrame:
    out = frame.copy()
    out["DinoEmbedding"] = ""
    out["DinoEmbeddingDim"] = np.nan
    out["DinoEmbeddingNorm"] = np.nan
    out["DinoOutlierScore"] = np.nan
    out["DinoModelName"] = model_name
    out["DinoStatus"] = "not_run"
    has_images = out.apply(lambda row: bool(primary_image_path(row)), axis=1) if len(out) else pd.Series(dtype=bool)
    if len(out) and not bool(has_images.any()):
        out["DinoStatus"] = "missing_image"
        return out
    if not package_available("transformers") or not package_available("torch"):
        out["DinoStatus"] = "transformers_or_torch_not_installed"
        return out

    try:
        processor, model, resolved_device, processor_status = _load_dino_model(model_name, device)
    except Exception as exc:
        out["DinoStatus"] = summarize_dino_load_error(exc, model_name)
        return out

    embeddings: dict[int, np.ndarray] = {}
    for idx, row in out.iterrows():
        path = primary_image_path(row)
        if not path:
            out.at[idx, "DinoStatus"] = "missing_image"
            continue
        try:
            vector, embedding_norm = _dino_embedding(path, processor, model, resolved_device)
        except Exception as exc:
            out.at[idx, "DinoStatus"] = f"score_failed:{type(exc).__name__}"
            continue
        embeddings[idx] = vector
        out.at[idx, "DinoEmbedding"] = embedding_to_json(vector)
        out.at[idx, "DinoEmbeddingDim"] = int(vector.shape[0])
        out.at[idx, "DinoEmbeddingNorm"] = embedding_norm
        out.at[idx, "DinoModelName"] = model_name
        out.at[idx, "DinoStatus"] = "ok" if processor_status == "auto_processor" else "ok_manual_processor"

    if not embeddings:
        return out
    matrix = np.vstack([embeddings[idx] for idx in embeddings])
    center = np.median(matrix, axis=0)
    distances = np.linalg.norm(matrix - center, axis=1)
    if len(distances) > 1 and float(distances.max() - distances.min()) > 1e-9:
        outlier_scores = (distances - distances.min()) / (distances.max() - distances.min())
    else:
        outlier_scores = np.zeros_like(distances)
    for idx, score in zip(embeddings, outlier_scores):
        out.at[idx, "DinoOutlierScore"] = float(np.clip(score, 0.0, 1.0))
    return out


def combine_bad_photo_scores(frame: pd.DataFrame) -> pd.Series:
    score_cols = [
        "SimpleBadPhotoScore",
        "PyiqaBadPhotoScore",
        "AestheticBadPhotoScore",
        "FashionClipBadPhotoScore",
        "DinoOutlierScore",
    ]
    available = [numeric_series(frame, col, np.nan) for col in score_cols if col in frame.columns]
    if not available:
        return pd.Series([np.nan] * len(frame), index=frame.index)
    matrix = pd.concat(available, axis=1)
    return matrix.mean(axis=1, skipna=True)


def add_quality_method_scores(frame: pd.DataFrame, config: MethodConfig | None = None) -> pd.DataFrame:
    config = config or MethodConfig()
    methods = normalize_methods(config.methods)
    out = frame.copy()
    if "simple" in methods:
        out = add_simple_quality_scores(out)
    if "pyiqa" in methods:
        out = add_pyiqa_scores(out, model_name=config.pyiqa_model, device=config.device, max_images_per_item=config.max_images_per_item)
    if "aesthetic" in methods:
        out = add_aesthetic_scores(out, model_name=config.aesthetic_model, device=config.device)
    if "fashionclip" in methods:
        out = add_fashionclip_scores(
            out,
            model_name=config.fashionclip_model,
            device=config.device,
            local_files_only=config.fashionclip_local_files_only,
        )
    if "dino" in methods:
        out = add_dino_scores(out, model_name=config.dino_model, device=config.device)
    for column in QUALITY_METHOD_COLUMNS:
        if column not in out.columns:
            is_numeric_column = (
                column.endswith("Score")
                or column.endswith("Norm")
                or column.endswith("Margin")
                or column in {"DinoEmbeddingDim", "FashionClipPromptCount"}
            )
            out[column] = np.nan if is_numeric_column else ""
    out["CombinedBadPhotoScore"] = combine_bad_photo_scores(out)
    status_cols = [col for col in ["PyiqaStatus", "AestheticStatus", "FashionClipStatus", "DinoStatus"] if col in out.columns]
    if status_cols:
        status_map: dict[str, pd.Series] = {}
        for column in dict.fromkeys(status_cols):
            values = out.loc[:, out.columns == column]
            if isinstance(values, pd.DataFrame):
                status_map[column] = values.iloc[:, 0].fillna("").astype(str)
            else:
                status_map[column] = values.fillna("").astype(str)
        status_frame = pd.DataFrame(status_map, index=out.index)
        joined_status = ["|".join(value for value in row if value) for row in status_frame.itertuples(index=False, name=None)]
        out["QualityMethodStatus"] = pd.Series(joined_status, index=out.index, dtype=object)
    else:
        out["QualityMethodStatus"] = "simple_only"
    return out
