#!/usr/bin/env python3
from __future__ import annotations

import ast
import io
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.vinted.it/",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

LUXURY_TERMS = {
    "gucci",
    "prada",
    "louis vuitton",
    "fendi",
    "dior",
    "balenciaga",
    "burberry",
    "chanel",
    "hermes",
}
GAME_TERMS = {"ps4", "ps5", "switch", "xbox", "game", "games", "playstation"}


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(num):
        return default
    return num


def normalize_image_sources(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        values = [str(item).strip() for item in raw if str(item).strip()]
        return values
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [text]


def infer_category(search_name: str, title: str) -> str:
    haystack = f"{search_name} {title}".lower()
    if any(term in haystack for term in LUXURY_TERMS):
        return "luxury"
    if any(term in haystack for term in GAME_TERMS):
        return "game"
    return "generic"


def average_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size, hash_size))
    arr = np.asarray(gray, dtype=np.float32)
    bits = arr >= arr.mean()
    return "".join("1" if bit else "0" for bit in bits.flatten())


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def image_from_source(source: str, timeout: float = 8.0) -> Image.Image:
    if source.startswith(("http://", "https://")):
        response = requests.get(source, headers=IMAGE_REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    return Image.open(Path(source)).convert("RGB")


@dataclass
class ImageMetrics:
    source: str
    width: int
    height: int
    brightness: float
    contrast: float
    blur_score: float
    aspect_ratio: float
    hash_value: str
    quality_score: float
    screenshot_risk: float
    notes: list[str]


def compute_image_metrics(source: str, timeout: float = 8.0) -> ImageMetrics:
    image = image_from_source(source, timeout=timeout)
    width, height = image.size
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    blur_score = float((dx.var() + dy.var()) / 2.0)
    aspect_ratio = float(width / height) if height else 1.0
    hash_value = average_hash(image)

    notes: list[str] = []
    quality_penalty = 0.0
    if min(width, height) < 220:
        quality_penalty += 0.18
        notes.append("low_resolution")
    if brightness < 0.18 or brightness > 0.92:
        quality_penalty += 0.14
        notes.append("extreme_brightness")
    if contrast < 0.10:
        quality_penalty += 0.15
        notes.append("low_contrast")
    if blur_score < 25.0:
        quality_penalty += 0.18
        notes.append("blurry")
    screenshot_risk = 0.0
    if aspect_ratio > 1.25 or aspect_ratio < 0.60:
        screenshot_risk = 0.22
        notes.append("suspicious_aspect_ratio")

    quality_score = clamp(1.0 - quality_penalty - screenshot_risk, 0.0, 1.0)
    return ImageMetrics(
        source=source,
        width=width,
        height=height,
        brightness=brightness,
        contrast=contrast,
        blur_score=blur_score,
        aspect_ratio=aspect_ratio,
        hash_value=hash_value,
        quality_score=quality_score,
        screenshot_risk=screenshot_risk,
        notes=notes,
    )


def clip_text_match_score(image_source: str, positive_label: str, negative_label: str, timeout: float = 8.0) -> float | None:
    try:
        from clip import check_item
    except Exception:
        return None
    try:
        image = image_from_source(image_source, timeout=timeout)
        suffix = ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            image.save(tmp.name)
            probs = check_item(positive_label, negative_label, tmp.name)
        return float(probs[0])
    except Exception:
        return None


def analyze_listing_images(
    image_sources: Any,
    *,
    title: str = "",
    search_name: str = "",
    max_images: int = 6,
    main_image_weight: float = 0.55,
    timeout: float = 8.0,
    enable_clip: bool = False,
) -> dict[str, Any]:
    sources = normalize_image_sources(image_sources)[: max(1, int(max_images))]
    if not sources:
        return {
            "VisualScore": 0.45,
            "VisualRiskPenalty": 0.08,
            "VisualQualityScore": 0.45,
            "VisualCompletenessScore": 0.35,
            "VisualConsistencyScore": 0.55,
            "VisualAuthenticityScore": 0.45,
            "VisualImageCount": 0,
            "VisualUniqueImageCount": 0,
            "VisualLowQualityFraction": 1.0,
            "VisualScreenshotFraction": 0.0,
            "VisualAnalysisNotes": "no_images",
        }

    metrics: list[ImageMetrics] = []
    errors: list[str] = []
    for source in sources:
        try:
            metrics.append(compute_image_metrics(source, timeout=timeout))
        except Exception:
            errors.append("image_fetch_failed")

    if not metrics:
        return {
            "VisualScore": 0.40,
            "VisualRiskPenalty": 0.10,
            "VisualQualityScore": 0.40,
            "VisualCompletenessScore": 0.30,
            "VisualConsistencyScore": 0.55,
            "VisualAuthenticityScore": 0.40,
            "VisualImageCount": len(sources),
            "VisualUniqueImageCount": 0,
            "VisualLowQualityFraction": 1.0,
            "VisualScreenshotFraction": 0.0,
            "VisualAnalysisNotes": "all_images_failed",
        }

    unique_hashes: list[str] = []
    duplicate_count = 0
    notes: set[str] = set(errors)
    for metric in metrics:
        notes.update(metric.notes)
        if any(hamming_distance(metric.hash_value, existing) <= 4 for existing in unique_hashes):
            duplicate_count += 1
        else:
            unique_hashes.append(metric.hash_value)

    image_count = len(metrics)
    unique_count = len(unique_hashes)
    duplicate_fraction = duplicate_count / image_count if image_count else 0.0
    low_quality_fraction = sum(m.quality_score < 0.45 for m in metrics) / image_count
    screenshot_fraction = sum(m.screenshot_risk > 0 for m in metrics) / image_count

    if image_count == 1:
        weighted_quality = metrics[0].quality_score
    else:
        main_weight = clamp(float(main_image_weight), 0.20, 0.90)
        rest_weight = (1.0 - main_weight) / max(1, image_count - 1)
        weighted_quality = metrics[0].quality_score * main_weight + sum(m.quality_score * rest_weight for m in metrics[1:])

    category = infer_category(search_name, title)
    if category == "luxury":
        target_unique_images = 4.0
    elif category == "game":
        target_unique_images = 2.0
    else:
        target_unique_images = 3.0

    coverage_score = clamp(unique_count / target_unique_images, 0.0, 1.0)
    completeness_score = clamp(0.65 * coverage_score + 0.35 * (1.0 - duplicate_fraction), 0.0, 1.0)
    quality_score = clamp(weighted_quality - 0.20 * duplicate_fraction, 0.0, 1.0)

    title_label = " ".join(str(title or search_name).strip().split()[:8]) or str(search_name or "listing item").strip()
    clip_score = None
    if enable_clip and title_label:
        clip_score = clip_text_match_score(
            metrics[0].source,
            positive_label=f"a marketplace listing photo of {title_label}",
            negative_label="an unrelated or misleading marketplace listing photo",
            timeout=timeout,
        )

    if clip_score is None:
        consistency_score = clamp(0.58 + 0.22 * quality_score + 0.20 * (1.0 - screenshot_fraction), 0.0, 1.0)
    else:
        consistency_score = clamp(0.45 * clip_score + 0.35 * quality_score + 0.20 * (1.0 - screenshot_fraction), 0.0, 1.0)
        if clip_score < 0.52:
            notes.add("weak_image_text_match")

    if category == "luxury":
        authenticity_score = clamp(0.45 * completeness_score + 0.35 * quality_score + 0.20 * consistency_score, 0.0, 1.0)
        if authenticity_score < 0.45:
            notes.add("low_authenticity_evidence")
    else:
        authenticity_score = clamp(0.40 + 0.35 * completeness_score + 0.25 * consistency_score, 0.0, 1.0)

    visual_score = clamp(
        0.36 * quality_score + 0.24 * completeness_score + 0.20 * consistency_score + 0.20 * authenticity_score,
        0.0,
        1.0,
    )

    risk_penalty = clamp(
        0.20 * (1.0 - quality_score)
        + 0.12 * (1.0 - completeness_score)
        + 0.10 * (1.0 - consistency_score)
        + (0.12 if category == "luxury" else 0.06) * (1.0 - authenticity_score)
        + 0.10 * duplicate_fraction
        + 0.08 * screenshot_fraction,
        0.0,
        0.55,
    )

    if duplicate_fraction >= 0.50:
        notes.add("duplicate_images")
    if unique_count < target_unique_images:
        notes.add("limited_visual_coverage")
    if low_quality_fraction >= 0.50:
        notes.add("mostly_low_quality_images")

    return {
        "VisualScore": float(visual_score),
        "VisualRiskPenalty": float(risk_penalty),
        "VisualQualityScore": float(quality_score),
        "VisualCompletenessScore": float(completeness_score),
        "VisualConsistencyScore": float(consistency_score),
        "VisualAuthenticityScore": float(authenticity_score),
        "VisualImageCount": int(image_count),
        "VisualUniqueImageCount": int(unique_count),
        "VisualLowQualityFraction": float(low_quality_fraction),
        "VisualScreenshotFraction": float(screenshot_fraction),
        "VisualAnalysisNotes": "|".join(sorted(notes)),
    }
