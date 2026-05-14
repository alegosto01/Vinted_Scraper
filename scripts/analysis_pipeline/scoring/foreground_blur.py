#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
for path in (SCRIPTS_DIR, ANALYSIS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
from PIL import Image, ImageFilter

from analysis_pipeline.scoring.visual_rerank import image_from_source


@dataclass
class ForegroundBlurMetrics:
    source: str
    segmentation_backend: str
    segmentation_confidence: float
    foreground_fraction: float
    whole_gradient_sharpness: float
    whole_laplacian_variance: float
    whole_tenengrad: float
    foreground_laplacian_variance: float
    foreground_tenengrad: float
    background_laplacian_variance: float
    background_tenengrad: float
    foreground_sharpness_score: float
    foreground_relative_sharpness: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def resize_for_segmentation(image: Image.Image, max_side: int = 384) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def normalize_quantile(values: np.ndarray, upper_q: float = 0.98) -> np.ndarray:
    values = values.astype(np.float32)
    lower = float(values.min())
    upper = float(np.quantile(values, upper_q))
    if upper <= lower + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def gaussian_center_prior(height: int, width: int, sigma: float = 0.30) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    sy = max(height * sigma, 1.0)
    sx = max(width * sigma, 1.0)
    dist = ((yy - cy) ** 2) / (2.0 * sy * sy) + ((xx - cx) ** 2) / (2.0 * sx * sx)
    return np.exp(-dist).astype(np.float32)


def border_pixels(rgb: np.ndarray, border: int) -> np.ndarray:
    top = rgb[:border, :, :]
    bottom = rgb[-border:, :, :]
    left = rgb[:, :border, :]
    right = rgb[:, -border:, :]
    return np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    )


def binary_filter(mask: np.ndarray, *, max_size: int | None = None, min_size: int | None = None) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if max_size and max_size >= 3:
        image = image.filter(ImageFilter.MaxFilter(max_size))
    if min_size and min_size >= 3:
        image = image.filter(ImageFilter.MinFilter(min_size))
    return np.asarray(image) > 127


def closest_true_pixel(mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    cy = (mask.shape[0] - 1) / 2.0
    cx = (mask.shape[1] - 1) / 2.0
    d2 = (coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2
    y, x = coords[int(d2.argmin())]
    return int(y), int(x)


def central_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    if mask[cy, cx]:
        seed = (cy, cx)
    else:
        seed = closest_true_pixel(mask)
        if seed is None:
            return mask
    out = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque([seed])
    out[seed] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                q.append((ny, nx))
    return out


def center_fallback_mask(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    ry = max(height * 0.33, 1.0)
    rx = max(width * 0.33, 1.0)
    ellipse = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    return ellipse


def heuristic_foreground_mask(image: Image.Image) -> tuple[np.ndarray, float]:
    small = resize_for_segmentation(image)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    border = max(6, int(round(min(height, width) * 0.09)))
    border_rgb = border_pixels(rgb, border)
    bg_mean = border_rgb.mean(axis=0)
    bg_std = border_rgb.std(axis=0) + 0.05
    color_distance = np.sqrt((((rgb - bg_mean) / bg_std) ** 2).sum(axis=2))

    gray = rgb.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    texture = dx + dy
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    center = gaussian_center_prior(height, width)

    score = (
        0.50 * normalize_quantile(color_distance)
        + 0.20 * normalize_quantile(saturation)
        + 0.10 * normalize_quantile(texture)
        + 0.20 * center
    )
    threshold = max(0.45, float(np.quantile(score, 0.68)))
    mask = score >= threshold
    mask = binary_filter(mask, max_size=5, min_size=5)
    mask = binary_filter(mask, min_size=3, max_size=3)
    mask = central_component(mask)

    if mask.mean() < 0.04 or mask.mean() > 0.82:
        mask = center_fallback_mask(height, width)
        confidence = 0.25
    else:
        border_region = np.concatenate(
            [
                mask[:border, :].reshape(-1),
                mask[-border:, :].reshape(-1),
                mask[:, :border].reshape(-1),
                mask[:, -border:].reshape(-1),
            ]
        )
        border_fraction = float(border_region.mean()) if border_region.size else 0.0
        center_hit = 1.0 if mask[height // 2, width // 2] else 0.0
        confidence = clamp(0.65 + 0.25 * center_hit - 0.35 * border_fraction, 0.15, 0.95)

    up = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(image.size, Image.Resampling.NEAREST)
    return np.asarray(up) > 127, confidence


def sam_foreground_mask(image: Image.Image) -> tuple[np.ndarray, float]:
    model, predictor_cls = get_sam_model_components()
    predictor = predictor_cls(model)
    image_np = np.asarray(image)
    predictor.set_image(image_np)

    seed_mask, seed_confidence = heuristic_foreground_mask(image)
    coords = np.argwhere(seed_mask)
    if coords.size == 0:
        raise RuntimeError("Heuristic seed mask was empty")

    cy = float(coords[:, 0].mean())
    cx = float(coords[:, 1].mean())
    y0, x0, y1, x1 = foreground_bbox(seed_mask)
    point_coords = np.asarray([[cx, cy]], dtype=np.float32)
    point_labels = np.asarray([1], dtype=np.int32)
    box = np.asarray([x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM predictor returned no masks")

    best_idx = int(np.argmax(scores))
    best = np.asarray(masks[best_idx], dtype=bool)
    overlap = float((best & seed_mask).sum()) / max(float((best | seed_mask).sum()), 1.0)
    confidence = clamp(0.45 * float(scores[best_idx]) + 0.35 * overlap + 0.20 * seed_confidence, 0.15, 0.98)
    return best, confidence


def resolve_sam_checkpoint(model_type: str) -> str:
    explicit = os.environ.get("SAM_CHECKPOINT")
    if explicit and Path(explicit).exists():
        return explicit

    candidate_names = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
    }
    filename = candidate_names.get(model_type, candidate_names["vit_b"])
    candidate_paths = [
        ROOT / "data/models/sam" / filename,
        ROOT / "models/sam" / filename,
    ]
    for path in candidate_paths:
        if path.exists():
            return str(path)
    raise RuntimeError(f"SAM checkpoint not found for model_type={model_type}. Set SAM_CHECKPOINT or place {filename} under data/models/sam.")


@lru_cache(maxsize=1)
def get_sam_model_components():
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except Exception as exc:
        raise RuntimeError("segment_anything and torch/torchvision are required for backend='sam'") from exc

    model_type = os.environ.get("SAM_MODEL_TYPE", "vit_b")
    checkpoint = resolve_sam_checkpoint(model_type)
    device = os.environ.get("SAM_DEVICE")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device=device)
    model.eval()
    return model, SamPredictor


def segment_foreground(image: Image.Image, backend: str = "auto") -> tuple[np.ndarray, str, float]:
    if backend in {"auto", "sam"}:
        try:
            mask, confidence = sam_foreground_mask(image)
            return mask, "sam", confidence
        except Exception:
            if backend == "sam":
                raise
    mask, confidence = heuristic_foreground_mask(image)
    return mask, "heuristic", confidence


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(kh):
        for x in range(kw):
            out += kernel[y, x] * padded[y:y + image.shape[0], x:x + image.shape[1]]
    return out


def blur_stats(gray: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    lap_kernel = np.asarray([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    sobel_x = np.asarray([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32)
    sobel_y = sobel_x.T
    lap = convolve2d(gray, lap_kernel)
    gx = convolve2d(gray, sobel_x)
    gy = convolve2d(gray, sobel_y)
    grad_energy = gx * gx + gy * gy
    if mask is None:
        pixels = np.ones_like(gray, dtype=bool)
    else:
        pixels = mask.astype(bool)
        if int(pixels.sum()) < 64:
            pixels = np.ones_like(gray, dtype=bool)
    return float(lap[pixels].var()), float(grad_energy[pixels].mean())


def foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, 0, mask.shape[0], mask.shape[1]
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    pad_y = max(2, int(round((y1 - y0) * 0.05)))
    pad_x = max(2, int(round((x1 - x0) * 0.05)))
    return max(0, y0 - pad_y), max(0, x0 - pad_x), min(mask.shape[0], y1 + pad_y), min(mask.shape[1], x1 + pad_x)


def save_segmentation_preview(image: Image.Image, mask: np.ndarray, output_path: str | Path) -> None:
    rgb = np.asarray(image, dtype=np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)
    overlay = rgb.copy()
    overlay[..., 0] = np.where(mask, np.clip(overlay[..., 0] * 0.5 + 120, 0, 255), overlay[..., 0])
    overlay[..., 1] = np.where(mask, np.clip(overlay[..., 1] * 0.85, 0, 255), overlay[..., 1])
    y0, x0, y1, x1 = foreground_bbox(mask)
    crop = rgb[y0:y1, x0:x1, :]
    panels = [
        image.resize((256, 256), Image.Resampling.BILINEAR),
        Image.fromarray(mask_u8, mode="L").convert("RGB").resize((256, 256), Image.Resampling.NEAREST),
        Image.fromarray(overlay, mode="RGB").resize((256, 256), Image.Resampling.BILINEAR),
        Image.fromarray(crop, mode="RGB").resize((256, 256), Image.Resampling.BILINEAR),
    ]
    canvas = Image.new("RGB", (512, 512), (255, 255, 255))
    positions = [(0, 0), (256, 0), (0, 256), (256, 256)]
    for panel, pos in zip(panels, positions):
        canvas.paste(panel, pos)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def compute_foreground_blur_metrics(
    source: str,
    *,
    backend: str = "auto",
    timeout: float = 8.0,
    save_preview_to: str | Path | None = None,
) -> ForegroundBlurMetrics:
    image = image_from_source(source, timeout=timeout)
    image = image.convert("RGB")
    mask, used_backend, confidence = segment_foreground(image, backend=backend)
    if save_preview_to is not None:
        save_segmentation_preview(image, mask, save_preview_to)

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    whole_gradient = float((dx.var() + dy.var()) / 2.0)
    whole_lap, whole_ten = blur_stats(gray, None)
    fg_lap, fg_ten = blur_stats(gray, mask)
    bg_mask = ~mask
    bg_lap, bg_ten = blur_stats(gray, bg_mask)

    foreground_score = math.log1p(max(fg_lap, 0.0)) + 0.35 * math.log1p(max(fg_ten, 0.0))
    background_score = math.log1p(max(bg_lap, 0.0)) + 0.35 * math.log1p(max(bg_ten, 0.0))
    relative = foreground_score / max(background_score, 1e-6)

    return ForegroundBlurMetrics(
        source=source,
        segmentation_backend=used_backend,
        segmentation_confidence=float(confidence),
        foreground_fraction=float(mask.mean()),
        whole_gradient_sharpness=float(whole_gradient),
        whole_laplacian_variance=float(whole_lap),
        whole_tenengrad=float(whole_ten),
        foreground_laplacian_variance=float(fg_lap),
        foreground_tenengrad=float(fg_ten),
        background_laplacian_variance=float(bg_lap),
        background_tenengrad=float(bg_ten),
        foreground_sharpness_score=float(foreground_score),
        foreground_relative_sharpness=float(relative),
    )
