"""Phase 4: first-image technical quality features, pure image measurements.

No product-type, price, brand, or seller logic here on purpose -- this module only
looks at pixels. Given a first-image path it computes resolution / exposure / blur /
glare / framing measurements, then `percentile_scores` turns those raw measurements
into within-dataset "worse-quality" percentiles and per-defect-family scores.

Resolution of item_id -> image path reuses embed_blocks.build_id_index/first_image
(same cache convention as the rest of multimodal_beat). Build the index ONCE and pass
it into any batch loop; do not rebuild it per image.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
MB = HERE.parent                       # experiments/current/multimodal_beat
sys.path.insert(0, str(MB))
from embed_blocks import build_id_index, first_image  # noqa: E402

FEATURE_KEYS = [
    "width", "height", "minimum_dimension", "pixel_count",
    "mean_luminance", "luminance_std", "dark_pixel_fraction",
    "bright_pixel_fraction", "dynamic_range",
    "gradient_sharpness", "laplacian_variance", "laplacian_variance_half",
    "tenengrad", "edge_density",
    "high_brightness_low_saturation_fraction", "largest_highlight_region_fraction",
    "aspect_ratio", "centre_edge_density", "border_edge_density",
    "centre_to_border_edge_ratio",
]

EDGE_THRESHOLD = 0.06
EPS = 1e-6
MAX_SIDE = 512


def resolve_first_image(item_id, id_index: dict[str, str]):
    """item_id -> Path|None via a pre-built id_index (see build_id_index())."""
    item_dir = id_index.get(str(item_id))
    if item_dir is None:
        return None
    return first_image(item_dir)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _gradient_stats(lum: np.ndarray):
    gy, gx = np.gradient(lum)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    gradient_sharpness = float(np.var(mag))
    tenengrad = float(np.mean(gx ** 2 + gy ** 2))
    edge_density = float(np.mean(mag > EDGE_THRESHOLD))
    return gradient_sharpness, tenengrad, edge_density


def _laplacian_variance(lum: np.ndarray) -> float:
    """4-neighbor discrete Laplacian via numpy slicing (no scipy/cv2)."""
    if lum.shape[0] < 3 or lum.shape[1] < 3:
        return 0.0
    centre = lum[1:-1, 1:-1]
    up = lum[:-2, 1:-1]
    down = lum[2:, 1:-1]
    left = lum[1:-1, :-2]
    right = lum[1:-1, 2:]
    lap = up + down + left + right - 4 * centre
    return float(np.var(lap))


def _saturation(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    return (mx - mn) / np.clip(mx, EPS, None)


def _largest_highlight_region_fraction(highlight: np.ndarray) -> float:
    """Fraction of pixels in the largest connected highlight blob.
    ponytail: uses scipy.ndimage.label when available (cheap, no extra dep beyond an
    optional import); if scipy is missing we fall back to the total highlight
    fraction (i.e. treat all highlight pixels as one region) rather than
    implementing a bespoke flood-fill, since scipy is the far simpler/safer path
    and is present in the target env in practice."""
    total = highlight.size
    if total == 0:
        return 0.0
    highlight_frac = float(highlight.sum()) / total
    if highlight_frac == 0.0:
        return 0.0
    try:
        from scipy import ndimage
        labeled, n = ndimage.label(highlight)
        if n == 0:
            return 0.0
        counts = np.bincount(labeled.ravel())
        counts[0] = 0  # background
        return float(counts.max()) / total
    except ImportError:
        return highlight_frac


def _downscale(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return img
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)


def _nan_result() -> dict[str, float]:
    d = {k: float("nan") for k in FEATURE_KEYS}
    d["feature_error"] = 1.0
    return d


def extract_features(image_path) -> dict[str, float]:
    """Pure pixel-measurement features for one first-image. See module docstring for
    the exact key list and definitions. Returns all-nan + feature_error=1.0 on any
    failure to load/decode."""
    try:
        img = Image.open(image_path)
        img.load()
        img = img.convert("RGB")
    except Exception:
        return _nan_result()

    try:
        width, height = img.size
        minimum_dimension = float(min(width, height))
        pixel_count = float(width * height)
        aspect_ratio = float(width) / float(height) if height else float("nan")

        small = _downscale(img)
        rgb = np.asarray(small, dtype=np.float64) / 255.0
        lum = _luminance(rgb)

        mean_luminance = float(np.mean(lum))
        luminance_std = float(np.std(lum))
        dark_pixel_fraction = float(np.mean(lum < 0.05))
        bright_pixel_fraction = float(np.mean(lum > 0.95))
        dynamic_range = float(np.percentile(lum, 95) - np.percentile(lum, 5))

        gradient_sharpness, tenengrad, edge_density = _gradient_stats(lum)
        laplacian_variance = _laplacian_variance(lum)
        half_h, half_w = max(2, lum.shape[0] // 2), max(2, lum.shape[1] // 2)
        lum_half = np.asarray(
            Image.fromarray((lum * 255).astype(np.uint8)).resize((half_w, half_h), Image.BILINEAR),
            dtype=np.float64,
        ) / 255.0
        laplacian_variance_half = _laplacian_variance(lum_half)

        sat = _saturation(rgb)
        glare_mask = (lum > 0.9) & (sat < 0.15)
        high_brightness_low_saturation_fraction = float(np.mean(glare_mask))
        highlight_mask = lum > 0.92
        largest_highlight_region_fraction = _largest_highlight_region_fraction(highlight_mask)

        h, w = lum.shape
        y0, y1 = h // 4, h - h // 4
        x0, x1 = w // 4, w - w // 4
        gy, gx = np.gradient(lum)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        centre_mag = mag[y0:y1, x0:x1]
        border_mask = np.ones_like(mag, dtype=bool)
        border_mask[y0:y1, x0:x1] = False
        centre_edge_density = float(np.mean(centre_mag > EDGE_THRESHOLD)) if centre_mag.size else float("nan")
        border_vals = mag[border_mask]
        border_edge_density = float(np.mean(border_vals > EDGE_THRESHOLD)) if border_vals.size else float("nan")
        centre_to_border_edge_ratio = centre_edge_density / (border_edge_density + EPS)

        return {
            "width": float(width), "height": float(height),
            "minimum_dimension": minimum_dimension, "pixel_count": pixel_count,
            "mean_luminance": mean_luminance, "luminance_std": luminance_std,
            "dark_pixel_fraction": dark_pixel_fraction,
            "bright_pixel_fraction": bright_pixel_fraction,
            "dynamic_range": dynamic_range,
            "gradient_sharpness": gradient_sharpness,
            "laplacian_variance": laplacian_variance,
            "laplacian_variance_half": laplacian_variance_half,
            "tenengrad": tenengrad, "edge_density": edge_density,
            "high_brightness_low_saturation_fraction": high_brightness_low_saturation_fraction,
            "largest_highlight_region_fraction": largest_highlight_region_fraction,
            "aspect_ratio": aspect_ratio,
            "centre_edge_density": centre_edge_density,
            "border_edge_density": border_edge_density,
            "centre_to_border_edge_ratio": centre_to_border_edge_ratio,
            "feature_error": 0.0,
        }
    except Exception:
        return _nan_result()


def _bad_pct(s: pd.Series, valid: pd.Series, invert: bool) -> pd.Series:
    """rank(pct=True) bad-quality percentile, computed only over `valid` rows."""
    out = pd.Series(np.nan, index=s.index)
    ranked = s[valid].rank(pct=True)
    out[valid] = (1.0 - ranked) if invert else ranked
    return out


def percentile_scores(df: pd.DataFrame, feature_cols_config=None) -> pd.DataFrame:
    """Add simple_* component scores + simple_overall_score + simple_top_defect to df.
    `feature_cols_config` is accepted for interface symmetry with other phases but the
    feature->component mapping is fixed by spec; pass None."""
    out = df.copy()
    valid = out["feature_error"] == 0.0

    blur_components = {
        "gradient_sharpness": True, "laplacian_variance": True,
        "tenengrad": True, "edge_density": True,
    }
    exposure_components = {
        "dark_pixel_fraction": False, "bright_pixel_fraction": False,
        "dynamic_range": True,
    }
    resolution_components = {"minimum_dimension": True, "pixel_count": True}
    glare_components = {
        "high_brightness_low_saturation_fraction": False,
        "largest_highlight_region_fraction": False,
    }
    framing_components = {"centre_to_border_edge_ratio": True}

    def stack_max(components: dict[str, bool]) -> pd.Series:
        cols = [_bad_pct(out[c], valid, inv) for c, inv in components.items()]
        return pd.concat(cols, axis=1).max(axis=1)

    out["_lum_dev"] = (out["mean_luminance"] - 0.5).abs()
    exposure_cols = [_bad_pct(out[c], valid, inv) for c, inv in exposure_components.items()]
    exposure_cols.append(_bad_pct(out["_lum_dev"], valid, invert=False))
    out["simple_exposure_score"] = pd.concat(exposure_cols, axis=1).max(axis=1)
    out.drop(columns=["_lum_dev"], inplace=True)

    out["simple_blur_score"] = stack_max(blur_components)
    out["simple_resolution_score"] = stack_max(resolution_components)
    out["simple_glare_score"] = stack_max(glare_components)

    out["_log_aspect_abs"] = np.log(out["aspect_ratio"].clip(lower=EPS)).abs()
    framing_cols = [_bad_pct(out[c], valid, inv) for c, inv in framing_components.items()]
    framing_cols.append(_bad_pct(out["_log_aspect_abs"], valid, invert=False))
    out["simple_framing_score"] = pd.concat(framing_cols, axis=1).max(axis=1)
    out.drop(columns=["_log_aspect_abs"], inplace=True)

    component_names = ["blur", "exposure", "resolution", "glare", "framing"]
    component_cols = [f"simple_{n}_score" for n in component_names]
    comp_df = out[component_cols]
    out["simple_overall_score"] = comp_df.max(axis=1)
    # idxmax raises on all-NaN rows in this pandas version -> compute only on valid rows
    out["simple_top_defect"] = pd.Series([np.nan] * len(out), index=out.index, dtype=object)
    valid_rows = valid & comp_df.notna().any(axis=1)
    if valid_rows.any():
        idxmax = comp_df.loc[valid_rows].idxmax(axis=1)
        out.loc[valid_rows, "simple_top_defect"] = idxmax.map(
            lambda c: c.replace("simple_", "").replace("_score", "") if isinstance(c, str) else np.nan
        )
    return out


if __name__ == "__main__":
    idx = build_id_index()
    test_path = None
    for item_id, item_dir in idx.items():
        p = first_image(item_dir)
        if p is not None:
            test_path = p
            break

    if test_path is None:
        for root in (
            "/home/ale/Desktop/vinted/Vinted_New_Version/experiments/current/time_to_sell/data",
            "/home/ale/Desktop/vinted/Vinted_New_Version/experiments/old/photo_arbitrage/data",
        ):
            for pattern in ("**/*.webp", "**/*.jpg"):
                hits = list(Path(root).glob(pattern))
                if hits:
                    test_path = hits[0]
                    break
            if test_path is not None:
                break

    print(f"test image: {test_path}")
    feats = extract_features(test_path) if test_path is not None else _nan_result()
    for k, v in feats.items():
        print(f"  {k}: {v}")
