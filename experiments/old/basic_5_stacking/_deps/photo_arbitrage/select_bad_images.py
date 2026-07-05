#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.old.basic_5_stacking._deps.photo_arbitrage.paths import EXPERIMENT_ROOT, assert_photo_path, utc_now_iso


ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_INPUT = EXPERIMENT_ROOT / "features" / "sold_unsold_visuals_20260514_full" / "combined_scored.csv"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "selected_bad_images"

IDENTITY_COLUMNS = [
    "SearchName",
    "PhotoOutcomeLabel",
    "Title",
    "Price",
    "Brand",
    "Link",
    "Dataid",
    "item_id",
    "LocalPrimaryImagePath",
]

LOW_IS_BAD = [
    "PyiqaQualityScore",
    "AestheticGoodScore",
    "PhotoImageCount",
    "PhotoUsableImageCount",
    "PictureCountNum",
    "PhotoPrimaryWidth",
    "PhotoPrimaryHeight",
    "PhotoPrimaryContrast",
    "PhotoAvgContrast",
    "PhotoPrimarySaturation",
    "PhotoAvgSaturation",
    "PhotoPrimarySharpness",
    "PhotoAvgSharpness",
    "PhotoPrimaryEdgeDensity",
    "PhotoAvgEdgeDensity",
    "DerivedPrimaryShortSide",
    "DerivedPrimaryAreaMegapixels",
    "DerivedPrimaryEntropy",
    "DerivedPrimaryColorfulness",
]

HIGH_IS_BAD = [
    "SimpleBadPhotoScore",
    "PyiqaBadPhotoScore",
    "AestheticBadPhotoScore",
    "DinoOutlierScore",
    "CombinedBadPhotoScore",
    "PhotoDuplicateImageCount",
    "PhotoLowQualityFraction",
    "PhotoScreenshotRiskFraction",
    "PhotoOnlyOneImage",
    "PhotoMissingImages",
    "DerivedPrimaryDarkPixelFraction",
    "DerivedPrimaryBrightPixelFraction",
    "DerivedPrimaryClippedPixelFraction",
    "DerivedPrimaryAspectRatioDeviation",
    "DerivedPrimaryNearBlankScore",
]

BOTH_DIRECTIONS = [
    "PhotoPrimaryBrightness",
    "PhotoAvgBrightness",
    "PhotoPrimaryAspectRatio",
    "DinoEmbeddingNorm",
]

EXTREME_FEATURE_TARGETS = {
    "PhotoPrimaryBrightness": 0.50,
    "PhotoAvgBrightness": 0.50,
}


@dataclass(frozen=True)
class FeatureSpec:
    column: str
    mode: str
    folder: str
    reason: str
    target: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy worst-scored photo examples into per-feature review folders.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Scored photo CSV to sample from.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT), help="Output directory under the photo arbitrage experiment root.")
    parser.add_argument("--top-n", type=int, default=50, help="Images to copy per feature.")
    parser.add_argument("--max-derived-images", type=int, default=None, help="Optional cap while computing extra image metrics.")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete previously generated files inside the output dir.")
    return parser.parse_args()


def sanitize(value: Any, *, max_len: int = 80) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "unknown")[:max_len].strip("_") or "unknown"


def numeric_text(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p")


def finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def existing_image_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "LocalPrimaryImagePath" not in df.columns:
        raise ValueError("Input CSV must include LocalPrimaryImagePath")
    out = df.copy()
    paths = out["LocalPrimaryImagePath"].fillna("").astype(str).str.strip()
    exists = paths.map(lambda value: bool(value) and Path(value).exists())
    out = out.loc[exists].copy()
    out["LocalPrimaryImagePath"] = paths.loc[exists]
    return out.reset_index(drop=True)


def resize_for_metrics(image: Image.Image, max_side: int = 256) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_side:
        return image.copy()
    scale = max_side / float(max(width, height))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.BICUBIC)


def image_entropy(gray: np.ndarray) -> float:
    counts = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    probs = counts[counts > 0] / max(float(counts.sum()), 1.0)
    return float(-(probs * np.log2(probs)).sum())


def colorfulness(rgb: np.ndarray) -> float:
    rgb255 = rgb.astype(np.float32)
    red, green, blue = rgb255[:, :, 0], rgb255[:, :, 1], rgb255[:, :, 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    std_root = math.sqrt(float(np.std(rg) ** 2 + np.std(yb) ** 2))
    mean_root = math.sqrt(float(np.mean(rg) ** 2 + np.mean(yb) ** 2))
    return float((std_root + 0.3 * mean_root) / 255.0)


def compute_derived_metrics(path: str) -> dict[str, float]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        small = resize_for_metrics(image)
    gray = np.asarray(small.convert("L"), dtype=np.uint8)
    rgb = np.asarray(small, dtype=np.uint8)
    gray_float = gray.astype(np.float32)
    gy, gx = np.gradient(gray_float)
    contrast = float(gray_float.std() / 255.0)
    edge_density = float((np.hypot(gx, gy) > 20.0).mean())
    entropy = image_entropy(gray)
    dark_fraction = float((gray <= 25).mean())
    bright_fraction = float((gray >= 245).mean())
    clipped_fraction = float(np.clip(dark_fraction + bright_fraction, 0.0, 1.0))
    aspect_ratio = float(width / height) if height else np.nan
    aspect_deviation = float(abs(math.log(aspect_ratio / 0.75))) if aspect_ratio and math.isfinite(aspect_ratio) else np.nan
    low_contrast = np.clip((0.16 - contrast) / 0.16, 0.0, 1.0)
    low_entropy = np.clip((3.5 - entropy) / 3.5, 0.0, 1.0)
    low_edges = np.clip((0.02 - edge_density) / 0.02, 0.0, 1.0)
    near_blank = float(np.clip(0.45 * low_contrast + 0.35 * low_entropy + 0.20 * low_edges, 0.0, 1.0))
    return {
        "DerivedPrimaryShortSide": float(min(width, height)),
        "DerivedPrimaryAreaMegapixels": float((width * height) / 1_000_000.0),
        "DerivedPrimaryEntropy": entropy,
        "DerivedPrimaryDarkPixelFraction": dark_fraction,
        "DerivedPrimaryBrightPixelFraction": bright_fraction,
        "DerivedPrimaryClippedPixelFraction": clipped_fraction,
        "DerivedPrimaryAspectRatioDeviation": aspect_deviation,
        "DerivedPrimaryColorfulness": colorfulness(rgb),
        "DerivedPrimaryNearBlankScore": near_blank,
    }


def add_derived_features(df: pd.DataFrame, *, max_images: int | None = None) -> pd.DataFrame:
    out = df.copy()
    rows: list[dict[str, float]] = []
    total = len(out) if max_images is None else min(len(out), int(max_images))
    for pos, path in enumerate(out["LocalPrimaryImagePath"].head(total), start=1):
        try:
            rows.append(compute_derived_metrics(str(path)))
        except Exception:
            rows.append({})
        if pos % 2000 == 0:
            print(f"computed derived image metrics for {pos}/{total}", flush=True)
    if total < len(out):
        rows.extend({} for _ in range(len(out) - total))
    derived = pd.DataFrame(rows, index=out.index)
    for column in sorted(set(LOW_IS_BAD + HIGH_IS_BAD) & set(derived.columns)):
        out[column] = pd.to_numeric(derived[column], errors="coerce")
    return out


def feature_specs(columns: set[str]) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for column in LOW_IS_BAD:
        if column in columns:
            specs.append(
                FeatureSpec(
                    column=column,
                    mode="low",
                    folder=sanitize(f"{column}_low"),
                    reason="lowest values are expected to be worse",
                )
            )
    for column in HIGH_IS_BAD:
        if column in columns:
            specs.append(
                FeatureSpec(
                    column=column,
                    mode="high",
                    folder=sanitize(f"{column}_high"),
                    reason="highest values are expected to be worse",
                )
            )
    for column in BOTH_DIRECTIONS:
        if column in columns:
            specs.append(
                FeatureSpec(
                    column=column,
                    mode="low",
                    folder=sanitize(f"{column}_low"),
                    reason="very low values may indicate bad photos or failed scoring",
                )
            )
            specs.append(
                FeatureSpec(
                    column=column,
                    mode="high",
                    folder=sanitize(f"{column}_high"),
                    reason="very high values may indicate bad photos, screenshots, or odd scoring",
                )
            )
    for column, target in EXTREME_FEATURE_TARGETS.items():
        if column in columns:
            specs.append(
                FeatureSpec(
                    column=column,
                    mode="extreme",
                    folder=sanitize(f"{column}_extreme"),
                    reason=f"values farthest from {target:g} are expected to be worse",
                    target=target,
                )
            )
    return specs


def clean_output_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    if out_dir.resolve() == EXPERIMENT_ROOT.resolve():
        raise ValueError(f"Refusing to clean experiment root directly: {out_dir}")
    for child in out_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def ordered_candidates(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    work = df.copy()
    raw = finite_series(work, spec.column)
    work["_feature_raw_value"] = raw
    finite = raw.dropna()
    if finite.empty or finite.nunique() <= 1:
        return work.iloc[0:0].copy()
    if spec.mode == "low":
        work["_feature_severity"] = -raw
        work = work[raw.notna() & raw.lt(float(finite.max()))].sort_values("_feature_raw_value", ascending=True, kind="stable")
    elif spec.mode == "high":
        work["_feature_severity"] = raw
        work = work[raw.notna() & raw.gt(float(finite.min()))].sort_values("_feature_raw_value", ascending=False, kind="stable")
    elif spec.mode == "extreme":
        if spec.target is None:
            raise ValueError(f"Extreme feature needs a target: {spec.column}")
        severity = (raw - float(spec.target)).abs()
        work["_feature_severity"] = severity
        finite_severity = severity.dropna()
        if finite_severity.empty or finite_severity.nunique() <= 1:
            return work.iloc[0:0].copy()
        work = work[severity.notna() & severity.gt(float(finite_severity.min()))].sort_values(
            "_feature_severity", ascending=False, kind="stable"
        )
    else:
        raise ValueError(f"Unknown feature mode: {spec.mode}")
    return work.drop_duplicates("LocalPrimaryImagePath", keep="first")


def selected_filename(rank: int, row: pd.Series, spec: FeatureSpec) -> str:
    source = Path(str(row["LocalPrimaryImagePath"]))
    suffix = source.suffix.lower() or ".img"
    search = sanitize(row.get("SearchName"), max_len=28)
    identity = row.get("Dataid") or row.get("item_id") or row.get("Link") or source.stem
    identity_text = sanitize(identity, max_len=36)
    value = numeric_text(float(row["_feature_raw_value"]))
    return f"{rank:03d}__value_{value}__{search}__{identity_text}{suffix}"


def copy_selection(out_dir: Path, frame: pd.DataFrame, spec: FeatureSpec, top_n: int) -> pd.DataFrame:
    feature_dir = assert_photo_path(out_dir / spec.folder)
    feature_dir.mkdir(parents=True, exist_ok=True)
    selected = frame.head(top_n).copy()
    copied_paths: list[str] = []
    ranks: list[int] = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        source = Path(str(row["LocalPrimaryImagePath"]))
        destination = feature_dir / selected_filename(rank, row, spec)
        shutil.copy2(source, destination)
        copied_paths.append(str(destination.relative_to(out_dir)))
        ranks.append(rank)
    selected.insert(0, "SelectionRank", ranks)
    selected.insert(1, "SelectionFeature", spec.column)
    selected.insert(2, "SelectionMode", spec.mode)
    selected.insert(3, "SelectionReason", spec.reason)
    selected.insert(4, "CopiedImageRelativePath", copied_paths)
    keep = [
        "SelectionRank",
        "SelectionFeature",
        "SelectionMode",
        "SelectionReason",
        "CopiedImageRelativePath",
        "_feature_raw_value",
        "_feature_severity",
        *[col for col in IDENTITY_COLUMNS if col in selected.columns],
    ]
    keep.extend(col for col in ["PyiqaStatus", "AestheticStatus", "DinoStatus", "QualityMethodStatus"] if col in selected.columns)
    keep = list(dict.fromkeys(keep))
    selected_index = selected[keep].rename(
        columns={
            "_feature_raw_value": "FeatureRawValue",
            "_feature_severity": "FeatureSeverity",
        }
    )
    selected_index.to_csv(feature_dir / "index.csv", index=False)
    return selected_index


def write_readme(out_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Selected Bad Images",
        "",
        "Each subfolder contains the 50 worst primary images for one visual feature or feature direction.",
        "Use `index.csv` inside each subfolder for the listing metadata and exact score value.",
        "",
        f"Source CSV: `{manifest['source_csv']}`",
        f"Generated at: `{manifest['created_at']}`",
        f"Rows with existing primary images: `{manifest['rows_with_existing_primary_images']}`",
        f"Feature folders written: `{manifest['feature_folders_written']}`",
        "",
        "Directions:",
        "- `_low`: lowest raw values were copied.",
        "- `_high`: highest raw values were copied.",
        "- `_extreme`: values farthest from the target were copied.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    out_dir = assert_photo_path(Path(args.out_dir))
    if not source.exists():
        raise FileNotFoundError(f"Input CSV not found: {source}")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")

    df = pd.read_csv(source, low_memory=False)
    df = existing_image_frame(df)
    df = add_derived_features(df, max_images=args.max_derived_images)
    if not args.no_clean:
        clean_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = feature_specs(set(df.columns))
    all_indexes: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    for spec in specs:
        ordered = ordered_candidates(df, spec)
        if ordered.empty:
            skipped.append({"feature": spec.column, "mode": spec.mode, "reason": "no varied finite values with existing images"})
            continue
        selected_index = copy_selection(out_dir, ordered, spec, int(args.top_n))
        all_indexes.append(selected_index)
        print(f"{spec.folder}: copied {len(selected_index)} images", flush=True)

    combined = pd.concat(all_indexes, ignore_index=True) if all_indexes else pd.DataFrame()
    combined.to_csv(out_dir / "all_selected_images.csv", index=False)
    skipped_frame = pd.DataFrame(skipped)
    skipped_frame.to_csv(out_dir / "skipped_features.csv", index=False)
    manifest = {
        "created_at": utc_now_iso(),
        "source_csv": str(source),
        "output_dir": str(out_dir),
        "top_n": int(args.top_n),
        "rows_in_source": int(len(pd.read_csv(source, usecols=["LocalPrimaryImagePath"], low_memory=False))),
        "rows_with_existing_primary_images": int(len(df)),
        "feature_folders_written": int(len(all_indexes)),
        "selected_images_total_with_repeats": int(len(combined)),
        "skipped_features": skipped,
        "derived_features": [
            "DerivedPrimaryShortSide",
            "DerivedPrimaryAreaMegapixels",
            "DerivedPrimaryEntropy",
            "DerivedPrimaryDarkPixelFraction",
            "DerivedPrimaryBrightPixelFraction",
            "DerivedPrimaryClippedPixelFraction",
            "DerivedPrimaryAspectRatioDeviation",
            "DerivedPrimaryColorfulness",
            "DerivedPrimaryNearBlankScore",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_readme(out_dir, manifest)
    print(f"Wrote selected bad image review set to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
