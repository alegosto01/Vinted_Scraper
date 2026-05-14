#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
for path in (SCRIPTS_DIR, ANALYSIS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

from analysis_pipeline.scoring.visual_rerank import compute_image_metrics, normalize_image_sources

DEFAULT_DATASETS = {
    "ps4": ROOT / "data/simple_scrape/ps4/image_cache/balanced_raw_eval/balanced_raw_with_local_images.csv",
    "gucci": ROOT / "data/simple_scrape/gucci/image_cache/balanced_raw_eval/balanced_raw_with_local_images.csv",
    "prada": ROOT / "data/simple_scrape/prada/image_cache/balanced_raw_eval/balanced_raw_with_local_images.csv",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Measure image blur/sharpness on local cached listing images and compare sold vs unsold rows. "
            "The underlying metric is the same blur_score used in visual_rerank.py, where higher means sharper."
        )
    )
    ap.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Input CSV with SoldLabel and LocalImagePaths/LocalPrimaryImagePath. Can be passed multiple times.",
    )
    ap.add_argument(
        "--out_dir",
        default=str(ROOT / "data/simple_scrape/tuning_reports/blur_signal"),
        help="Output directory for CSV and JSON reports.",
    )
    ap.add_argument(
        "--blur_threshold",
        type=float,
        default=25.0,
        help="Threshold below which an image is treated as blurry. Matches visual_rerank default.",
    )
    ap.add_argument(
        "--permutations",
        type=int,
        default=2000,
        help="Number of label permutations for the mean-difference p-value.",
    )
    ap.add_argument(
        "--bootstrap_samples",
        type=int,
        default=2000,
        help="Number of bootstrap samples for the mean-difference confidence interval.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for permutation/bootstrap calculations.",
    )
    ap.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Optional row cap per dataset for a faster dry run.",
    )
    return ap.parse_args()


def parse_dataset_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"Invalid --dataset value '{raw}'. Expected NAME=PATH.")
    name, path = raw.split("=", 1)
    return name.strip(), Path(path).expanduser().resolve()


def resolve_datasets(dataset_args: list[str]) -> list[tuple[str, Path]]:
    if not dataset_args:
        return [(name, path.resolve()) for name, path in DEFAULT_DATASETS.items()]
    resolved: list[tuple[str, Path]] = []
    for raw in dataset_args:
        name, path = parse_dataset_arg(raw)
        resolved.append((name, path))
    return resolved


def sample_std(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    return float(values.std(ddof=1))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    pooled_num = (a.size - 1) * var_a + (b.size - 1) * var_b
    pooled_den = a.size + b.size - 2
    if pooled_den <= 0:
        return None
    pooled_std = math.sqrt(pooled_num / pooled_den) if pooled_num > 0 else 0.0
    if pooled_std == 0:
        return None
    return float((a.mean() - b.mean()) / pooled_std)


def common_language_effect(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size == 0 or b.size == 0:
        return None
    diffs = a[:, None] - b[None, :]
    wins = float((diffs > 0).sum())
    ties = float((diffs == 0).sum())
    total = float(diffs.size)
    return (wins + 0.5 * ties) / total if total else None


def permutation_pvalue(a: np.ndarray, b: np.ndarray, permutations: int, seed: int) -> float | None:
    if a.size == 0 or b.size == 0:
        return None
    observed = abs(float(a.mean() - b.mean()))
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    n_a = a.size
    for _ in range(max(permutations, 1)):
        perm = rng.permutation(pooled)
        diff = abs(float(perm[:n_a].mean() - perm[n_a:].mean()))
        if diff >= observed - 1e-12:
            count += 1
    return float((count + 1) / (max(permutations, 1) + 1))


def bootstrap_mean_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    if a.size == 0 or b.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(max(samples, 1)):
        a_idx = rng.integers(0, a.size, size=a.size)
        b_idx = rng.integers(0, b.size, size=b.size)
        draws.append(float(a[a_idx].mean() - b[b_idx].mean()))
    low = float(np.quantile(draws, alpha / 2))
    high = float(np.quantile(draws, 1 - alpha / 2))
    return low, high


def describe(values: np.ndarray, blur_threshold: float) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "blurry_rate": None,
        }
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": sample_std(values),
        "min": float(values.min()),
        "max": float(values.max()),
        "blurry_rate": float((values < blur_threshold).mean()),
    }


def summarize_signal(
    sold: np.ndarray,
    unsold: np.ndarray,
    *,
    blur_threshold: float,
    permutations: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    sold_stats = describe(sold, blur_threshold)
    unsold_stats = describe(unsold, blur_threshold)
    mean_diff = None
    blurry_rate_diff = None
    if sold.size and unsold.size:
        mean_diff = float(sold.mean() - unsold.mean())
        blurry_rate_diff = float((sold < blur_threshold).mean() - (unsold < blur_threshold).mean())
    ci_low, ci_high = bootstrap_mean_diff_ci(sold, unsold, bootstrap_samples, seed)
    return {
        "sold_n": sold_stats["n"],
        "sold_mean": sold_stats["mean"],
        "sold_median": sold_stats["median"],
        "sold_std": sold_stats["std"],
        "sold_blurry_rate": sold_stats["blurry_rate"],
        "unsold_n": unsold_stats["n"],
        "unsold_mean": unsold_stats["mean"],
        "unsold_median": unsold_stats["median"],
        "unsold_std": unsold_stats["std"],
        "unsold_blurry_rate": unsold_stats["blurry_rate"],
        "mean_diff_sold_minus_unsold": mean_diff,
        "mean_diff_ci_low": ci_low,
        "mean_diff_ci_high": ci_high,
        "blurry_rate_diff_sold_minus_unsold": blurry_rate_diff,
        "cohens_d": cohens_d(sold, unsold),
        "common_language_effect": common_language_effect(sold, unsold),
        "permutation_pvalue": permutation_pvalue(sold, unsold, permutations, seed),
    }


def get_row_image_sources(row: pd.Series) -> list[str]:
    local_paths = normalize_image_sources(row.get("LocalImagePaths"))
    if local_paths:
        return local_paths
    primary = row.get("LocalPrimaryImagePath")
    if isinstance(primary, str) and primary.strip():
        return [primary.strip()]
    return normalize_image_sources(row.get("Images"))


def analyze_dataset(
    name: str,
    path: Path,
    *,
    blur_threshold: float,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(path)
    if max_rows > 0:
        df = df.head(max_rows).copy()

    image_cache: dict[str, float] = {}
    listing_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    counters = {
        "dataset_rows": int(len(df)),
        "rows_with_images": 0,
        "rows_scored": 0,
        "image_count": 0,
        "image_failures": 0,
    }

    for _, row in df.iterrows():
        sources = get_row_image_sources(row)
        if sources:
            counters["rows_with_images"] += 1
        scores: list[float] = []
        for source in sources:
            key = str(source)
            if key not in image_cache:
                try:
                    image_cache[key] = float(compute_image_metrics(key).blur_score)
                except Exception:
                    counters["image_failures"] += 1
                    continue
            score = image_cache[key]
            scores.append(score)
            counters["image_count"] += 1
            image_rows.append(
                {
                    "Dataset": name,
                    "Dataid": row.get("Dataid"),
                    "SoldLabel": int(row.get("SoldLabel", 0)),
                    "ImagePath": key,
                    "SharpnessScore": score,
                    "BlurryFlag": int(score < blur_threshold),
                }
            )

        if not scores:
            continue

        counters["rows_scored"] += 1
        arr = np.asarray(scores, dtype=float)
        listing_rows.append(
            {
                "Dataset": name,
                "Dataid": row.get("Dataid"),
                "SoldLabel": int(row.get("SoldLabel", 0)),
                "Title": row.get("Title"),
                "SearchName": row.get("SearchName", name),
                "ImageCountMeasured": int(arr.size),
                "PrimarySharpnessScore": float(arr[0]),
                "MeanSharpnessScore": float(arr.mean()),
                "MedianSharpnessScore": float(np.median(arr)),
                "MinSharpnessScore": float(arr.min()),
                "MaxSharpnessScore": float(arr.max()),
                "PrimaryBlurryFlag": int(arr[0] < blur_threshold),
                "AnyBlurryFlag": int((arr < blur_threshold).any()),
            }
        )

    return pd.DataFrame(listing_rows), pd.DataFrame(image_rows), counters


def compute_summary_rows(
    listing_df: pd.DataFrame,
    image_df: pd.DataFrame,
    *,
    blur_threshold: float,
    permutations: int,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_summary(level: str, dataset: str, sold: np.ndarray, unsold: np.ndarray) -> None:
        summary = summarize_signal(
            sold,
            unsold,
            blur_threshold=blur_threshold,
            permutations=permutations,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        summary.update({"Dataset": dataset, "Level": level})
        rows.append(summary)

    for dataset in sorted(listing_df["Dataset"].dropna().unique()):
        dataset_listing = listing_df[listing_df["Dataset"] == dataset]
        sold_primary = dataset_listing.loc[dataset_listing["SoldLabel"] == 1, "PrimarySharpnessScore"].to_numpy(dtype=float)
        unsold_primary = dataset_listing.loc[dataset_listing["SoldLabel"] == 0, "PrimarySharpnessScore"].to_numpy(dtype=float)
        sold_mean = dataset_listing.loc[dataset_listing["SoldLabel"] == 1, "MeanSharpnessScore"].to_numpy(dtype=float)
        unsold_mean = dataset_listing.loc[dataset_listing["SoldLabel"] == 0, "MeanSharpnessScore"].to_numpy(dtype=float)
        add_summary("listing_primary", dataset, sold_primary, unsold_primary)
        add_summary("listing_mean_images", dataset, sold_mean, unsold_mean)

        dataset_images = image_df[image_df["Dataset"] == dataset]
        sold_images = dataset_images.loc[dataset_images["SoldLabel"] == 1, "SharpnessScore"].to_numpy(dtype=float)
        unsold_images = dataset_images.loc[dataset_images["SoldLabel"] == 0, "SharpnessScore"].to_numpy(dtype=float)
        add_summary("all_images", dataset, sold_images, unsold_images)

    add_summary(
        "listing_primary",
        "combined",
        listing_df.loc[listing_df["SoldLabel"] == 1, "PrimarySharpnessScore"].to_numpy(dtype=float),
        listing_df.loc[listing_df["SoldLabel"] == 0, "PrimarySharpnessScore"].to_numpy(dtype=float),
    )
    add_summary(
        "listing_mean_images",
        "combined",
        listing_df.loc[listing_df["SoldLabel"] == 1, "MeanSharpnessScore"].to_numpy(dtype=float),
        listing_df.loc[listing_df["SoldLabel"] == 0, "MeanSharpnessScore"].to_numpy(dtype=float),
    )
    add_summary(
        "all_images",
        "combined",
        image_df.loc[image_df["SoldLabel"] == 1, "SharpnessScore"].to_numpy(dtype=float),
        image_df.loc[image_df["SoldLabel"] == 0, "SharpnessScore"].to_numpy(dtype=float),
    )

    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    listing_frames: list[pd.DataFrame] = []
    image_frames: list[pd.DataFrame] = []
    dataset_counters: dict[str, dict[str, int]] = {}

    for dataset_name, dataset_path in resolve_datasets(args.dataset):
        listing_df, image_df, counters = analyze_dataset(
            dataset_name,
            dataset_path,
            blur_threshold=args.blur_threshold,
            max_rows=args.max_rows,
        )
        listing_frames.append(listing_df)
        image_frames.append(image_df)
        dataset_counters[dataset_name] = counters

    listing_all = pd.concat(listing_frames, ignore_index=True) if listing_frames else pd.DataFrame()
    image_all = pd.concat(image_frames, ignore_index=True) if image_frames else pd.DataFrame()
    summary_rows = compute_summary_rows(
        listing_all,
        image_all,
        blur_threshold=args.blur_threshold,
        permutations=args.permutations,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    summary_df = pd.DataFrame(summary_rows)

    listing_path = out_dir / "blur_listing_metrics.csv"
    image_path = out_dir / "blur_image_metrics.csv"
    summary_path = out_dir / "blur_signal_summary.csv"
    report_path = out_dir / "blur_signal_report.json"

    listing_all.to_csv(listing_path, index=False)
    image_all.to_csv(image_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    report = {
        "metric_note": "Higher SharpnessScore means sharper image. Lower values mean blurrier image.",
        "blur_threshold": args.blur_threshold,
        "datasets": dataset_counters,
        "files": {
            "listing_metrics": str(listing_path),
            "image_metrics": str(image_path),
            "summary": str(summary_path),
        },
        "summary": summary_rows,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
