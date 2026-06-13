#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder import model_sweep as base_sweep
from experiments.full_scrape_model import compare_feature_modalities as cfm
from experiments.full_scrape_model.paths import OFFLINE_RUNS_DIR, assert_experiment_path, run_id, write_json, write_manifest


DEFAULT_MODALITY_RUN = "sold_status_feature_modalities_20260515_full_visual"
DEFAULT_SHAP_RUN = "no_dino_20260515_232153"
ABLATION_COLUMNS = ("SearchCount", "Page")
UPLOAD_PATTERNS = ("Upload_date", "upload_date")


def parse_upload_age_days(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan
    if text in {"oggi", "today"}:
        return 0.0
    if text in {"ieri", "yesterday"}:
        return 1.0
    number_match = re.search(r"(\d+(?:[.,]\d+)?)", text)
    amount = float(number_match.group(1).replace(",", ".")) if number_match else 1.0
    if any(token in text for token in ("minuto", "minuti", "minute", "minutes")):
        return amount / 1440.0
    if any(token in text for token in ("ora", "ore", "hour", "hours")):
        return amount / 24.0
    if any(token in text for token in ("giorno", "giorni", "day", "days")):
        return amount
    if any(token in text for token in ("settimana", "settimane", "week", "weeks")):
        return amount * 7.0
    if any(token in text for token in ("mese", "mesi", "month", "months")):
        return amount * 30.0
    if any(token in text for token in ("anno", "anni", "year", "years")):
        return amount * 365.0
    return np.nan


def upload_age_bucket(days: object) -> str:
    value = pd.to_numeric(pd.Series([days]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "unknown"
    if value < 1:
        return "<1d"
    if value < 2:
        return "1d"
    if value < 7:
        return "2-6d"
    if value < 14:
        return "1-2w"
    if value < 30:
        return "2-4w"
    if value < 90:
        return "1-3m"
    if value < 365:
        return "3-12m"
    return "1y+"


def run_no_search_page_ablation(args: argparse.Namespace, out_dir: Path) -> Path:
    ablation_dir = assert_experiment_path(out_dir / "no_search_count_page")
    original_numeric = list(cfm.FULL_NUMERIC)
    original_modes = tuple(cfm.FEATURE_MODES)
    original_argv = list(sys.argv)
    try:
        cfm.FULL_NUMERIC = [feature for feature in cfm.FULL_NUMERIC if feature not in ABLATION_COLUMNS]
        cfm.FEATURE_MODES = ("full_scrape", "full_scrape_plus_visual")
        argv = [
            "compare_feature_modalities.py",
            "--out-dir",
            str(ablation_dir),
            "--visual-run",
            args.visual_run,
            "--seed",
            str(args.seed),
        ]
        if args.all_searches:
            argv.append("--all-searches")
        for search in args.search:
            argv.extend(["--search", search])
        for approach in args.approach:
            argv.extend(["--approach", approach])
        if args.limit_rows:
            argv.extend(["--limit-rows", str(args.limit_rows)])
        if args.include_excluded_searches:
            argv.append("--include-excluded-searches")
        if not args.include_dino_embedding:
            argv.append("--no-include-dino-embedding")
        if args.max_dino_dims is not None:
            argv.extend(["--max-dino-dims", str(args.max_dino_dims)])
        sys.argv = argv
        cfm.main()
    finally:
        cfm.FULL_NUMERIC = original_numeric
        cfm.FEATURE_MODES = original_modes
        sys.argv = original_argv
    return ablation_dir


def read_best(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rename = {}
    keep = ["search_name", "feature_mode", "approach", "threshold", "validation_precision", "validation_count", "test_precision", "test_count", "test_precision_at_10", "test_roc_auc", "test_pr_auc"]
    df = df[[col for col in keep if col in df.columns]].copy()
    for col in df.columns:
        if col not in {"search_name", "feature_mode"}:
            rename[col] = f"{label}_{col}"
    return df.rename(columns=rename)


def build_ablation_comparison(original_run_dir: Path, ablation_dir: Path, out_dir: Path) -> pd.DataFrame:
    original = read_best(original_run_dir / "best_by_search_mode.csv", label="original")
    ablated = read_best(ablation_dir / "best_by_search_mode.csv", label="ablated")
    if original.empty or ablated.empty:
        return pd.DataFrame()
    comparison = original.merge(ablated, on=["search_name", "feature_mode"], how="inner")
    for metric in ("test_precision", "test_precision_at_10", "test_roc_auc", "test_pr_auc", "test_count"):
        left = f"original_{metric}"
        right = f"ablated_{metric}"
        if left in comparison.columns and right in comparison.columns:
            comparison[f"delta_{metric}"] = pd.to_numeric(comparison[right], errors="coerce") - pd.to_numeric(comparison[left], errors="coerce")
    path = assert_experiment_path(out_dir / "ablation_vs_original.csv")
    comparison.to_csv(path, index=False)
    return comparison


def is_upload_feature(row: pd.Series) -> bool:
    values = [
        str(row.get("display_feature", "")),
        str(row.get("original_feature", "")),
        str(row.get("transformed_feature", "")),
    ]
    return any(any(pattern.lower() in value.lower() for pattern in UPLOAD_PATTERNS) for value in values)


def build_upload_shap_summary(original_run_dir: Path, shap_run: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    shap_path = original_run_dir / "shap_analysis" / shap_run / "shap_feature_importance_long.csv"
    if not shap_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    shap_df = pd.read_csv(shap_path)
    shap_df["is_upload_feature"] = shap_df.apply(is_upload_feature, axis=1)
    totals = (
        shap_df.groupby(["search_name", "feature_mode"], as_index=False)
        .agg(total_mean_abs_shap=("mean_abs_shap", "sum"))
    )
    upload = shap_df[shap_df["is_upload_feature"]].copy()
    grouped = (
        upload.groupby(["search_name", "feature_mode"], as_index=False)
        .agg(
            upload_mean_abs_shap=("mean_abs_shap", "sum"),
            upload_mean_shap=("mean_shap", "sum"),
            upload_feature_count=("transformed_feature", "count"),
        )
        if not upload.empty
        else pd.DataFrame(columns=["search_name", "feature_mode", "upload_mean_abs_shap", "upload_mean_shap", "upload_feature_count"])
    )
    summary = totals.merge(grouped, on=["search_name", "feature_mode"], how="left")
    for col in ("upload_mean_abs_shap", "upload_mean_shap", "upload_feature_count"):
        summary[col] = summary[col].fillna(0)
    summary["upload_shap_share"] = summary["upload_mean_abs_shap"] / summary["total_mean_abs_shap"].replace(0, np.nan)
    summary_path = assert_experiment_path(out_dir / "upload_date_shap_summary.csv")
    detail_path = assert_experiment_path(out_dir / "upload_date_shap_features.csv")
    summary.to_csv(summary_path, index=False)
    upload.sort_values(["search_name", "feature_mode", "mean_abs_shap"], ascending=[True, True, False]).to_csv(detail_path, index=False)
    return summary, upload


def build_upload_distribution(original_run_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    phrase_rows = []
    dataset_dir = original_run_dir / "datasets"
    for path in sorted(dataset_dir.glob("*.csv")):
        search = path.stem
        df = pd.read_csv(path, low_memory=False)
        if "Upload_date" not in df.columns or base_sweep.TARGET_COL not in df.columns:
            continue
        age = df["Upload_date"].map(parse_upload_age_days)
        work = df[["Upload_date", base_sweep.TARGET_COL]].copy()
        work["upload_age_days_parsed"] = age
        work["upload_age_bucket"] = age.map(upload_age_bucket)
        work[base_sweep.TARGET_COL] = pd.to_numeric(work[base_sweep.TARGET_COL], errors="coerce").fillna(0).astype(int)
        if "Price" in df.columns:
            work["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", ".", regex=False).str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False), errors="coerce")
        if "Likes" in df.columns:
            work["Likes"] = pd.to_numeric(df["Likes"], errors="coerce")
        for bucket, part in work.groupby("upload_age_bucket", sort=False):
            rows.append(
                {
                    "search_name": search,
                    "upload_age_bucket": bucket,
                    "rows": int(len(part)),
                    "positive_rate": float(part[base_sweep.TARGET_COL].mean()) if len(part) else np.nan,
                    "sold_rows": int(part[base_sweep.TARGET_COL].sum()),
                    "avg_price": float(part["Price"].mean()) if "Price" in part else np.nan,
                    "avg_likes": float(part["Likes"].mean()) if "Likes" in part else np.nan,
                    "median_upload_age_days": float(part["upload_age_days_parsed"].median()) if part["upload_age_days_parsed"].notna().any() else np.nan,
                }
            )
        for phrase, part in work.groupby("Upload_date", dropna=False):
            phrase_rows.append(
                {
                    "search_name": search,
                    "upload_date_phrase": phrase,
                    "upload_age_days_parsed": float(part["upload_age_days_parsed"].median()) if part["upload_age_days_parsed"].notna().any() else np.nan,
                    "rows": int(len(part)),
                    "positive_rate": float(part[base_sweep.TARGET_COL].mean()) if len(part) else np.nan,
                    "sold_rows": int(part[base_sweep.TARGET_COL].sum()),
                }
            )
    bucket_df = pd.DataFrame(rows)
    phrase_df = pd.DataFrame(phrase_rows)
    if not bucket_df.empty:
        order = {"<1d": 0, "1d": 1, "2-6d": 2, "1-2w": 3, "2-4w": 4, "1-3m": 5, "3-12m": 6, "1y+": 7, "unknown": 8}
        bucket_df["bucket_order"] = bucket_df["upload_age_bucket"].map(order).fillna(99).astype(int)
        bucket_df = bucket_df.sort_values(["search_name", "bucket_order"]).drop(columns=["bucket_order"])
    if not phrase_df.empty:
        phrase_df = phrase_df.sort_values(["search_name", "rows"], ascending=[True, False])
    bucket_path = assert_experiment_path(out_dir / "upload_date_bucket_distribution.csv")
    phrase_path = assert_experiment_path(out_dir / "upload_date_phrase_distribution.csv")
    bucket_df.to_csv(bucket_path, index=False)
    phrase_df.to_csv(phrase_path, index=False)
    return bucket_df, phrase_df


def write_report(
    out_dir: Path,
    original_run_dir: Path,
    ablation_dir: Path,
    comparison: pd.DataFrame,
    upload_shap: pd.DataFrame,
    upload_distribution: pd.DataFrame,
) -> Path:
    path = assert_experiment_path(out_dir / "ablation_upload_date_report.md")
    lines = [
        "# SearchCount/Page Ablation And Upload-Date Analysis",
        "",
        f"Original modality run: `{original_run_dir}`",
        f"Ablation run: `{ablation_dir}`",
        "",
        "The ablation retrains `full_scrape` and `full_scrape_plus_visual` after removing `SearchCount` and `Page`.",
        "`basic_5` is not retrained because it does not use those fields in this framework.",
        "",
    ]
    if not comparison.empty:
        lines.extend([
            "## Ablation Result",
            "",
            "| search | mode | original approach | ablated approach | original test precision | ablated test precision | delta precision | original PR AUC | ablated PR AUC | delta PR AUC |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for _, row in comparison.sort_values(["search_name", "feature_mode"]).iterrows():
            lines.append(
                f"| {row.get('search_name')} | {row.get('feature_mode')} | {row.get('original_approach')} | {row.get('ablated_approach')} | "
                f"{float(row.get('original_test_precision', np.nan)):.3f} | {float(row.get('ablated_test_precision', np.nan)):.3f} | "
                f"{float(row.get('delta_test_precision', np.nan)):+.3f} | {float(row.get('original_test_pr_auc', np.nan)):.3f} | "
                f"{float(row.get('ablated_test_pr_auc', np.nan)):.3f} | {float(row.get('delta_test_pr_auc', np.nan)):+.3f} |"
            )
    if not upload_shap.empty:
        lines.extend(["", "## Upload-Date SHAP Share", ""])
        lines.extend(["| search | mode | upload SHAP share | upload mean abs SHAP | upload direction |", "| --- | --- | ---: | ---: | ---: |"])
        top = upload_shap.sort_values("upload_shap_share", ascending=False)
        for _, row in top.iterrows():
            lines.append(
                f"| {row.get('search_name')} | {row.get('feature_mode')} | "
                f"{float(row.get('upload_shap_share', 0.0)):.3f} | {float(row.get('upload_mean_abs_shap', 0.0)):.4f} | "
                f"{float(row.get('upload_mean_shap', 0.0)):.4f} |"
            )
    if not upload_distribution.empty:
        lines.extend(["", "## Upload-Date Label Distribution", ""])
        lines.append("Rows are grouped by parsed upload-age bucket. This helps check whether upload age is a genuine signal or a collection artifact.")
        lines.extend(["", "| search | bucket | rows | positive rate | avg price | avg likes |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for _, row in upload_distribution.sort_values(["search_name", "upload_age_bucket"]).iterrows():
            if int(row.get("rows", 0)) < 20:
                continue
            lines.append(
                f"| {row.get('search_name')} | {row.get('upload_age_bucket')} | {int(row.get('rows', 0))} | "
                f"{float(row.get('positive_rate', np.nan)):.3f} | {float(row.get('avg_price', np.nan)):.2f} | "
                f"{float(row.get('avg_likes', np.nan)):.2f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- If performance drops strongly after removing `SearchCount` and `Page`, those fields were carrying a large part of the offline signal.",
            "- Upload-date features can be useful, but they are risky if the timestamp is from full-scrape/backfill time rather than first observation time.",
            "- For live usage, prefer upload-age computed at first-page collection time, not a later enrichment timestamp.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SearchCount/Page ablation and upload-date analysis for full-scrape models.")
    parser.add_argument("--original-run", default=DEFAULT_MODALITY_RUN)
    parser.add_argument("--shap-run", default=DEFAULT_SHAP_RUN)
    parser.add_argument("--visual-run", default=cfm.DEFAULT_VISUAL_RUN)
    parser.add_argument("--all-searches", action="store_true")
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--include-excluded-searches", action="store_true")
    parser.add_argument("--approach", action="append", default=[])
    parser.add_argument("--seed", type=int, default=base_sweep.DEFAULT_SEED)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--include-dino-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-dino-dims", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.all_searches and not args.search:
        raise SystemExit("Use --all-searches or at least one --search.")
    original_run_dir = OFFLINE_RUNS_DIR / args.original_run
    if not original_run_dir.exists():
        raise FileNotFoundError(f"Original run folder not found: {original_run_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else original_run_dir / "ablation_analysis" / run_id("no_search_count_page")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_dir = run_no_search_page_ablation(args, out_dir)
    comparison = build_ablation_comparison(original_run_dir, ablation_dir, out_dir)
    upload_shap, _upload_detail = build_upload_shap_summary(original_run_dir, args.shap_run, out_dir)
    upload_distribution, phrase_distribution = build_upload_distribution(original_run_dir, out_dir)
    report_path = write_report(out_dir, original_run_dir, ablation_dir, comparison, upload_shap, upload_distribution)

    manifest = {
        "original_run": str(original_run_dir),
        "ablation_dir": str(ablation_dir),
        "removed_features": list(ABLATION_COLUMNS),
        "shap_run": args.shap_run,
        "outputs": {
            "report": str(report_path),
            "ablation_vs_original": str(out_dir / "ablation_vs_original.csv"),
            "upload_date_shap_summary": str(out_dir / "upload_date_shap_summary.csv"),
            "upload_date_shap_features": str(out_dir / "upload_date_shap_features.csv"),
            "upload_date_bucket_distribution": str(out_dir / "upload_date_bucket_distribution.csv"),
            "upload_date_phrase_distribution": str(out_dir / "upload_date_phrase_distribution.csv"),
        },
        "rows": {
            "comparison": int(len(comparison)),
            "upload_shap": int(len(upload_shap)),
            "upload_distribution": int(len(upload_distribution)),
            "upload_phrase_distribution": int(len(phrase_distribution)),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    write_manifest(out_dir / "run_manifest.json", command="full_scrape_model ablation_upload_date_analysis", extra=manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
