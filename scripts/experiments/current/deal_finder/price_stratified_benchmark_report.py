#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.deal_finder.paths import REPORTS_DIR  # noqa: E402


DEFAULT_RUN_DIR = ROOT / "data" / "experiments" / "deal_finder" / "live_runs" / "hourly_all_models_benchmark_scheduled"
PRICE_BINS = [0, 25, 50, 100, 200, 500, math.inf]
PRICE_LABELS = ["<=25", "25-50", "50-100", "100-200", "200-500", ">500"]
HORIZONS = ("2h", "12h", "2d", "7d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a price-stratified performance report for saved all-model live benchmark results."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--tracked-file", default="tracked_model_threshold_items.csv")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--min-evaluated", type=int, default=5)
    parser.add_argument("--selected-only", action="store_true", default=True)
    return parser.parse_args()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def normalize_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "sold"})


def load_tracked(path: Path, *, selected_only: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Tracked benchmark file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if selected_only and "above_threshold" in df.columns:
        df = df[parse_bool_series(df["above_threshold"])].copy()
    df["PriceNum"] = pd.to_numeric(df.get("Price"), errors="coerce")
    df = df[df["PriceNum"].notna() & (df["PriceNum"] > 0)].copy()
    df["PriceBand"] = pd.cut(
        df["PriceNum"],
        bins=PRICE_BINS,
        labels=PRICE_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)
    df.loc[df["PriceBand"].eq("nan"), "PriceBand"] = "unknown"
    if "Dataid" in df.columns:
        item_ids = df["Dataid"].map(normalize_id)
    else:
        item_ids = pd.Series([""] * len(df), index=df.index)
    if "Link" in df.columns:
        item_ids = item_ids.where(item_ids.astype(str).str.len() > 0, df["Link"].fillna("").astype(str))
    df["_ItemKey"] = df["SearchName"].fillna("").astype(str) + "|" + item_ids.astype(str)
    for horizon in HORIZONS:
        sold_col = f"sold_within_{horizon}"
        eval_col = f"evaluated_{horizon}_at"
        if sold_col not in df.columns:
            df[sold_col] = False
        df[f"_{horizon}_sold"] = parse_bool_series(df[sold_col])
        evaluated = df[eval_col].notna() if eval_col in df.columns else pd.Series(False, index=df.index)
        df[f"_{horizon}_evaluated"] = evaluated | df[f"_{horizon}_sold"]
    return df


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return np.nan
    return float(numerator) / float(denominator)


def grouped_metrics(df: pd.DataFrame, group_cols: Iterable[str], *, min_evaluated: int) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(list(group_cols), dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "selected_rows": int(len(group)),
                "unique_items": int(group["_ItemKey"].nunique()),
                "price_min": float(group["PriceNum"].min()),
                "price_median": float(group["PriceNum"].median()),
                "price_mean": float(group["PriceNum"].mean()),
                "price_max": float(group["PriceNum"].max()),
            }
        )
        for horizon in HORIZONS:
            eval_count = int(group[f"_{horizon}_evaluated"].sum())
            sold_count = int((group[f"_{horizon}_evaluated"] & group[f"_{horizon}_sold"]).sum())
            precision = safe_divide(sold_count, eval_count)
            row[f"evaluated_{horizon}"] = eval_count
            row[f"sold_{horizon}"] = sold_count
            row[f"precision_{horizon}"] = precision
            row[f"enough_data_{horizon}"] = bool(eval_count >= min_evaluated)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [col for col in group_cols if col in out.columns]
    return out.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def unique_item_frame(df: pd.DataFrame) -> pd.DataFrame:
    agg: dict[str, object] = {
        "SearchName": "first",
        "PriceNum": "median",
        "PriceBand": "first",
        "Title": "first",
        "Link": "first",
        "Dataid": "first",
        "approach": lambda values: "|".join(sorted(set(str(v) for v in values if str(v) and str(v) != "nan"))),
        "threshold_label": lambda values: "|".join(sorted(set(str(v) for v in values if str(v) and str(v) != "nan"))),
        "model_probability": "max",
    }
    for horizon in HORIZONS:
        agg[f"_{horizon}_evaluated"] = "max"
        agg[f"_{horizon}_sold"] = "max"
    out = df.groupby("_ItemKey", as_index=False).agg(agg)
    for horizon in HORIZONS:
        out[f"_{horizon}_evaluated"] = out[f"_{horizon}_evaluated"].astype(bool)
        out[f"_{horizon}_sold"] = out[f"_{horizon}_sold"].astype(bool)
    return out


def top_slice(df: pd.DataFrame, metric: str, *, min_evaluated_col: str, n: int = 12) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return df.iloc[0:0].copy()
    eligible = df[df[min_evaluated_col]].copy() if min_evaluated_col in df.columns else df.copy()
    if eligible.empty:
        return eligible
    return eligible.sort_values([metric, "evaluated_2d", "selected_rows"], ascending=[False, False, False]).head(n)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return ""
    table = frame.loc[:, [col for col in columns if col in frame.columns]].copy()
    rendered_rows: list[list[str]] = []
    for _, row in table.iterrows():
        rendered = []
        for value in row.tolist():
            if isinstance(value, float):
                rendered.append("" if np.isnan(value) else f"{value:.3f}")
            else:
                rendered.append(str(value))
        rendered_rows.append(rendered)
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered_rows)
    return "\n".join(lines)


def write_report_markdown(
    path: Path,
    *,
    tracked_path: Path,
    df: pd.DataFrame,
    model_band: pd.DataFrame,
    search_band: pd.DataFrame,
    item_band: pd.DataFrame,
    min_evaluated: int,
) -> None:
    price_distribution = grouped_metrics(df, ["SearchName", "PriceBand"], min_evaluated=min_evaluated)
    ps4_rows = search_band[search_band["SearchName"].astype(str).eq("ps4")].copy()
    best_model_2d = top_slice(model_band, "precision_2d", min_evaluated_col="enough_data_2d", n=15)
    best_search_2d = top_slice(search_band, "precision_2d", min_evaluated_col="enough_data_2d", n=15)

    lines = [
        "# Price-Stratified All-Model Benchmark Report",
        "",
        f"Generated at `{datetime.now().astimezone().isoformat(timespec='seconds')}`.",
        "",
        f"Input: `{tracked_path}`",
        "",
        "## Scope",
        "",
        f"- Selected model-threshold rows: `{len(df):,}`",
        f"- Unique selected items: `{df['_ItemKey'].nunique():,}`",
        f"- Searches: `{', '.join(sorted(df['SearchName'].dropna().astype(str).unique()))}`",
        f"- Price bands: `{', '.join(PRICE_LABELS)}`",
        f"- Minimum evaluated rows highlighted as reliable: `{min_evaluated}`",
        "",
        "## Important Reading Notes",
        "",
        "- `precision_2h`, `precision_12h`, and `precision_2d` mean sold within that horizon divided by evaluated selected rows in that band.",
        "- Rows are model-threshold selections, so the same item can appear once per model and threshold label.",
        "- The unique-item CSV removes most of that duplication and is better for understanding buyer behavior by price.",
        "- The 7-day fields in this old run look incomplete/biased because evaluated 7-day rows are mostly already-sold rows; use 2h, 12h, and 2d first.",
        "",
        "## Best Model/Threshold/Price Slices By 2-Day Precision",
        "",
    ]
    if best_model_2d.empty:
        lines.append("_No model/price slice has enough 2-day evaluated rows yet._")
    else:
        cols = [
            "SearchName",
            "approach",
            "threshold_label",
            "PriceBand",
            "selected_rows",
            "unique_items",
            "evaluated_2d",
            "sold_2d",
            "precision_2d",
            "precision_12h",
            "price_median",
        ]
        lines.append(markdown_table(best_model_2d, cols))
    lines.extend(["", "## Best Search/Price Slices By 2-Day Precision", ""])
    if best_search_2d.empty:
        lines.append("_No search/price slice has enough 2-day evaluated rows yet._")
    else:
        cols = [
            "SearchName",
            "PriceBand",
            "selected_rows",
            "unique_items",
            "evaluated_2d",
            "sold_2d",
            "precision_2d",
            "precision_12h",
            "price_median",
        ]
        lines.append(markdown_table(best_search_2d, cols))
    lines.extend(["", "## PS4 By Price Band", ""])
    if ps4_rows.empty:
        lines.append("_No PS4 rows found._")
    else:
        cols = [
            "PriceBand",
            "selected_rows",
            "unique_items",
            "evaluated_2h",
            "sold_2h",
            "precision_2h",
            "evaluated_12h",
            "sold_12h",
            "precision_12h",
            "evaluated_2d",
            "sold_2d",
            "precision_2d",
            "price_median",
        ]
        lines.append(markdown_table(ps4_rows, cols))
    lines.extend(["", "## Price Distribution By Search", ""])
    if price_distribution.empty:
        lines.append("_No rows found._")
    else:
        cols = ["SearchName", "PriceBand", "selected_rows", "unique_items", "price_median", "evaluated_2d", "precision_2d"]
        lines.append(markdown_table(price_distribution, cols))
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `model_price_band_metrics.csv`: model + threshold + price-band performance.",
            "- `search_price_band_metrics.csv`: search + price-band performance across model selections.",
            "- `unique_item_price_band_metrics.csv`: deduplicated item-level price-band performance.",
            "- `price_band_manifest.json`: report metadata.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    tracked_path = run_dir / args.tracked_file
    out_dir = Path(args.out_dir) if args.out_dir else REPORTS_DIR / f"price_stratified_benchmark_{utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_tracked(tracked_path, selected_only=bool(args.selected_only))
    model_band = grouped_metrics(
        df,
        ["SearchName", "approach", "threshold_label", "PriceBand"],
        min_evaluated=int(args.min_evaluated),
    )
    search_band = grouped_metrics(df, ["SearchName", "PriceBand"], min_evaluated=int(args.min_evaluated))
    overall_band = grouped_metrics(df, ["PriceBand"], min_evaluated=int(args.min_evaluated))
    item_df = unique_item_frame(df)
    unique_item_band = grouped_metrics(item_df, ["SearchName", "PriceBand"], min_evaluated=int(args.min_evaluated))

    model_path = out_dir / "model_price_band_metrics.csv"
    search_path = out_dir / "search_price_band_metrics.csv"
    overall_path = out_dir / "overall_price_band_metrics.csv"
    item_path = out_dir / "unique_item_price_band_metrics.csv"
    item_rows_path = out_dir / "unique_item_rows.csv"
    report_path = out_dir / "price_stratified_report.md"
    manifest_path = out_dir / "price_band_manifest.json"

    model_band.to_csv(model_path, index=False)
    search_band.to_csv(search_path, index=False)
    overall_band.to_csv(overall_path, index=False)
    unique_item_band.to_csv(item_path, index=False)
    item_df.to_csv(item_rows_path, index=False)
    write_report_markdown(
        report_path,
        tracked_path=tracked_path,
        df=df,
        model_band=model_band,
        search_band=search_band,
        item_band=unique_item_band,
        min_evaluated=int(args.min_evaluated),
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "run_dir": str(run_dir),
        "tracked_path": str(tracked_path),
        "selected_rows": int(len(df)),
        "unique_items": int(df["_ItemKey"].nunique()),
        "price_bins": PRICE_BINS[:-1] + ["inf"],
        "price_labels": PRICE_LABELS,
        "min_evaluated": int(args.min_evaluated),
        "outputs": {
            "model_price_band_metrics": str(model_path),
            "search_price_band_metrics": str(search_path),
            "overall_price_band_metrics": str(overall_path),
            "unique_item_price_band_metrics": str(item_path),
            "unique_item_rows": str(item_rows_path),
            "report": str(report_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Wrote model metrics: {model_path}")
    print(f"Wrote search metrics: {search_path}")
    print(f"Wrote unique-item metrics: {item_path}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
