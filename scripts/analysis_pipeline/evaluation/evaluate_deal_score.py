#!/usr/bin/env python3
"""
Evaluate how well deal-side scoring predicts sold outcomes.

This script works on deals_ranked.csv, adds sold/sold_eventually labels,
and reports score quality metrics such as precision@k, threshold precision,
and sold-vs-unsold score summaries.
It evaluates deal scoring, not the final buy decision layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_TOPK = [10, 20, 50, 100, 200, 500, 1000]
DEFAULT_SCORE_THRESHOLDS = [0.0, 1.0, 2.0, 3.0, 4.0]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", required=True, help="Path to deals_ranked.csv")
    ap.add_argument("--sold", required=True, help="Path to sold_df.csv or the primary sold-label CSV")
    ap.add_argument("--sold_eventually", default=None, help="Optional path to sold_eventually.csv. When provided, evaluation uses the union of immediate sold and eventual sold labels.")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--id_col", default="Dataid", help="Identifier column shared by both CSVs")
    ap.add_argument("--score_col", default="DealScore", help="Deal score column in deals CSV")
    ap.add_argument("--search_col", default="SearchName", help="Optional search/grouping column")
    ap.add_argument("--product_col", default="ProductId", help="Optional product column")
    ap.add_argument("--variant_col", default="VariantId", help="Optional variant column")
    ap.add_argument("--bins", type=int, default=10, help="Number of quantile bins")
    ap.add_argument("--topk", default=",".join(map(str, DEFAULT_TOPK)), help="Comma-separated list like 10,20,50,100")
    ap.add_argument("--score_thresholds", default=",".join(map(str, DEFAULT_SCORE_THRESHOLDS)), help="Comma-separated thresholds like 0,1,2,3")
    ap.add_argument("--high_score_threshold", type=float, default=2.0)
    ap.add_argument("--low_score_threshold", type=float, default=-1.0)
    ap.add_argument("--no_dedupe", action="store_true", help="Disable unique-listing deduplication before evaluation")
    return ap.parse_args()


def safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dedupe_listings(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if df.empty or id_col not in df.columns:
        return df.copy()
    out = df.copy()
    out["_row_order"] = range(len(out))
    sort_cols = []
    temp_cols = ["_row_order"]
    if "SearchCount" in out.columns:
        out["_SearchCountNum"] = pd.to_numeric(out["SearchCount"], errors="coerce")
        sort_cols.append("_SearchCountNum")
        temp_cols.append("_SearchCountNum")
    if "SearchDate" in out.columns:
        out["_SearchDateTs"] = pd.to_datetime(out["SearchDate"], errors="coerce", dayfirst=True)
        sort_cols.append("_SearchDateTs")
        temp_cols.append("_SearchDateTs")
    if "Page" in out.columns:
        out["_PageNum"] = pd.to_numeric(out["Page"], errors="coerce")
        sort_cols.append("_PageNum")
        temp_cols.append("_PageNum")
    sort_cols.append("_row_order")
    out = out.sort_values(sort_cols, kind="stable")
    out = out.drop_duplicates(subset=[id_col], keep="last")
    out = out.drop(columns=temp_cols, errors="ignore")
    return out.reset_index(drop=True)


def quantile_binning(values: pd.Series, bins: int) -> pd.Series:
    valid = values.notna()
    out = pd.Series([pd.NA] * len(values), index=values.index, dtype="object")
    n_unique = values[valid].nunique()
    if n_unique == 0:
        return out
    q = min(bins, n_unique)
    try:
        out.loc[valid] = pd.qcut(values.loc[valid], q=q, duplicates="drop")
    except ValueError:
        ranks = values.loc[valid].rank(method="first")
        out.loc[valid] = pd.qcut(ranks, q=min(q, ranks.nunique()), duplicates="drop")
    return out


def precision_at_k(df: pd.DataFrame, k_values: Iterable[int], score_col: str) -> pd.DataFrame:
    rows = []
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n_total = len(ranked)
    for k in k_values:
        kk = min(k, n_total)
        if kk <= 0:
            continue
        top = ranked.head(kk)
        sold_rate = float(top["SoldLabel"].mean()) if len(top) else np.nan
        rows.append({"TopK": kk, "SoldRate": sold_rate, "SoldCount": int(top["SoldLabel"].sum()), "Count": int(len(top))})
    return pd.DataFrame(rows)


def precision_above_thresholds(df: pd.DataFrame, thresholds: Iterable[float], score_col: str) -> pd.DataFrame:
    rows = []
    scored = df.dropna(subset=[score_col]).copy()
    for threshold in thresholds:
        subset = scored[scored[score_col] >= threshold]
        sold_rate = float(subset["SoldLabel"].mean()) if len(subset) else np.nan
        rows.append({
            "ScoreThreshold": float(threshold),
            "SoldRate": sold_rate,
            "SoldCount": int(subset["SoldLabel"].sum()) if len(subset) else 0,
            "Count": int(len(subset)),
        })
    return pd.DataFrame(rows)


def best_sold_rate_row(df: pd.DataFrame) -> dict:
    if df.empty or "SoldRate" not in df.columns:
        return {}
    valid = df.dropna(subset=["SoldRate"])
    if valid.empty:
        return {}
    return valid.loc[valid["SoldRate"].idxmax()].to_dict()


def correlation_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    tmp = df[[score_col, "SoldLabel"]].copy()
    tmp[score_col] = safe_numeric_series(tmp[score_col])
    tmp = tmp.dropna(subset=[score_col])
    pearson = tmp[score_col].corr(tmp["SoldLabel"], method="pearson") if len(tmp) >= 2 else np.nan
    if len(tmp) >= 2:
        rank_x = tmp[score_col].rank(method="average")
        rank_y = tmp["SoldLabel"].rank(method="average")
        spearman = rank_x.corr(rank_y, method="pearson")
    else:
        spearman = np.nan
    return pd.DataFrame([
        {"Metric": "pearson_corr", "Value": pearson},
        {"Metric": "spearman_corr", "Value": spearman},
        {"Metric": "n_scored_items", "Value": int(len(tmp))},
    ])


def sold_vs_unsold_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    g = df.groupby("SoldLabel", dropna=False)[score_col]
    summary = g.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    summary["SoldLabelName"] = summary["SoldLabel"].map({0: "unsold", 1: "sold"})
    return summary[["SoldLabel", "SoldLabelName", "count", "mean", "median", "std", "min", "max"]]


def grouped_summary(df: pd.DataFrame, by_col: str, score_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by_col):
        if pd.isna(key):
            continue
        row = {
            by_col: key,
            "Count": int(len(g)),
            "SoldRate": float(g["SoldLabel"].mean()),
            "MeanDealScore": float(g[score_col].mean()) if g[score_col].notna().any() else np.nan,
            "MedianDealScore": float(g[score_col].median()) if g[score_col].notna().any() else np.nan,
        }
        if "DealConfidence" in g.columns:
            row["MeanDealConfidence"] = float(g["DealConfidence"].mean()) if g["DealConfidence"].notna().any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Count", "SoldRate"], ascending=[False, False])
    return out


def save_bar_plot(data: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path, rotate: bool = True) -> None:
    if data.empty:
        return
    fig = plt.figure(figsize=(10, 5))
    x = np.arange(len(data))
    plt.bar(x, data[y_col].to_numpy())
    plt.xticks(x, data[x_col].astype(str).tolist(), rotation=45 if rotate else 0, ha="right" if rotate else "center")
    plt.title(title)
    plt.ylabel(y_col)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_hist_by_class(df: pd.DataFrame, score_col: str, out_path: Path) -> None:
    scored = df.dropna(subset=[score_col]).copy()
    if scored.empty:
        return
    fig = plt.figure(figsize=(9, 5))
    sold = scored.loc[scored["SoldLabel"] == 1, score_col].to_numpy()
    unsold = scored.loc[scored["SoldLabel"] == 0, score_col].to_numpy()
    bins = 30
    if len(unsold):
        plt.hist(unsold, bins=bins, alpha=0.6, label="Unsold")
    if len(sold):
        plt.hist(sold, bins=bins, alpha=0.6, label="Sold")
    plt.xlabel(score_col)
    plt.ylabel("Count")
    plt.title("DealScore distribution by sold label")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_precision_plot(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["TopK"], df["SoldRate"], marker="o")
    plt.xlabel("Top K highest DealScore items")
    plt.ylabel("Sold rate")
    plt.title("Precision@K")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    deals = pd.read_csv(args.deals)
    sold = pd.read_csv(args.sold)
    sold_eventually = pd.read_csv(args.sold_eventually) if args.sold_eventually else pd.DataFrame(columns=[args.id_col])
    input_rows = len(deals)

    if args.id_col not in deals.columns:
        raise ValueError(f"{args.id_col!r} not found in deals file")
    if args.id_col not in sold.columns:
        raise ValueError(f"{args.id_col!r} not found in sold file")
    if args.sold_eventually and args.id_col not in sold_eventually.columns:
        raise ValueError(f"{args.id_col!r} not found in sold_eventually file")
    if args.score_col not in deals.columns:
        raise ValueError(f"{args.score_col!r} not found in deals file")

    deals[args.id_col] = normalize_id_series(deals[args.id_col])
    sold[args.id_col] = normalize_id_series(sold[args.id_col])
    if args.sold_eventually and not sold_eventually.empty:
        sold_eventually[args.id_col] = normalize_id_series(sold_eventually[args.id_col])
    if not args.no_dedupe:
        deals = dedupe_listings(deals, args.id_col)
        sold = dedupe_listings(sold, args.id_col)
        if args.sold_eventually and not sold_eventually.empty:
            sold_eventually = dedupe_listings(sold_eventually, args.id_col)
    sold_ids = set(sold[args.id_col].dropna().unique().tolist())
    sold_eventually_ids = set(sold_eventually[args.id_col].dropna().unique().tolist()) if args.sold_eventually else set()
    combined_sold_ids = sold_ids | sold_eventually_ids

    deals[args.score_col] = safe_numeric_series(deals[args.score_col])
    deals["SoldImmediateLabel"] = deals[args.id_col].isin(sold_ids).astype(int)
    deals["SoldEventuallyLabel"] = deals[args.id_col].isin(sold_eventually_ids).astype(int)
    deals["SoldLabel"] = deals[args.id_col].isin(combined_sold_ids).astype(int)
    deals["SoldLabelName"] = deals["SoldLabel"].map({0: "unsold", 1: "sold"})

    labeled_path = out_dir / "deals_ranked_labeled.csv"
    deals.to_csv(labeled_path, index=False)

    overall = sold_vs_unsold_summary(deals, args.score_col)
    overall.to_csv(out_dir / "overall_sold_vs_unsold_summary.csv", index=False)

    corr = correlation_summary(deals, args.score_col)
    corr.to_csv(out_dir / "correlation_summary.csv", index=False)

    deals["ScoreBin"] = quantile_binning(deals[args.score_col], args.bins)
    binned = (
        deals.dropna(subset=["ScoreBin"])
        .groupby("ScoreBin", observed=False)
        .agg(
            Count=("SoldLabel", "size"),
            SoldCount=("SoldLabel", "sum"),
            SoldRate=("SoldLabel", "mean"),
            MeanDealScore=(args.score_col, "mean"),
            MedianDealScore=(args.score_col, "median"),
            MinDealScore=(args.score_col, "min"),
            MaxDealScore=(args.score_col, "max"),
        )
        .reset_index()
    )
    binned["ScoreBin"] = binned["ScoreBin"].astype(str)
    binned.to_csv(out_dir / "sell_rate_by_score_bin.csv", index=False)

    k_values = [int(x) for x in str(args.topk).split(",") if str(x).strip()]
    topk_df = precision_at_k(deals.dropna(subset=[args.score_col]), k_values, args.score_col)
    topk_df.to_csv(out_dir / "precision_at_k.csv", index=False)

    thresholds = [float(x) for x in str(args.score_thresholds).split(",") if str(x).strip()]
    threshold_df = precision_above_thresholds(deals, thresholds, args.score_col)
    threshold_df.to_csv(out_dir / "precision_by_score_threshold.csv", index=False)
    if "ExpectedProfitMargin" in deals.columns:
        margin_df = precision_above_thresholds(deals.rename(columns={"ExpectedProfitMargin": "_MetricThresholdCol"}), [0.0, 0.1, 0.2, 0.3, 0.5], "_MetricThresholdCol")
        margin_df = margin_df.rename(columns={"ScoreThreshold": "ExpectedProfitMarginThreshold"})
        margin_df.to_csv(out_dir / "precision_by_expected_profit_margin_threshold.csv", index=False)
    if "ExpectedProfit" in deals.columns:
        profit_df = precision_above_thresholds(deals.rename(columns={"ExpectedProfit": "_MetricThresholdCol"}), [0.0, 5.0, 10.0, 20.0, 50.0], "_MetricThresholdCol")
        profit_df = profit_df.rename(columns={"ScoreThreshold": "ExpectedProfitThreshold"})
        profit_df.to_csv(out_dir / "precision_by_expected_profit_threshold.csv", index=False)
    if "ResaleSafetyScore" in deals.columns:
        safety_df = precision_above_thresholds(deals.rename(columns={"ResaleSafetyScore": "_MetricThresholdCol"}), [20.0, 40.0, 60.0, 80.0], "_MetricThresholdCol")
        safety_df = safety_df.rename(columns={"ScoreThreshold": "ResaleSafetyScoreThreshold"})
        safety_df.to_csv(out_dir / "precision_by_resale_safety_threshold.csv", index=False)

    if args.search_col in deals.columns:
        grouped_summary(deals, args.search_col, args.score_col).to_csv(out_dir / "summary_by_search.csv", index=False)
    if args.product_col in deals.columns:
        grouped_summary(deals, args.product_col, args.score_col).to_csv(out_dir / "summary_by_product.csv", index=False)
    if args.variant_col in deals.columns:
        grouped_summary(deals, args.variant_col, args.score_col).to_csv(out_dir / "summary_by_variant.csv", index=False)

    if "DealConfidence" in deals.columns:
        deals["ConfidenceBin"] = quantile_binning(safe_numeric_series(deals["DealConfidence"]), min(args.bins, 5))
        conf_df = (
            deals.dropna(subset=["ConfidenceBin"])
            .groupby("ConfidenceBin", observed=False)
            .agg(Count=("SoldLabel", "size"), SoldRate=("SoldLabel", "mean"), MeanDealScore=(args.score_col, "mean"))
            .reset_index()
        )
        conf_df["ConfidenceBin"] = conf_df["ConfidenceBin"].astype(str)
        conf_df.to_csv(out_dir / "sell_rate_by_confidence_bin.csv", index=False)

    false_positives = deals[(deals[args.score_col] >= args.high_score_threshold) & (deals["SoldLabel"] == 0)].copy()
    false_positives = false_positives.sort_values(args.score_col, ascending=False)
    false_positives.to_csv(out_dir / "false_positives_high_score_unsold.csv", index=False)

    false_negatives = deals[(deals[args.score_col] <= args.low_score_threshold) & (deals["SoldLabel"] == 1)].copy()
    false_negatives = false_negatives.sort_values(args.score_col, ascending=True)
    false_negatives.to_csv(out_dir / "false_negatives_low_score_sold.csv", index=False)

    save_hist_by_class(deals, args.score_col, out_dir / "dealscore_distribution_sold_vs_unsold.png")
    save_bar_plot(binned, "ScoreBin", "SoldRate", "Sold rate by DealScore quantile bin", out_dir / "sell_rate_by_score_bin.png")
    save_precision_plot(topk_df, out_dir / "precision_at_k.png")

    best_topk = best_sold_rate_row(topk_df)
    best_threshold = best_sold_rate_row(threshold_df)
    report = {
        "n_input_deals_rows": int(input_rows),
        "n_deals_rows": int(len(deals)),
        "n_unique_sold_ids": int(len(sold_ids)),
        "n_unique_sold_eventually_ids": int(len(sold_eventually_ids)),
        "n_unique_combined_sold_ids": int(len(combined_sold_ids)),
        "n_rows_labeled_sold": int(deals["SoldLabel"].sum()),
        "n_rows_labeled_sold_immediate": int(deals["SoldImmediateLabel"].sum()),
        "n_rows_labeled_sold_eventually": int(deals["SoldEventuallyLabel"].sum()),
        "overall_sold_rate": float(deals["SoldLabel"].mean()) if len(deals) else None,
        "mean_score_sold": float(deals.loc[deals["SoldLabel"] == 1, args.score_col].mean()) if (deals["SoldLabel"] == 1).any() else None,
        "mean_score_unsold": float(deals.loc[deals["SoldLabel"] == 0, args.score_col].mean()) if (deals["SoldLabel"] == 0).any() else None,
        "best_precision_at_k": best_topk,
        "best_precision_threshold": best_threshold,
        "output_files": sorted([p.name for p in out_dir.iterdir()]),
    }
    with open(out_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Done.")
    print(f"Labeled file: {labeled_path}")
    if report["overall_sold_rate"] is not None:
        print(f"Overall sold rate: {report['overall_sold_rate']:.4f}")
    if report["mean_score_sold"] is not None and report["mean_score_unsold"] is not None:
        print(f"Mean DealScore sold:   {report['mean_score_sold']:.4f}")
        print(f"Mean DealScore unsold: {report['mean_score_unsold']:.4f}")
    if best_topk:
        print(f"Best Precision@K row: {best_topk}")
    if best_threshold:
        print(f"Best threshold row: {best_threshold}")


if __name__ == "__main__":
    main()
