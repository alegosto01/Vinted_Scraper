#!/usr/bin/env python3
"""
Evaluate how good the final buy/not-buy decisions were.

This script reads a CSV that already has BuyDecisionScore/WorthBuying,
labels rows with sold and sold_eventually outcomes, and reports precision,
recall, confusion-matrix, and profit-oriented summary metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / 'scripts'
ANALYSIS_DIR = SCRIPTS_DIR / 'analysis_pipeline'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from analysis_pipeline.evaluation.evaluate_deal_score import (
    best_sold_rate_row,
    dedupe_listings,
    ensure_dir,
    normalize_id_series,
    precision_above_thresholds,
    precision_at_k,
    safe_numeric_series,
)


DEFAULT_TOPK = [5, 10, 20, 30, 50, 100]
DEFAULT_SCORE_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Evaluate final buy/not-buy decisions against sold and sold_eventually labels.')
    ap.add_argument('--input', required=True, help='Path to buy_candidates_enriched.csv or another CSV with buy decisions')
    ap.add_argument('--sold', required=True, help='Path to sold_df.csv')
    ap.add_argument('--sold_eventually', default=None, help='Optional path to sold_eventually.csv')
    ap.add_argument('--out_dir', required=True, help='Output directory')
    ap.add_argument('--id_col', default='Dataid')
    ap.add_argument('--buy_flag_col', default='WorthBuying')
    ap.add_argument('--buy_score_col', default='BuyDecisionScore')
    ap.add_argument('--expected_profit_col', default='ExpectedProfit')
    ap.add_argument('--price_col', default='Price')
    ap.add_argument('--topk', default=','.join(map(str, DEFAULT_TOPK)))
    ap.add_argument('--score_thresholds', default=','.join(map(str, DEFAULT_SCORE_THRESHOLDS)))
    ap.add_argument('--buy_score_threshold', type=float, default=0.62, help='Used only if the buy flag column is missing')
    ap.add_argument('--no_dedupe', action='store_true')
    return ap.parse_args()


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if str(series.dtype).lower() in {'bool', 'boolean'}:
        return series.fillna(False).astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin(['true', '1', 'yes', 'y'])


def add_sold_labels(df: pd.DataFrame, sold_df: pd.DataFrame, sold_eventually_df: pd.DataFrame, id_col: str, no_dedupe: bool) -> pd.DataFrame:
    deals = df.copy()
    sold = sold_df.copy()
    sold_eventually = sold_eventually_df.copy()

    if id_col not in deals.columns:
        raise ValueError(f'{id_col!r} not found in input file')
    if id_col not in sold.columns:
        raise ValueError(f'{id_col!r} not found in sold file')
    if not sold_eventually.empty and id_col not in sold_eventually.columns:
        raise ValueError(f'{id_col!r} not found in sold_eventually file')

    deals[id_col] = normalize_id_series(deals[id_col])
    sold[id_col] = normalize_id_series(sold[id_col])
    if not sold_eventually.empty:
        sold_eventually[id_col] = normalize_id_series(sold_eventually[id_col])

    if not no_dedupe:
        deals = dedupe_listings(deals, id_col)
        sold = dedupe_listings(sold, id_col)
        if not sold_eventually.empty:
            sold_eventually = dedupe_listings(sold_eventually, id_col)

    sold_ids = set(sold[id_col].dropna().unique().tolist())
    sold_eventually_ids = set(sold_eventually[id_col].dropna().unique().tolist()) if not sold_eventually.empty else set()
    combined_sold_ids = sold_ids | sold_eventually_ids

    deals['SoldImmediateLabel'] = deals[id_col].isin(sold_ids).astype(int)
    deals['SoldEventuallyLabel'] = deals[id_col].isin(sold_eventually_ids).astype(int)
    deals['SoldLabel'] = deals[id_col].isin(combined_sold_ids).astype(int)
    return deals


def ensure_buy_columns(df: pd.DataFrame, buy_flag_col: str, buy_score_col: str, buy_score_threshold: float) -> pd.DataFrame:
    out = df.copy()
    if buy_score_col in out.columns:
        out[buy_score_col] = safe_numeric_series(out[buy_score_col])
    if buy_flag_col in out.columns:
        out[buy_flag_col] = normalize_bool_series(out[buy_flag_col])
        return out
    if buy_score_col not in out.columns:
        raise ValueError(
            f'Neither {buy_flag_col!r} nor {buy_score_col!r} were found in the input file, so buy decisions cannot be evaluated.'
        )
    out[buy_flag_col] = out[buy_score_col] >= float(buy_score_threshold)
    return out


def safe_sum(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors='coerce').fillna(0.0).sum())


def build_buy_report(df: pd.DataFrame, buy_flag_col: str, expected_profit_col: str, price_col: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out[buy_flag_col] = normalize_bool_series(out[buy_flag_col])
    if expected_profit_col in out.columns:
        out[expected_profit_col] = safe_numeric_series(out[expected_profit_col])
    if price_col in out.columns:
        out[price_col] = safe_numeric_series(out[price_col])

    selected = out[out[buy_flag_col]].copy()
    rejected = out[~out[buy_flag_col]].copy()

    tp = int(((out[buy_flag_col]) & (out['SoldLabel'] == 1)).sum())
    fp = int(((out[buy_flag_col]) & (out['SoldLabel'] == 0)).sum())
    tn = int((~out[buy_flag_col] & (out['SoldLabel'] == 0)).sum())
    fn = int((~out[buy_flag_col] & (out['SoldLabel'] == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    accuracy = (tp + tn) / len(out) if len(out) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
    baseline_sold_rate = float(out['SoldLabel'].mean()) if len(out) else None
    lift_vs_baseline = (precision / baseline_sold_rate) if precision is not None and baseline_sold_rate not in {None, 0} else None

    selected_expected_profit = safe_sum(selected[expected_profit_col]) if expected_profit_col in selected.columns else 0.0
    selected_expected_profit_sold = safe_sum(selected.loc[selected['SoldLabel'] == 1, expected_profit_col]) if expected_profit_col in selected.columns else 0.0
    selected_expected_profit_unsold = safe_sum(selected.loc[selected['SoldLabel'] == 0, expected_profit_col]) if expected_profit_col in selected.columns else 0.0
    selected_cost_basis = safe_sum(selected[price_col]) if price_col in selected.columns else 0.0
    selected_expected_roi = (selected_expected_profit / selected_cost_basis) if selected_cost_basis > 0 else None

    summary_by_flag = pd.DataFrame([
        {
            'Decision': 'buy',
            'Count': int(len(selected)),
            'SoldCount': int(selected['SoldLabel'].sum()) if len(selected) else 0,
            'SoldRate': float(selected['SoldLabel'].mean()) if len(selected) else None,
            'ExpectedProfitSum': selected_expected_profit,
            'MeanExpectedProfit': float(selected[expected_profit_col].mean()) if expected_profit_col in selected.columns and len(selected) else None,
            'CapitalRequired': selected_cost_basis,
        },
        {
            'Decision': 'not_buy',
            'Count': int(len(rejected)),
            'SoldCount': int(rejected['SoldLabel'].sum()) if len(rejected) else 0,
            'SoldRate': float(rejected['SoldLabel'].mean()) if len(rejected) else None,
            'ExpectedProfitSum': safe_sum(rejected[expected_profit_col]) if expected_profit_col in rejected.columns else 0.0,
            'MeanExpectedProfit': float(rejected[expected_profit_col].mean()) if expected_profit_col in rejected.columns and len(rejected) else None,
            'CapitalRequired': safe_sum(rejected[price_col]) if price_col in rejected.columns else 0.0,
        },
    ])

    report = {
        'n_rows': int(len(out)),
        'n_buy': int(len(selected)),
        'n_not_buy': int(len(rejected)),
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'precision_buy': precision,
        'recall_buy': recall,
        'specificity_not_buy': specificity,
        'accuracy': accuracy,
        'f1_buy': f1,
        'baseline_sold_rate': baseline_sold_rate,
        'lift_vs_baseline': lift_vs_baseline,
        'selected_expected_profit_sum': selected_expected_profit,
        'selected_expected_profit_sum_sold_only': selected_expected_profit_sold,
        'selected_expected_profit_sum_unsold_only': selected_expected_profit_unsold,
        'selected_capital_required': selected_cost_basis,
        'selected_expected_roi': selected_expected_roi,
    }

    confusion = pd.DataFrame([
        {'Outcome': 'true_positive_buy_and_sold', 'Count': tp},
        {'Outcome': 'false_positive_buy_and_unsold', 'Count': fp},
        {'Outcome': 'true_negative_not_buy_and_unsold', 'Count': tn},
        {'Outcome': 'false_negative_not_buy_and_sold', 'Count': fn},
    ])
    return report, summary_by_flag, confusion


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    buys = pd.read_csv(args.input)
    sold = pd.read_csv(args.sold)
    sold_eventually = pd.read_csv(args.sold_eventually) if args.sold_eventually else pd.DataFrame(columns=[args.id_col])

    buys = add_sold_labels(buys, sold, sold_eventually, args.id_col, args.no_dedupe)
    buys = ensure_buy_columns(buys, args.buy_flag_col, args.buy_score_col, args.buy_score_threshold)
    labeled_path = out_dir / 'buy_candidates_labeled.csv'
    buys.to_csv(labeled_path, index=False)

    report, summary_by_flag, confusion = build_buy_report(
        buys,
        args.buy_flag_col,
        args.expected_profit_col,
        args.price_col,
    )
    summary_by_flag.to_csv(out_dir / 'summary_by_buy_flag.csv', index=False)
    confusion.to_csv(out_dir / 'buy_confusion_matrix.csv', index=False)

    if args.buy_score_col in buys.columns:
        scored = buys.dropna(subset=[args.buy_score_col]).copy()
        topk_values = [int(x) for x in str(args.topk).split(',') if str(x).strip()]
        topk_df = precision_at_k(scored, topk_values, args.buy_score_col)
        topk_df.to_csv(out_dir / 'precision_at_k_buy_score.csv', index=False)

        thresholds = [float(x) for x in str(args.score_thresholds).split(',') if str(x).strip()]
        threshold_df = precision_above_thresholds(scored, thresholds, args.buy_score_col)
        threshold_df = threshold_df.rename(columns={'ScoreThreshold': 'BuyScoreThreshold'})
        threshold_df.to_csv(out_dir / 'precision_by_buy_score_threshold.csv', index=False)
        report['best_precision_at_k'] = best_sold_rate_row(topk_df)
        report['best_precision_threshold'] = best_sold_rate_row(threshold_df)

        false_positives = scored[(scored[args.buy_flag_col]) & (scored['SoldLabel'] == 0)].sort_values(args.buy_score_col, ascending=False)
        false_positives.to_csv(out_dir / 'false_positive_buys_unsold.csv', index=False)
        false_negatives = scored[(~scored[args.buy_flag_col]) & (scored['SoldLabel'] == 1)].sort_values(args.buy_score_col, ascending=True)
        false_negatives.to_csv(out_dir / 'false_negative_skips_sold.csv', index=False)

    report['output_files'] = sorted(p.name for p in out_dir.iterdir())
    with open(out_dir / 'buy_evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f'Evaluated rows: {len(buys)}')
    if report['precision_buy'] is not None:
        print(f"Buy precision: {report['precision_buy']:.4f}")
    if report['lift_vs_baseline'] is not None:
        print(f"Lift vs baseline sold rate: {report['lift_vs_baseline']:.4f}")
    print(f"Selected expected profit sum: {report['selected_expected_profit_sum']:.2f}")
    print(f'Outputs in: {out_dir}')


if __name__ == '__main__':
    main()
