#!/usr/bin/env python3
"""
Refresh eventual-sold labels for an existing CSV of listings.

This script checks whether listings later sold and writes outputs like
sold_eventually.csv and a labeled copy of the input.
Use it to improve later evaluation with eventual-sale outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scraping_options import update_eventual_sale_labels_for_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check which listings eventually sold and export per-search CSVs.")
    ap.add_argument("--input", required=True, help="Input CSV to check, typically big_raw.csv or deals_ranked.csv")
    ap.add_argument("--out_dir", default=None, help="Output directory. Defaults to the input CSV folder")
    ap.add_argument("--api_token", default=None, help="Override API_TOKEN for rendered item-page fetches in this run")
    ap.add_argument("--max_workers", type=int, default=1)
    ap.add_argument("--delay", type=float, default=0.0, help="Delay between spawned status-check jobs")
    ap.add_argument("--allow_residential_fallback", action="store_true", help="Allow residential fallback after rendered fetch failures")
    ap.add_argument("--no_residential", action="store_true", help="Use only the datacenter proxy path and never the residential fallback")
    ap.add_argument("--initial_delay", type=float, default=0.0, help="Extra delay before each item-page check starts")
    ap.add_argument("--fetch_sleep", type=float, default=60.0, help="Sleep before each rendered item-page fetch attempt")
    ap.add_argument("--fetch_max_attempts", type=int, default=1, help="Max rendered fetch attempts per item-page check")
    ap.add_argument("--recheck_sold_rows", action="store_true", help="Also re-check rows already marked as Sold in the input CSV")
    ap.add_argument("--exclude_known_sold_csv", default=None, help="Optional CSV of already-sold listings to exclude from sold_eventually outputs")
    ap.add_argument("--min_deal_score", type=float, default=None, help="Only check rows with DealScore >= this value")
    ap.add_argument("--min_deal_confidence", type=float, default=None, help="Only check rows with DealConfidence >= this value")
    ap.add_argument("--top_n", type=int, default=None, help="Only check the top N rows after filtering/sorting")
    ap.add_argument("--require_deal_eligible", action="store_true", help="Only check rows where DealEligible is true")
    ap.add_argument("--sort_by", default="DealScore,DealConfidence,SearchCount", help="Comma-separated sort columns applied before top_n")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.api_token:
        os.environ["API_TOKEN"] = args.api_token
    result = update_eventual_sale_labels_for_csv(
        args.input,
        out_dir=args.out_dir,
        max_workers=args.max_workers,
        delay=args.delay,
        min_deal_score=args.min_deal_score,
        min_deal_confidence=args.min_deal_confidence,
        top_n=args.top_n,
        require_deal_eligible=args.require_deal_eligible,
        sort_by=args.sort_by,
        allow_residential_fallback=args.allow_residential_fallback,
        initial_delay=args.initial_delay,
        fetch_sleep=args.fetch_sleep,
        fetch_max_attempts=args.fetch_max_attempts,
        no_residential=args.no_residential,
        recheck_sold_rows=args.recheck_sold_rows,
        exclude_known_sold_csv=args.exclude_known_sold_csv,
    )
    if not os.getenv("API_TOKEN"):
        result["warning"] = (
            "API_TOKEN is not set for this process. Rendered-only eventual-sale checks will fail unless you pass "
            "--allow_residential_fallback or configure API_TOKEN."
        )
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.input).resolve().parent
    summary_path = out_dir / "eventual_sale_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({**result, "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
