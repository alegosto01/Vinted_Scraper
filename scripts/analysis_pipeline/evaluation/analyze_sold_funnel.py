#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SEARCHES = ("ps4", "gucci", "prada")


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def safe_read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns or [])


def top_note_counts(series: pd.Series, top_n: int = 10) -> dict[str, int]:
    exploded = (
        series.fillna("")
        .astype(str)
        .str.split("|")
        .explode()
        .replace("", pd.NA)
        .dropna()
    )
    if exploded.empty:
        return {}
    return {str(k): int(v) for k, v in exploded.value_counts().head(top_n).items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze where sold items are dropped between big_raw, deals_ranked, and buy candidates.")
    ap.add_argument("--base_dir", default="data/simple_scrape")
    ap.add_argument("--out_dir", default="data/simple_scrape/tuning_reports")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    report: dict[str, dict] = {}

    for search in SEARCHES:
        folder = base_dir / search
        big_raw = safe_read_csv(folder / "big_raw.csv", ["Dataid"])
        items = safe_read_csv(folder / "pipeline_out" / "items_with_product_and_variant.csv", ["Dataid"])
        deals = safe_read_csv(folder / "pipeline_out" / "deals_ranked.csv", ["Dataid"])
        buy_candidates = safe_read_csv(folder / "buy_decision_eval" / "buy_candidates_input.csv", ["Dataid"])
        buy_enriched = safe_read_csv(folder / "buy_decision_eval" / "buy_candidates_enriched.csv", ["Dataid"])
        sold = safe_read_csv(folder / "sold_df.csv", ["Dataid"])
        sold_eventually = safe_read_csv(folder / "eventual_sale_check" / "sold_eventually.csv", ["Dataid"])

        for df in (big_raw, items, deals, buy_candidates, buy_enriched, sold, sold_eventually):
            if "Dataid" in df.columns:
                df["Dataid"] = normalize_id_series(df["Dataid"])

        sold_ids = set(sold["Dataid"].dropna()) | set(sold_eventually["Dataid"].dropna())
        big_raw_ids = set(big_raw["Dataid"].dropna())
        item_ids = set(items["Dataid"].dropna())
        deal_ids = set(deals["Dataid"].dropna())
        buy_candidate_ids = set(buy_candidates["Dataid"].dropna())
        worth_buying_ids = set(
            buy_enriched.loc[
                buy_enriched.get("WorthBuying", False).fillna(False).astype(bool),
                "Dataid",
            ].dropna()
        ) if "WorthBuying" in buy_enriched.columns else set()

        sold_in_big_raw = sold_ids & big_raw_ids
        sold_in_items = sold_ids & item_ids
        sold_in_deals = sold_ids & deal_ids
        sold_in_buy_candidates = sold_ids & buy_candidate_ids
        sold_in_worth_buying = sold_ids & worth_buying_ids

        dropped_before_deals = items[items["Dataid"].isin(sold_in_items - deal_ids)].copy()
        dropped_before_buy = deals[deals["Dataid"].isin(sold_in_deals - buy_candidate_ids)].copy()

        summary_rows.append(
            {
                "Search": search,
                "SoldGroundTruthTotal": int(len(sold_ids)),
                "SoldInBigRaw": int(len(sold_in_big_raw)),
                "SoldInItemsPipeline": int(len(sold_in_items)),
                "SoldInDealsRanked": int(len(sold_in_deals)),
                "SoldInBuyCandidates": int(len(sold_in_buy_candidates)),
                "SoldInWorthBuying": int(len(sold_in_worth_buying)),
                "Recall_BigRaw_to_Deals": (len(sold_in_deals) / len(sold_in_big_raw)) if sold_in_big_raw else None,
                "Recall_Deals_to_BuyCandidates": (len(sold_in_buy_candidates) / len(sold_in_deals)) if sold_in_deals else None,
                "Recall_BigRaw_to_WorthBuying": (len(sold_in_worth_buying) / len(sold_in_big_raw)) if sold_in_big_raw else None,
            }
        )

        report[search] = {
            "dropped_before_deals_count": int(len(dropped_before_deals)),
            "dropped_before_deals_top_notes": top_note_counts(dropped_before_deals.get("DealNotes", pd.Series(dtype=object))),
            "dropped_before_deals_example_ids": dropped_before_deals["Dataid"].head(10).tolist(),
            "dropped_before_buy_count": int(len(dropped_before_buy)),
            "dropped_before_buy_example_rows": dropped_before_buy[
                [c for c in [
                    "Dataid",
                    "Title",
                    "Price",
                    "ResaleSafetyScore",
                    "ExpectedProfit",
                    "ExpectedProfitMargin",
                    "DealScore",
                    "DealConfidence",
                    "DealEligible",
                ] if c in dropped_before_buy.columns]
            ].head(10).to_dict(orient="records"),
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "sold_funnel_summary.csv"
    report_path = out_dir / "sold_funnel_report.json"
    summary_df.to_csv(summary_path, index=False)
    report_path.write_text(json.dumps(report, indent=2))

    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
