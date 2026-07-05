#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
for path in (SCRIPTS_DIR, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.old.clustering_approach.vinted_pipeline_batch import run as run_batch

sys.modules.setdefault("full_scraper", types.SimpleNamespace(Full_Scraper=object))
from analysis_pipeline.scoring.final_buy_filter import select_candidates


SEARCHES = ("ps4", "gucci", "prada")


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


TARGET_CONFIG = {
    "targeted_recall": dict(
        product_threshold=0.28,
        autotune_variants=True,
        variant_threshold=0.38,
        core_frac=0.70,
        variant_price_weight=0.35,
        min_product_size_for_variants=3,
        min_variant_size_for_deals=2,
        min_variant_size_for_confident_deals=4,
        min_variant_silhouette=0.20,
        max_variant_mad_ratio=0.35,
        hard_max_variant_mad_ratio=0.60,
        min_deal_confidence=0.45,
        min_centroid_similarity=0.45,
        min_informative_tokens=1,
    )
}


BUY_FILTER_ARGS = SimpleNamespace(
    require_deal_eligible=False,
    min_resale_safety=35.0,
    min_deal_confidence=0.60,
    min_expected_profit=8.0,
    min_expected_profit_margin=0.12,
    top_n=30,
    low_price_cutoff=25.0,
    low_price_min_expected_profit=3.0,
    low_price_profit_ratio=0.35,
    low_price_search_terms="ps4,ps5,switch,xbox,game,games",
)


def save_reports(out_root: Path, rows: list[dict], nested: dict[str, dict]) -> None:
    summary = pd.DataFrame(rows)
    summary_path = out_root / "summary.csv"
    report_path = out_root / "report.json"
    summary.to_csv(summary_path, index=False)
    report_path.write_text(json.dumps(nested, indent=2))


def make_batch_args(search: str, out_dir: Path, db_path: Path, overrides: dict) -> SimpleNamespace:
    base = dict(
        input=str(ROOT / "data" / "simple_scrape" / search / "big_raw.csv"),
        out_dir=str(out_dir),
        db=str(db_path),
        model="paraphrase-multilingual-MiniLM-L12-v2",
        resale_fee_rate=0.10,
        resale_fixed_cost=0.0,
        resale_safety_discount=0.05,
        min_expected_profit=0.0,
        min_expected_profit_margin=0.0,
        price_buffer_size=200,
        make_plots=False,
        exclude_negflag_deals=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def evaluate_outputs(search: str, out_dir: Path) -> dict:
    deals = pd.read_csv(out_dir / "deals_ranked.csv")
    items = pd.read_csv(out_dir / "items_with_product_and_variant.csv")
    sold = pd.read_csv(ROOT / "data" / "simple_scrape" / search / "sold_df.csv")
    sold_eventually_path = ROOT / "data" / "simple_scrape" / search / "eventual_sale_check" / "sold_eventually.csv"
    sold_eventually = pd.read_csv(sold_eventually_path) if sold_eventually_path.exists() else pd.DataFrame(columns=["Dataid"])

    for df in (deals, items, sold, sold_eventually):
        if "Dataid" in df.columns:
            df["Dataid"] = normalize_id_series(df["Dataid"])

    sold_ids = set(sold["Dataid"].dropna()) | set(sold_eventually["Dataid"].dropna())
    sold_in_items = int(items["Dataid"].isin(sold_ids).sum())
    sold_in_deals = int(deals["Dataid"].isin(sold_ids).sum())

    selected_buy = select_candidates(deals, BUY_FILTER_ARGS)
    sold_in_buy = int(selected_buy["Dataid"].isin(sold_ids).sum()) if "Dataid" in selected_buy.columns else 0

    top20 = deals.head(min(20, len(deals))).copy()
    top20_sold = int(top20["Dataid"].isin(sold_ids).sum()) if len(top20) else 0

    return {
        "items_rows": int(len(items)),
        "deals_rows": int(len(deals)),
        "sold_in_items": sold_in_items,
        "sold_in_deals": sold_in_deals,
        "deal_recall_from_items": (sold_in_deals / sold_in_items) if sold_in_items else None,
        "deal_precision": (sold_in_deals / len(deals)) if len(deals) else None,
        "top20_sold": top20_sold,
        "top20_precision": (top20_sold / len(top20)) if len(top20) else None,
        "buy_candidates_rows_adaptive": int(len(selected_buy)),
        "sold_in_buy_candidates_adaptive": sold_in_buy,
    }


def evaluate_existing_baseline(search: str) -> dict:
    return evaluate_outputs(search, ROOT / "data" / "simple_scrape" / search / "pipeline_out")


def main() -> None:
    out_root = ROOT / "data" / "simple_scrape" / "tuning_reports" / "upstream_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    nested: dict[str, dict] = {}

    for search in SEARCHES:
        nested[search] = {}
        baseline_metrics = evaluate_existing_baseline(search)
        nested[search]["existing_baseline"] = baseline_metrics
        rows.append({"Search": search, "Config": "existing_baseline", **baseline_metrics})
        save_reports(out_root, rows, nested)

        for config_name, overrides in TARGET_CONFIG.items():
            out_dir = out_root / search / config_name
            out_dir.mkdir(parents=True, exist_ok=True)
            db_path = out_dir / "index.sqlite"
            run_batch(make_batch_args(search, out_dir, db_path, overrides))
            metrics = evaluate_outputs(search, out_dir)
            nested[search][config_name] = metrics
            rows.append({"Search": search, "Config": config_name, **metrics})
            save_reports(out_root, rows, nested)

    summary_path = out_root / "summary.csv"
    report_path = out_root / "report.json"
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
