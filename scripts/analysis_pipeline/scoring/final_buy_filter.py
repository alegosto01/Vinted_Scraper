#!/usr/bin/env python3
"""
Generate final buy/not-buy decisions for deal candidates.

This script enriches candidate listings with extra item and seller data,
then computes BuyDecisionScore and WorthBuying.
Use this before evaluate_buy_decisions.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from full_scraper import Full_Scraper
from scraping_options import parse_relative_upload_date_to_days
from analysis_pipeline.scoring.visual_rerank import analyze_listing_images, infer_category


DEFAULT_SORT_COLS = ["WorthBuying", "BuyDecisionScore", "ResaleSafetyScore", "ExpectedProfitMargin", "DealScore"]
DEFAULT_LOW_PRICE_SEARCH_TERMS = ["ps4", "ps5", "switch", "xbox", "game", "games"]
DEFAULT_COMPONENT_WEIGHTS = {
    "resale": 0.34,
    "profit": 0.18,
    "margin": 0.14,
    "seller": 0.14,
    "demand": 0.08,
    "fresh": 0.06,
    "condition": 0.06,
    "visual": 0.0,
}
DESCRIPTION_HARD_VETO_KEYWORDS = [
    "replica", "fake", "falso", "contraff", "ispirat", "inspired",
    "non funziona", "doesn't work", "doesnt work", "not working", "rotto", "broken",
]
DESCRIPTION_SOFT_RISK_KEYWORDS = [
    "difetto", "defect", "difett", "graffi", "graffio", "scratch", "stain", "macchia",
    "senza scatola", "no box", "senza dust bag", "no dust bag", "missing", "manca",
    "solo disco", "disc only", "solo cartuccia", "no manual", "senza accessori",
    "bundle", "lotto", "stock", "blocco", "untested", "da pulire", "da sistemare",
]
CONDITION_BAD_KEYWORDS = [
    "satisfactory", "acceptable", "fair", "da sistemare", "poor", "rotto", "broken",
]


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def safe_float(value, default=np.nan) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x):
        return default
    return x


def contains_any(text: str, keywords: list[str]) -> list[str]:
    norm = str(text or "").strip().lower()
    return [kw for kw in keywords if kw in norm]


def normalize_search_terms(raw_terms: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if raw_terms is None:
        return []
    if isinstance(raw_terms, str):
        parts = raw_terms.split(",")
    else:
        parts = list(raw_terms)
    return [str(part).strip().lower() for part in parts if str(part).strip()]


def parse_named_float_map(raw_value: str | None) -> dict[str, float]:
    if raw_value is None:
        return {}
    mapping: dict[str, float] = {}
    for chunk in str(raw_value).split(","):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip().lower()
        if not key:
            continue
        try:
            mapping[key] = float(value.strip())
        except ValueError:
            continue
    return mapping


def resolve_component_weights(raw_value: str | None) -> dict[str, float]:
    weights = dict(DEFAULT_COMPONENT_WEIGHTS)
    weights.update(parse_named_float_map(raw_value))
    return weights


def resolve_min_buy_score(row: pd.Series | dict, base_min_buy_score: float, category_thresholds: dict[str, float] | None = None) -> float:
    thresholds = category_thresholds or {}
    search_name = str(row.get("SearchName", "")) if isinstance(row, dict) else str(row.get("SearchName", ""))
    title = str(row.get("Title", "")) if isinstance(row, dict) else str(row.get("Title", ""))
    category = infer_category(search_name, title)
    specific_search = search_name.strip().lower()
    if specific_search in thresholds:
        return float(thresholds[specific_search])
    if category in thresholds:
        return float(thresholds[category])
    return float(base_min_buy_score)


def normalize_stars(value) -> float:
    text = str(value or "").strip().replace(",", ".")
    try:
        stars = float(text)
    except ValueError:
        return np.nan
    if stars > 5:
        stars = stars / 10.0
    return stars if 0 <= stars <= 5 else np.nan


def seller_metrics_state(stars, reviews_count) -> str:
    stars_num = normalize_stars(stars)
    reviews_num = safe_float(reviews_count, np.nan)
    has_stars = np.isfinite(stars_num)
    has_reviews = np.isfinite(reviews_num)
    positive_reviews = has_reviews and reviews_num > 0
    zero_reviews = has_reviews and reviews_num <= 0

    if positive_reviews and has_stars:
        return "reviewed"
    if (zero_reviews or not has_reviews) and not has_stars:
        return "new_unreviewed"
    return "incomplete"


def seller_quality_score(stars, reviews_count) -> float:
    stars_num = normalize_stars(stars)
    reviews_num = max(0.0, safe_float(reviews_count, 0.0))
    state = seller_metrics_state(stars, reviews_count)
    review_component = clamp(math.log1p(reviews_num) / math.log1p(50.0), 0.0, 1.0)

    if state == "new_unreviewed":
        # Realistic Vinted case: no reviews means no star average yet.
        return 0.35
    if state == "incomplete":
        # Reviews without stars or stars without reviews are scrape inconsistencies on Vinted.
        return float(0.20 + 0.15 * review_component)

    stars_component = clamp(stars_num / 5.0, 0.0, 1.0)
    # Review count acts as confidence in the seller's average rather than a separate quality axis.
    confidence_component = 0.70 + 0.30 * review_component
    return float(stars_component * confidence_component)


def demand_score(interested_count, view_count) -> float:
    interested = max(0.0, safe_float(interested_count, 0.0))
    views = max(0.0, safe_float(view_count, 0.0))
    interest_component = clamp(interested / 10.0, 0.0, 1.0)
    if views <= 0:
        ratio_component = 0.5 if interested > 0 else 0.0
    else:
        ratio_component = clamp((interested / views) / 0.15, 0.0, 1.0)
    return float(0.6 * interest_component + 0.4 * ratio_component)


def freshness_score(upload_days) -> float:
    days = safe_float(upload_days, np.nan)
    if not np.isfinite(days):
        return 0.55
    if days <= 1:
        return 1.0
    if days <= 3:
        return 0.9
    if days <= 7:
        return 0.75
    if days <= 14:
        return 0.55
    if days <= 30:
        return 0.35
    return 0.15


def condition_quality_score(condition_text: str) -> float:
    norm = str(condition_text or "").strip().lower()
    if not norm or norm == "unknown":
        return 0.6
    if any(kw in norm for kw in CONDITION_BAD_KEYWORDS):
        return 0.2
    if "new with tags" in norm or "nuovo con cartellino" in norm:
        return 1.0
    if "new" in norm or "nuovo" in norm:
        return 0.9
    if "very good" in norm or "ottime" in norm or "excellent" in norm:
        return 0.82
    if "good" in norm or "buone" in norm:
        return 0.68
    return 0.55


def extract_buy_components(enriched_row: dict, min_seller_score: float) -> dict:
    base_resale_safety = safe_float(enriched_row.get("ResaleSafetyScore"), np.nan)
    if np.isfinite(base_resale_safety):
        resale_component = clamp(base_resale_safety / 100.0, 0.0, 1.0)
    else:
        deal_score = safe_float(enriched_row.get("DealScore"), np.nan)
        confidence = safe_float(enriched_row.get("DealConfidence"), 0.0)
        profit_margin = safe_float(enriched_row.get("ExpectedProfitMargin"), np.nan)
        resale_component = clamp(((deal_score if np.isfinite(deal_score) else 0.0) / 6.0), 0.0, 1.0)
        resale_component *= clamp(confidence, 0.0, 1.0)
        if np.isfinite(profit_margin):
            resale_component *= clamp((profit_margin + 0.05) / 0.55, 0.0, 1.0)

    expected_profit = safe_float(enriched_row.get("ExpectedProfit"), np.nan)
    expected_margin = safe_float(enriched_row.get("ExpectedProfitMargin"), np.nan)
    profit_component = clamp(expected_profit / 40.0, 0.0, 1.0) if np.isfinite(expected_profit) else 0.45
    margin_component = clamp(expected_margin / 0.50, 0.0, 1.0) if np.isfinite(expected_margin) else 0.45
    seller_state = seller_metrics_state(enriched_row.get("Stars"), enriched_row.get("ReviewsCount"))
    seller_component = seller_quality_score(enriched_row.get("Stars"), enriched_row.get("ReviewsCount"))
    demand_component = demand_score(enriched_row.get("Interested_count"), enriched_row.get("View_count"))
    fresh_component = freshness_score(enriched_row.get("Upload_date_days"))
    condition_component = condition_quality_score(enriched_row.get("Condition"))
    visual_penalty = safe_float(enriched_row.get("VisualRiskPenalty"), 0.0) or 0.0
    visual_score = safe_float(enriched_row.get("VisualScore"), np.nan)

    hard_flags = contains_any(enriched_row.get("Description"), DESCRIPTION_HARD_VETO_KEYWORDS)
    soft_flags = contains_any(enriched_row.get("Description"), DESCRIPTION_SOFT_RISK_KEYWORDS)
    soft_flags += contains_any(enriched_row.get("Condition"), CONDITION_BAD_KEYWORDS)
    soft_flags = sorted(set(soft_flags))

    notes = []
    penalty = 0.0
    if hard_flags:
        penalty += 0.55
        notes.append("hard_description_risk")
    if soft_flags:
        penalty += min(0.25, 0.06 * len(soft_flags))
        notes.append("description_or_condition_risk")
    if seller_state == "new_unreviewed":
        notes.append("new_unreviewed_seller")
    elif seller_state == "incomplete":
        notes.append("seller_metrics_incomplete")
    if seller_component < min_seller_score:
        penalty += 0.20
        notes.append("weak_seller_profile")
    if condition_component < 0.35:
        penalty += 0.20
        notes.append("poor_condition")
    if np.isfinite(expected_margin) and expected_margin < 0.10:
        penalty += 0.18
        notes.append("thin_expected_margin")
    if np.isfinite(expected_profit) and expected_profit < 8.0:
        penalty += 0.12
        notes.append("thin_expected_profit")
    if np.isfinite(visual_score) and visual_score < 0.42:
        notes.append("visual_low_confidence")
    if visual_penalty >= 0.12:
        notes.append("visual_risk")

    return {
        "resale": resale_component,
        "profit": profit_component,
        "margin": margin_component,
        "seller": seller_component,
        "demand": demand_component,
        "fresh": fresh_component,
        "condition": condition_component,
        "visual": clamp(visual_score, 0.0, 1.0) if np.isfinite(visual_score) else 0.0,
        "visual_penalty": float(visual_penalty),
        "rule_penalty": float(penalty),
        "hard_flags": hard_flags,
        "soft_flags": soft_flags,
        "notes": notes,
    }


def compute_buy_decision(
    enriched_row: dict,
    min_buy_score: float,
    min_seller_score: float,
    component_weights: dict[str, float] | None = None,
    visual_penalty_scale: float = 1.0,
) -> tuple[float, bool, str]:
    weights = component_weights or DEFAULT_COMPONENT_WEIGHTS
    parts = extract_buy_components(enriched_row, min_seller_score)
    score = (
        float(weights.get("resale", 0.0)) * parts["resale"]
        + float(weights.get("profit", 0.0)) * parts["profit"]
        + float(weights.get("margin", 0.0)) * parts["margin"]
        + float(weights.get("seller", 0.0)) * parts["seller"]
        + float(weights.get("demand", 0.0)) * parts["demand"]
        + float(weights.get("fresh", 0.0)) * parts["fresh"]
        + float(weights.get("condition", 0.0)) * parts["condition"]
        + float(weights.get("visual", 0.0)) * parts["visual"]
        - parts["rule_penalty"]
        - float(visual_penalty_scale) * parts["visual_penalty"]
    )
    score = clamp(score, 0.0, 1.0)
    worth_buying = (score >= min_buy_score) and not parts["hard_flags"]

    note_bits = parts["notes"][:]
    if parts["hard_flags"]:
        note_bits.append("hard:" + "|".join(parts["hard_flags"]))
    if parts["soft_flags"]:
        note_bits.append("soft:" + "|".join(parts["soft_flags"]))
    return score, worth_buying, ";".join(note_bits)


def enrich_one(scraper: Full_Scraper, row_dict: dict) -> dict:
    data_id = row_dict.get("Dataid")
    link = row_dict.get("Link")
    item_info, seller_info = scraper.scrape_single_product(url=link, data_id=data_id, get_images=True)
    out = dict(row_dict)
    out["Description"] = item_info.get("Description", "")
    out["Condition"] = item_info.get("Condition", "")
    out["Upload_date"] = item_info.get("Upload_date", "")
    out["Upload_date_days"] = parse_relative_upload_date_to_days(out["Upload_date"])
    out["Interested_count"] = item_info.get("Interested_count", np.nan)
    out["View_count"] = item_info.get("View_count", np.nan)
    out["SellerName"] = seller_info.get("SellerName", item_info.get("SellerName", ""))
    out["SellerId"] = seller_info.get("SellerId", item_info.get("SellerId", ""))
    out["Location"] = seller_info.get("Location", "")
    out["ReviewsCount"] = seller_info.get("ReviewsCount", np.nan)
    out["Stars"] = seller_info.get("Stars", np.nan)
    out["PrimaryImageUrl"] = item_info.get("PrimaryImageUrl", "")
    out["FullImageUrls"] = item_info.get("FullImageUrls", [])
    out["VisiblePictureCount"] = item_info.get("VisiblePictureCount", 0)
    out["HiddenPictureCount"] = item_info.get("HiddenPictureCount", 0)
    out["PictureCount"] = item_info.get("PictureCount", 0)
    if item_info.get("Images"):
        out["Images"] = item_info.get("Images")
    return out


def enrich_candidates(df: pd.DataFrame, max_workers: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows = [row._asdict() for row in df.itertuples(index=False)]
    results: list[dict] = []
    scraper = Full_Scraper()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(enrich_one, scraper, row) for row in rows]
        for fut in as_completed(futures):
            results.append(fut.result())
    enriched = pd.DataFrame(results)
    ordered_cols = [c for c in df.columns if c in enriched.columns] + [c for c in enriched.columns if c not in df.columns]
    return enriched[ordered_cols]


def apply_visual_rerank(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    visual = out.apply(
        lambda row: analyze_listing_images(
            row.get("LocalImagePaths") or row.get("LocalPrimaryImagePath") or row.get("Images"),
            title=str(row.get("Title", "")),
            search_name=str(row.get("SearchName", "")),
            max_images=args.visual_max_images,
            main_image_weight=args.visual_main_image_weight,
            timeout=args.visual_timeout,
            enable_clip=args.visual_enable_clip,
        ),
        axis=1,
        result_type="expand",
    )
    return pd.concat([out.reset_index(drop=True), visual.reset_index(drop=True)], axis=1)


def is_low_price_context(row: pd.Series, low_price_cutoff: float, low_price_search_terms: list[str]) -> bool:
    price = pd.to_numeric(pd.Series([row.get("Price")]), errors="coerce").iloc[0]
    if np.isfinite(price) and price <= float(low_price_cutoff):
        return True

    haystacks = [str(row.get("SearchName", "")).lower(), str(row.get("Title", "")).lower()]
    return any(term in haystack for term in low_price_search_terms for haystack in haystacks if haystack)


def required_expected_profit(row: pd.Series, args: argparse.Namespace) -> float | None:
    if args.min_expected_profit is None:
        return None

    base_threshold = float(args.min_expected_profit)
    low_price_terms = normalize_search_terms(getattr(args, "low_price_search_terms", DEFAULT_LOW_PRICE_SEARCH_TERMS))
    if not is_low_price_context(row, args.low_price_cutoff, low_price_terms):
        return base_threshold

    price = pd.to_numeric(pd.Series([row.get("Price")]), errors="coerce").iloc[0]
    relative_floor = float(args.low_price_min_expected_profit)
    if np.isfinite(price) and price > 0:
        relative_floor = max(relative_floor, float(price) * float(args.low_price_profit_ratio))
    return min(base_threshold, relative_floor)


def select_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if "DealEligible" in out.columns and args.require_deal_eligible:
        out = out[out["DealEligible"].fillna(False)]
    if "ResaleSafetyScore" in out.columns and args.min_resale_safety is not None:
        out = out[pd.to_numeric(out["ResaleSafetyScore"], errors="coerce") >= args.min_resale_safety]
    if "DealConfidence" in out.columns and args.min_deal_confidence is not None:
        out = out[pd.to_numeric(out["DealConfidence"], errors="coerce") >= args.min_deal_confidence]
    if "ExpectedProfit" in out.columns and args.min_expected_profit is not None:
        out["_ExpectedProfitNum"] = pd.to_numeric(out["ExpectedProfit"], errors="coerce")
        out["_RequiredExpectedProfit"] = out.apply(lambda row: required_expected_profit(row, args), axis=1)
        out = out[out["_ExpectedProfitNum"] >= out["_RequiredExpectedProfit"]]
    if "ExpectedProfitMargin" in out.columns and args.min_expected_profit_margin is not None:
        out = out[pd.to_numeric(out["ExpectedProfitMargin"], errors="coerce") >= args.min_expected_profit_margin]
    sort_cols = [c for c in DEFAULT_SORT_COLS if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    if args.top_n is not None:
        out = out.head(args.top_n)
    out = out.drop(columns=["_ExpectedProfitNum"], errors="ignore")
    return out.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Enrich top deal candidates with full item metadata and compute a final buy/not-buy decision.")
    ap.add_argument("--input", required=True, help="Path to deals_ranked.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--min_resale_safety", type=float, default=55.0)
    ap.add_argument("--min_deal_confidence", type=float, default=0.60)
    ap.add_argument("--min_expected_profit", type=float, default=15.0)
    ap.add_argument("--min_expected_profit_margin", type=float, default=0.30)
    ap.add_argument("--low_price_cutoff", type=float, default=25.0)
    ap.add_argument("--low_price_min_expected_profit", type=float, default=3.0)
    ap.add_argument("--low_price_profit_ratio", type=float, default=0.35)
    ap.add_argument(
        "--low_price_search_terms",
        default=",".join(DEFAULT_LOW_PRICE_SEARCH_TERMS),
        help="Comma-separated tokens that identify low-price searches where absolute profit floors should be relaxed.",
    )
    ap.add_argument("--min_buy_score", type=float, default=0.75)
    ap.add_argument("--min_seller_score", type=float, default=0.45)
    ap.add_argument(
        "--category_min_buy_scores",
        default="",
        help="Comma-separated overrides like 'prada=0.70,gucci=0.78,game=0.80,luxury=0.72'. Search-specific keys win over category keys.",
    )
    ap.add_argument(
        "--component_weights",
        default="",
        help="Comma-separated score weights like 'resale=0.30,profit=0.12,margin=0.10,seller=0.18,demand=0.08,fresh=0.04,condition=0.06,visual=0.12'.",
    )
    ap.add_argument("--visual_penalty_scale", type=float, default=1.0)
    ap.add_argument("--visual_max_images", type=int, default=6)
    ap.add_argument("--visual_main_image_weight", type=float, default=0.55)
    ap.add_argument("--visual_timeout", type=float, default=8.0)
    ap.add_argument("--visual_enable_clip", action="store_true")
    ap.add_argument("--require_deal_eligible", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    category_thresholds = parse_named_float_map(args.category_min_buy_scores)
    component_weights = resolve_component_weights(args.component_weights)

    deals = pd.read_csv(args.input)
    candidates = select_candidates(deals, args)
    candidates.to_csv(out_dir / "buy_candidates_input.csv", index=False)

    if candidates.empty:
        summary = {
            "n_input_rows": int(len(deals)),
            "n_candidates": 0,
            "n_worth_buying": 0,
        }
        (out_dir / "buy_decision_summary.json").write_text(json.dumps(summary, indent=2))
        print("No candidates matched the final buy-filter thresholds.")
        return

    enriched = enrich_candidates(candidates, max_workers=args.max_workers)
    enriched = apply_visual_rerank(enriched, args)
    scores = enriched.apply(
        lambda row: compute_buy_decision(
            row.to_dict(),
            resolve_min_buy_score(row, args.min_buy_score, category_thresholds),
            args.min_seller_score,
            component_weights=component_weights,
            visual_penalty_scale=args.visual_penalty_scale,
        ),
        axis=1,
        result_type="expand",
    )
    scores.columns = ["BuyDecisionScore", "WorthBuying", "BuyDecisionNotes"]
    enriched[["BuyDecisionScore", "WorthBuying", "BuyDecisionNotes"]] = scores

    enriched["SellerQualityScore"] = enriched.apply(lambda row: seller_quality_score(row.get("Stars"), row.get("ReviewsCount")), axis=1)
    enriched["DemandScore"] = enriched.apply(lambda row: demand_score(row.get("Interested_count"), row.get("View_count")), axis=1)
    enriched["FreshnessScore"] = enriched["Upload_date_days"].apply(freshness_score)
    enriched["ConditionQualityScore"] = enriched["Condition"].apply(condition_quality_score)
    enriched["DescriptionHardFlags"] = enriched["Description"].apply(lambda x: "|".join(contains_any(x, DESCRIPTION_HARD_VETO_KEYWORDS)))
    enriched["DescriptionSoftFlags"] = enriched["Description"].apply(lambda x: "|".join(contains_any(x, DESCRIPTION_SOFT_RISK_KEYWORDS)))

    sort_cols = [c for c in DEFAULT_SORT_COLS if c in enriched.columns]
    enriched = enriched.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    enriched.to_csv(out_dir / "buy_candidates_enriched.csv", index=False)

    final_buys = enriched[enriched["WorthBuying"].fillna(False)].copy()
    final_buys.to_csv(out_dir / "buy_candidates_recommended.csv", index=False)

    summary = {
        "n_input_rows": int(len(deals)),
        "n_candidates": int(len(candidates)),
        "n_enriched": int(len(enriched)),
        "n_worth_buying": int(len(final_buys)),
        "mean_buy_decision_score": float(enriched["BuyDecisionScore"].mean()) if not enriched.empty else None,
    }
    (out_dir / "buy_decision_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Candidates enriched: {len(enriched)}")
    print(f"Worth buying: {len(final_buys)}")
    print(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
