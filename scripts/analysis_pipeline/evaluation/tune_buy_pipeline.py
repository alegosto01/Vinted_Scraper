#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SEARCHES = ("ps4", "gucci", "prada")


@dataclass
class RuleResult:
    stage: str
    precision: float
    sold_count: int
    selected_count: int
    recall: float
    details: dict


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_buy_frame(base_dir: Path, search: str) -> pd.DataFrame:
    df = pd.read_csv(base_dir / search / "buy_decision_eval" / "buy_candidates_enriched.csv")
    sold = pd.read_csv(base_dir / search / "sold_df.csv")
    sold_eventually_path = base_dir / search / "eventual_sale_check" / "sold_eventually.csv"
    sold_eventually = pd.read_csv(sold_eventually_path) if sold_eventually_path.exists() else pd.DataFrame(columns=["Dataid"])

    df["Dataid"] = normalize_id_series(df["Dataid"])
    sold_ids = set(normalize_id_series(sold["Dataid"]).dropna()) if "Dataid" in sold.columns else set()
    sold_eventually_ids = set(normalize_id_series(sold_eventually["Dataid"]).dropna()) if "Dataid" in sold_eventually.columns else set()
    df["SoldLabel"] = df["Dataid"].isin(sold_ids | sold_eventually_ids).astype(int)
    df["Search"] = search
    return df


def load_deal_frame(base_dir: Path, search: str) -> pd.DataFrame:
    df = pd.read_csv(base_dir / search / "deal_score_eval" / "deals_ranked_labeled.csv")
    df["Search"] = search
    return df


def precision_summary(mask: np.ndarray, y: np.ndarray) -> tuple[float, int, int, float]:
    selected = int(mask.sum())
    sold = int(y[mask].sum()) if selected else 0
    precision = (sold / selected) if selected else 0.0
    total_sold = int(y.sum())
    recall = (sold / total_sold) if total_sold else 0.0
    return precision, sold, selected, recall


def top_rule(results: list[RuleResult], min_selected: int) -> RuleResult | None:
    eligible = [r for r in results if r.selected_count >= min_selected]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda r: (r.precision, r.sold_count, -r.selected_count, r.recall),
        reverse=True,
    )[0]


def scan_deal_rules(df: pd.DataFrame) -> list[RuleResult]:
    for col in ("ResaleSafetyScore", "ExpectedProfitMargin", "ExpectedProfit", "DealScore", "DealConfidence"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    y = df["SoldLabel"].to_numpy(dtype=int)
    out: list[RuleResult] = []
    for rs in (50, 60, 70, 80, 90):
        for margin in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            for profit in (5, 10, 15, 20, 30, 40, 50):
                for deal in (0, 1, 2, 3, 4, 5):
                    for conf in (0.55, 0.6, 0.7, 0.8, 0.9):
                        mask = (
                            (df["ResaleSafetyScore"] >= rs)
                            & (df["ExpectedProfitMargin"] >= margin)
                            & (df["ExpectedProfit"] >= profit)
                            & (df["DealScore"] >= deal)
                            & (df["DealConfidence"] >= conf)
                        ).to_numpy(dtype=bool)
                        if not mask.any():
                            continue
                        precision, sold, selected, recall = precision_summary(mask, y)
                        out.append(
                            RuleResult(
                                stage="deal",
                                precision=precision,
                                sold_count=sold,
                                selected_count=selected,
                                recall=recall,
                                details={
                                    "min_resale_safety": rs,
                                    "min_expected_profit_margin": margin,
                                    "min_expected_profit": profit,
                                    "min_deal_score": deal,
                                    "min_deal_confidence": conf,
                                },
                            )
                        )
    return out


def scan_buy_rules(df: pd.DataFrame, trials: int, seed: int) -> list[RuleResult]:
    random.seed(seed)
    np.random.seed(seed)

    numeric_cols = [
        "ResaleSafetyScore",
        "ExpectedProfit",
        "ExpectedProfitMargin",
        "SellerQualityScore",
        "DemandScore",
        "FreshnessScore",
        "ConditionQualityScore",
        "DealScore",
        "DealConfidence",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def clip01(values: pd.Series) -> np.ndarray:
        return np.clip(np.nan_to_num(values.to_numpy(dtype=float), nan=0.0), 0.0, 1.0)

    features = {
        "resale": clip01(df["ResaleSafetyScore"] / 100.0),
        "profit": clip01(df["ExpectedProfit"] / 60.0),
        "margin": clip01(df["ExpectedProfitMargin"] / 0.8),
        "seller": clip01(df["SellerQualityScore"]),
        "demand": clip01(df["DemandScore"]),
        "fresh": clip01(df["FreshnessScore"]),
        "condition": clip01(df["ConditionQualityScore"]),
        "deal": clip01(df["DealScore"] / 6.0),
        "conf": clip01(df["DealConfidence"]),
    }
    feature_names = list(features)
    y = df["SoldLabel"].to_numpy(dtype=int)

    hard = (df["DescriptionHardFlags"].fillna("").astype(str) != "").to_numpy(dtype=float)
    soft = (df["DescriptionSoftFlags"].fillna("").astype(str) != "").to_numpy(dtype=float)
    weak_seller = (df["SellerQualityScore"] < 0.45).fillna(False).to_numpy(dtype=float)
    poor_condition = (df["ConditionQualityScore"] < 0.35).fillna(False).to_numpy(dtype=float)
    thin_margin = (df["ExpectedProfitMargin"] < 0.10).fillna(False).to_numpy(dtype=float)
    thin_profit = (df["ExpectedProfit"] < 8.0).fillna(False).to_numpy(dtype=float)

    out: list[RuleResult] = []
    for _ in range(trials):
        weights = np.array([random.uniform(0.0, 1.0) for _ in feature_names], dtype=float)
        weights /= max(weights.sum(), 1e-9)
        penalties = {
            "hard": random.uniform(0.0, 0.8),
            "soft": random.uniform(0.0, 0.3),
            "weak_seller": random.uniform(0.0, 0.35),
            "poor_condition": random.uniform(0.0, 0.35),
            "thin_margin": random.uniform(0.0, 0.25),
            "thin_profit": random.uniform(0.0, 0.2),
        }
        threshold = random.uniform(0.35, 0.95)
        min_conf = random.choice([None, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9])
        min_resale = random.choice([None, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8])
        min_margin = random.choice([None, 0.1, 0.2, 0.3, 0.4, 0.5])
        min_profit = random.choice([None, 0.1, 0.2, 0.3, 0.4, 0.5])

        score = np.zeros(len(df), dtype=float)
        for idx, name in enumerate(feature_names):
            score += weights[idx] * features[name]
        score -= (
            penalties["hard"] * hard
            + penalties["soft"] * soft
            + penalties["weak_seller"] * weak_seller
            + penalties["poor_condition"] * poor_condition
            + penalties["thin_margin"] * thin_margin
            + penalties["thin_profit"] * thin_profit
        )

        mask = score >= threshold
        if min_conf is not None:
            mask &= features["conf"] >= min_conf
        if min_resale is not None:
            mask &= features["resale"] >= min_resale
        if min_margin is not None:
            mask &= features["margin"] >= min_margin
        if min_profit is not None:
            mask &= features["profit"] >= min_profit
        if not mask.any():
            continue

        precision, sold, selected, recall = precision_summary(mask, y)
        out.append(
            RuleResult(
                stage="buy",
                precision=precision,
                sold_count=sold,
                selected_count=selected,
                recall=recall,
                details={
                    "threshold": threshold,
                    "weights": {name: float(weights[idx]) for idx, name in enumerate(feature_names)},
                    "penalties": penalties,
                    "min_conf": min_conf,
                    "min_resale": min_resale,
                    "min_margin": min_margin,
                    "min_profit": min_profit,
                },
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune buy/deal rules across ps4, gucci, and prada.")
    ap.add_argument("--base_dir", default="data/simple_scrape")
    ap.add_argument("--out_dir", default="data/simple_scrape/tuning_reports")
    ap.add_argument("--buy_trials", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buy_df = pd.concat([load_buy_frame(base_dir, search) for search in SEARCHES], ignore_index=True)
    deal_df = pd.concat([load_deal_frame(base_dir, search) for search in SEARCHES], ignore_index=True)

    buy_results = scan_buy_rules(buy_df.copy(), trials=args.buy_trials, seed=args.seed)
    deal_results = scan_deal_rules(deal_df.copy())

    report = {
        "buy_rows": int(len(buy_df)),
        "buy_sold_total": int(buy_df["SoldLabel"].sum()),
        "deal_rows": int(len(deal_df)),
        "deal_sold_total": int(deal_df["SoldLabel"].sum()),
        "best_buy_min_1": asdict(top_rule(buy_results, 1)) if top_rule(buy_results, 1) else None,
        "best_buy_min_2": asdict(top_rule(buy_results, 2)) if top_rule(buy_results, 2) else None,
        "best_buy_min_3": asdict(top_rule(buy_results, 3)) if top_rule(buy_results, 3) else None,
        "best_buy_min_5": asdict(top_rule(buy_results, 5)) if top_rule(buy_results, 5) else None,
        "best_deal_min_1": asdict(top_rule(deal_results, 1)) if top_rule(deal_results, 1) else None,
        "best_deal_min_2": asdict(top_rule(deal_results, 2)) if top_rule(deal_results, 2) else None,
        "best_deal_min_5": asdict(top_rule(deal_results, 5)) if top_rule(deal_results, 5) else None,
        "best_deal_min_10": asdict(top_rule(deal_results, 10)) if top_rule(deal_results, 10) else None,
    }

    out_path = out_dir / "buy_pipeline_tuning_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
