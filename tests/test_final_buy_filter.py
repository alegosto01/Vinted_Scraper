import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

sys.modules.setdefault("full_scraper", types.SimpleNamespace(Full_Scraper=object))
existing_scraping_options = sys.modules.get("scraping_options")
if existing_scraping_options is not None and not hasattr(existing_scraping_options, "parse_relative_upload_date_to_days"):
    sys.modules.pop("scraping_options", None)
sys.modules.setdefault(
    "scraping_options",
    types.SimpleNamespace(parse_relative_upload_date_to_days=lambda value: value),
)

from analysis_pipeline.scoring.final_buy_filter import (
    apply_visual_rerank,
    compute_buy_decision,
    parse_named_float_map,
    required_expected_profit,
    resolve_min_buy_score,
    select_candidates,
    seller_metrics_state,
    seller_quality_score,
)


class FinalBuyFilterTests(unittest.TestCase):
    def make_args(self, **overrides):
        base = dict(
            require_deal_eligible=False,
            min_resale_safety=35.0,
            min_deal_confidence=0.60,
            min_expected_profit=8.0,
            min_expected_profit_margin=0.12,
            top_n=None,
            low_price_cutoff=25.0,
            low_price_min_expected_profit=3.0,
            low_price_profit_ratio=0.35,
            low_price_search_terms="ps4,ps5,switch,xbox,game,games",
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_required_expected_profit_relaxes_for_low_price_context(self):
        args = self.make_args(min_expected_profit=15.0)
        row = pd.Series({"SearchName": "ps4", "Title": "Detroit become human ps4", "Price": 11.2})
        self.assertAlmostEqual(required_expected_profit(row, args), 3.92, places=2)

    def test_required_expected_profit_keeps_base_for_high_price_item(self):
        args = self.make_args(min_expected_profit=15.0)
        row = pd.Series({"SearchName": "prada", "Title": "Prada shoes", "Price": 79.45})
        self.assertAlmostEqual(required_expected_profit(row, args), 15.0)

    def test_select_candidates_keeps_low_price_high_margin_item(self):
        args = self.make_args(min_expected_profit=15.0, min_expected_profit_margin=0.30)
        df = pd.DataFrame(
            [
                {
                    "SearchName": "ps4",
                    "Title": "Detroit become human ps4",
                    "Price": 11.2,
                    "DealEligible": True,
                    "ResaleSafetyScore": 87.35,
                    "DealConfidence": 1.0,
                    "ExpectedProfit": 7.35,
                    "ExpectedProfitMargin": 0.65,
                    "WorthBuying": False,
                    "BuyDecisionScore": 0.0,
                    "DealScore": 6.7,
                },
                {
                    "SearchName": "prada",
                    "Title": "Prada shoes",
                    "Price": 79.45,
                    "DealEligible": True,
                    "ResaleSafetyScore": 55.0,
                    "DealConfidence": 1.0,
                    "ExpectedProfit": 7.35,
                    "ExpectedProfitMargin": 0.65,
                    "WorthBuying": False,
                    "BuyDecisionScore": 0.0,
                    "DealScore": 6.7,
                },
            ]
        )

        selected = select_candidates(df, args)
        self.assertEqual(selected["SearchName"].tolist(), ["ps4"])
        self.assertAlmostEqual(float(selected["_RequiredExpectedProfit"].iloc[0]), 3.92, places=2)

    def test_compute_buy_decision_applies_visual_penalty(self):
        row = {
            "ResaleSafetyScore": 80.0,
            "ExpectedProfit": 20.0,
            "ExpectedProfitMargin": 0.40,
            "Stars": 5.0,
            "ReviewsCount": 20,
            "Interested_count": 5,
            "View_count": 40,
            "Upload_date_days": 1,
            "Condition": "very good",
            "Description": "",
            "VisualRiskPenalty": 0.25,
            "VisualScore": 0.30,
        }
        score, worth_buying, notes = compute_buy_decision(row, min_buy_score=0.75, min_seller_score=0.45)
        self.assertLess(score, 0.75)
        self.assertFalse(worth_buying)
        self.assertIn("visual_risk", notes)

    def test_compute_buy_decision_supports_visual_weight_override(self):
        row = {
            "ResaleSafetyScore": 20.0,
            "ExpectedProfit": 5.0,
            "ExpectedProfitMargin": 0.05,
            "Stars": 5.0,
            "ReviewsCount": 50,
            "Interested_count": 0,
            "View_count": 0,
            "Upload_date_days": 1,
            "Condition": "new",
            "Description": "",
            "VisualRiskPenalty": 0.0,
            "VisualScore": 1.0,
        }
        score, worth_buying, _ = compute_buy_decision(
            row,
            min_buy_score=0.40,
            min_seller_score=0.45,
            component_weights={
                "resale": 0.05,
                "profit": 0.0,
                "margin": 0.0,
                "seller": 0.0,
                "demand": 0.0,
                "fresh": 0.0,
                "condition": 0.0,
                "visual": 1.0,
            },
            visual_penalty_scale=0.0,
        )
        self.assertGreaterEqual(score, 0.40)
        self.assertTrue(worth_buying)

    def test_resolve_min_buy_score_prefers_search_specific_override(self):
        row = pd.Series({"SearchName": "prada", "Title": "Prada Cups - Size 42"})
        thresholds = parse_named_float_map("luxury=0.72,prada=0.68")
        self.assertAlmostEqual(resolve_min_buy_score(row, 0.75, thresholds), 0.68)

    def test_apply_visual_rerank_adds_visual_columns(self):
        args = self.make_args(
            visual_max_images=3,
            visual_main_image_weight=0.55,
            visual_timeout=0.1,
            visual_enable_clip=False,
        )
        df = pd.DataFrame(
            [
                {
                    "Title": "Sample item",
                    "SearchName": "ps4",
                    "Images": "https://example.com/image.webp",
                }
            ]
        )

        import analysis_pipeline.scoring.final_buy_filter as mod

        original = mod.analyze_listing_images
        mod.analyze_listing_images = lambda *a, **k: {
            "VisualScore": 0.7,
            "VisualRiskPenalty": 0.05,
            "VisualQualityScore": 0.8,
            "VisualCompletenessScore": 0.7,
            "VisualConsistencyScore": 0.6,
            "VisualAuthenticityScore": 0.65,
            "VisualImageCount": 2,
            "VisualUniqueImageCount": 2,
            "VisualLowQualityFraction": 0.0,
            "VisualScreenshotFraction": 0.0,
            "VisualAnalysisNotes": "ok",
        }
        try:
            out = apply_visual_rerank(df, args)
        finally:
            mod.analyze_listing_images = original

        self.assertAlmostEqual(float(out.loc[0, "VisualScore"]), 0.7)
        self.assertEqual(int(out.loc[0, "VisualImageCount"]), 2)

    def test_seller_quality_score_treats_new_seller_as_unproven(self):
        self.assertEqual(seller_metrics_state(np.nan, 0), "new_unreviewed")
        self.assertAlmostEqual(seller_quality_score(np.nan, 0), 0.35)

    def test_seller_quality_score_flags_incomplete_seller_metrics(self):
        self.assertEqual(seller_metrics_state(np.nan, 20), "incomplete")
        self.assertLess(seller_quality_score(np.nan, 20), 0.45)

    def test_compute_buy_decision_notes_incomplete_seller_metrics(self):
        row = {
            "ResaleSafetyScore": 85.0,
            "ExpectedProfit": 20.0,
            "ExpectedProfitMargin": 0.40,
            "Stars": np.nan,
            "ReviewsCount": 20,
            "Interested_count": 5,
            "View_count": 40,
            "Upload_date_days": 1,
            "Condition": "very good",
            "Description": "",
            "VisualRiskPenalty": 0.0,
            "VisualScore": 0.8,
        }
        _, worth_buying, notes = compute_buy_decision(row, min_buy_score=0.75, min_seller_score=0.45)
        self.assertFalse(worth_buying)
        self.assertIn("seller_metrics_incomplete", notes)
        self.assertIn("weak_seller_profile", notes)


if __name__ == "__main__":
    unittest.main()
