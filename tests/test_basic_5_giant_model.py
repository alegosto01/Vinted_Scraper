import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.basic_5_giant_model import run as giant
from experiments.current.basic_5_giant_model import apply_to_live_collector
from experiments.current.full_scrape_giant_model import full_scrape_features
from experiments.old.full_scrape_reranker import full_scrape_reranker
from experiments.current.basic_5_giant_model import title_features
from experiments.current.basic_5_giant_model import report_per_search_thresholds
from experiments.current.basic_5_giant_model import report_weighted_voting
from experiments.old.deal_finder import model_sweep


class Basic5GiantModelTests(unittest.TestCase):
    def test_search_onehot_columns_are_stable(self):
        frame = pd.DataFrame({"SearchName": ["ps4", "gucci", "ps4"]})

        out, cols = giant.add_search_onehot(frame, ["gucci", "ps4"])

        self.assertEqual(cols, ["search__gucci", "search__ps4"])
        self.assertEqual(out["search__gucci"].tolist(), [0, 1, 0])
        self.assertEqual(out["search__ps4"].tolist(), [1, 0, 1])

    def test_stratified_global_split_preserves_search_label_slices(self):
        rows = []
        for search in ("gucci", "ps4"):
            for label in (0, 1):
                for idx in range(20):
                    rows.append(
                        {
                            "SearchName": search,
                            "offline_sold_label": label,
                            "item_id": f"{search}-{label}-{idx}",
                        }
                    )
        frame = pd.DataFrame(rows)

        splits = giant.stratified_global_split(frame, seed=42)

        self.assertEqual((len(splits.train), len(splits.validation), len(splits.test)), (48, 16, 16))
        for part in (splits.train, splits.validation, splits.test):
            observed = set(zip(part["SearchName"], part["offline_sold_label"]))
            self.assertEqual(observed, {("gucci", 0), ("gucci", 1), ("ps4", 0), ("ps4", 1)})

    def test_feature_selection_adds_search_identity_without_full_features(self):
        frame = pd.DataFrame(
            {
                "Price": [10, 20],
                "Likes": [1, 2],
                "Title": ["nike shoes", "prada bag"],
                "Brand": ["Nike", "Prada"],
                "Size": ["Medium", "Large"],
                "search__nike": [1, 0],
                "search__prada": [0, 1],
                "SearchCount": [99, 88],
                "MarketStatus": ["Sold", "On Sale"],
            }
        )
        spec = next(item for item in giant.available_basic5_specs() if item.name == "random_forest_basic_v1")

        numeric, text = giant.select_basic5_features(frame, spec, ["search__nike", "search__prada"])

        self.assertEqual(numeric, ["Price", "Likes", "search__nike", "search__prada"])
        self.assertEqual(text, ["Title", "Brand", "Size"])
        self.assertNotIn("SearchCount", numeric)
        self.assertFalse(model_sweep.has_leakage_features(numeric + text))

    def test_per_search_threshold_rows_tune_each_search(self):
        validation = pd.DataFrame(
            {
                "SearchName": ["gucci", "gucci", "gucci", "ps4", "ps4", "ps4"],
                "offline_sold_label": [1, 1, 0, 1, 0, 0],
                "score__demo": [0.9, 0.8, 0.1, 0.7, 0.6, 0.1],
            }
        )
        test = pd.DataFrame(
            {
                "SearchName": ["gucci", "gucci", "ps4", "ps4"],
                "offline_sold_label": [1, 0, 1, 0],
                "score__demo": [0.85, 0.2, 0.75, 0.55],
            }
        )

        rows = report_per_search_thresholds.per_search_threshold_rows(
            validation,
            test,
            min_precision=0.8,
            min_count=1,
        )

        by_search = {row["search_name"]: row for row in rows}
        self.assertEqual(set(by_search), {"gucci", "ps4"})
        self.assertGreater(by_search["gucci"]["per_search_threshold"], by_search["ps4"]["per_search_threshold"])
        self.assertEqual(by_search["gucci"]["threshold_count"], 1)
        self.assertEqual(by_search["ps4"]["threshold_count"], 1)

    def test_weighted_soft_vote_uses_validation_weights_by_search(self):
        validation_metrics = pd.DataFrame(
            {
                "search_name": ["gucci", "gucci"],
                "approach": ["weak", "strong"],
                "roc_auc": [0.55, 0.75],
                "pr_auc": [0.50, 0.80],
                "base_rate": [0.40, 0.40],
                "precision_at_25": [0.45, 0.90],
                "threshold": [0.5, 0.5],
                "threshold_count": [10, 10],
                "threshold_precision": [0.6, 0.9],
                "threshold_qualified": [False, True],
            }
        )
        frame = pd.DataFrame(
            {
                "SearchName": ["gucci", "gucci"],
                "score__weak": [1.0, 1.0],
                "score__strong": [0.0, 1.0],
            }
        )
        scheme = report_weighted_voting.Scheme("demo", "soft", "auc")

        scores = report_weighted_voting.ensemble_scores_for_frame(frame, validation_metrics, scheme)

        self.assertAlmostEqual(scores[0], 1.0 / 6.0)
        self.assertAlmostEqual(scores[1], 1.0)

    def test_weighted_hard_vote_uses_thresholded_components(self):
        validation_metrics = pd.DataFrame(
            {
                "search_name": ["ps4", "ps4"],
                "approach": ["left", "right"],
                "roc_auc": [0.7, 0.7],
                "pr_auc": [0.7, 0.7],
                "base_rate": [0.5, 0.5],
                "precision_at_25": [0.8, 0.8],
                "threshold": [0.6, 0.8],
                "threshold_count": [20, 20],
                "threshold_precision": [0.9, 0.9],
                "threshold_qualified": [True, True],
            }
        )
        frame = pd.DataFrame(
            {
                "SearchName": ["ps4", "ps4"],
                "score__left": [0.7, 0.2],
                "score__right": [0.7, 0.9],
            }
        )
        scheme = report_weighted_voting.Scheme("demo", "hard", "uniform")

        scores = report_weighted_voting.ensemble_scores_for_frame(frame, validation_metrics, scheme)

        self.assertEqual(scores.tolist(), [0.5, 0.5])

    def test_telegram_candidates_dedupe_items_across_model_passes(self):
        scored = pd.DataFrame(
            {
                "Dataid": ["123", "456"],
                "item_id": ["123", "456"],
                "SearchName": ["prada", "ps4"],
                "Title": ["Prada bag", "PS4 game"],
                "Price": [50.0, 40.0],
                "Link": ["https://example.com/123", "https://example.com/456"],
                "score__left": [0.9, 0.2],
                "threshold__left": [0.8, 0.8],
                "pass__left": [True, False],
                "score__right": [0.95, 0.91],
                "threshold__right": [0.7, 0.9],
                "pass__right": [True, True],
            }
        )
        records = [{"approach": "left"}, {"approach": "right"}]

        candidates = apply_to_live_collector.build_telegram_candidates(scored, records)

        self.assertEqual(candidates["TelegramItemKey"].tolist(), ["123", "456"])
        self.assertEqual(candidates.iloc[0]["GiantPassedModels"], "left, right")
        self.assertEqual(candidates.iloc[0]["GiantPassedModelCount"], 2)
        self.assertIn("Giant model pass", candidates.iloc[0]["RecommendationReason"])

    def test_title_feature_normalization_and_keyword_flags(self):
        series = pd.Series(["  NÚOVO bundle set 120€  ", "Authentic Prada da riparare"])

        out = title_features.build_title_feature_frame(series)

        self.assertEqual(out.loc[0, "TitleTextNormalized"], "nuovo bundle set 120")
        self.assertEqual(int(out.loc[0, "title_has_new_word_full"]), 1)
        self.assertEqual(int(out.loc[0, "title_has_bundle_word_full"]), 1)
        self.assertEqual(int(out.loc[0, "title_has_price_like_number_full"]), 1)
        self.assertEqual(int(out.loc[0, "title_keyword_positive_count_full"]), 1)
        self.assertEqual(int(out.loc[0, "title_keyword_caution_count_full"]), 2)
        self.assertEqual(int(out.loc[1, "title_has_auth_word_full"]), 1)
        self.assertEqual(int(out.loc[1, "title_has_defect_word_full"]), 1)

    def test_full_scrape_features_include_normalized_title_columns(self):
        frame = pd.DataFrame(
            {
                "Title": ["NÚOVO Prada bundle 120€"],
                "Description": ["Condizioni ottime"],
                "Brand": ["Prada"],
                "Condition": ["new"],
                "Location": ["Milano, Italia"],
                "Price": [120],
                "Likes": [4],
            }
        )

        out = full_scrape_features.add_full_scrape_features(frame)

        self.assertEqual(out.loc[0, "TitleTextNormalized"], "nuovo prada bundle 120")
        self.assertEqual(int(out.loc[0, "title_token_count_full_norm"]), 4)
        self.assertEqual(int(out.loc[0, "title_unique_token_count_full"]), 4)
        self.assertEqual(int(out.loc[0, "title_has_bundle_word_full"]), 1)
        self.assertIn("TitleTextNormalized", full_scrape_reranker.TEXT_FEATURES)
        self.assertIn("title_keyword_positive_count_full", full_scrape_reranker.BASE_NUMERIC_FEATURES)

    def test_full_scrape_enrichment_overlays_blank_description(self):
        scored = pd.DataFrame(
            {
                "tracking_key": ["nike:1", "nike:2"],
                "Description": ["", "already present"],
                "Condition": ["", "good"],
            }
        )
        enriched = pd.DataFrame(
            {
                "tracking_key": ["nike:1", "nike:2"],
                "Description": ["real full scrape description", ""],
                "Condition": ["new", ""],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            full_items = run_dir / "full_items"
            full_items.mkdir()
            enriched.to_csv(full_items / "items_enriched.csv", index=False)

            merged = full_scrape_reranker.merge_full_scrape_enrichment(scored, run_dir)

        self.assertEqual(merged.loc[0, "Description"], "real full scrape description")
        self.assertEqual(merged.loc[0, "Condition"], "new")
        self.assertEqual(merged.loc[1, "Description"], "already present")
        self.assertEqual(merged.loc[1, "Condition"], "good")


if __name__ == "__main__":
    unittest.main()
