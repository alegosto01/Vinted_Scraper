import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder import (
    dataset,
    model_sweep,
    modeling,
    paper_trade_model_benchmark,
    paper_trade_six_search_strict_hourly,
    paper_trading,
)
from experiments.basic_5_voting import run as basic_5_voting
from experiments.basic_5_stacking import run as basic_5_stacking


class DealFinderDatasetTests(unittest.TestCase):
    def test_normalize_id_series_removes_float_suffix(self):
        values = dataset.normalize_id_series(pd.Series([123.0, "456.0", " abc ", None]))
        self.assertEqual(values.tolist(), ["123", "456", "abc", ""])

    def test_dedupe_by_identity_keeps_first_by_search_date(self):
        df = pd.DataFrame(
            [
                {"Dataid": "1", "SearchDate": "02/01/2026 10:00:00", "Price": 20},
                {"Dataid": "1", "SearchDate": "01/01/2026 10:00:00", "Price": 10},
            ]
        )
        out = dataset.dedupe_by_identity(df, keep="first")
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out.iloc[0]["Price"]), 10.0)

    def test_build_search_dataset_labels_fast_sale_and_active_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            search_dir = Path(tmp) / "ps4"
            (search_dir / "eventual_sale_check").mkdir(parents=True)
            pd.DataFrame(
                [
                    {"Dataid": "1", "Link": "a", "SearchDate": "01/01/2026 10:00:00", "Price": 10},
                    {"Dataid": "2", "Link": "b", "SearchDate": "01/01/2026 10:00:00", "Price": 20},
                    {"Dataid": "3", "Link": "c", "SearchDate": "01/01/2026 10:00:00", "Price": 30},
                ]
            ).to_csv(search_dir / "big_raw.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "Dataid": "1",
                        "Link": "a",
                        "SearchDate": "01/01/2026 10:00:00",
                        "PriorityQueueEnqueuedAt": "2026-01-02T09:00:00+00:00",
                        "MarketStatus": "Sold",
                    }
                ]
            ).to_csv(search_dir / "sold_df.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "Dataid": "2",
                        "Link": "b",
                        "SearchDate": "01/01/2026 10:00:00",
                        "PriorityQueueLastCheckedAt": "2026-01-04T10:00:00+00:00",
                        "MarketStatus": "On Sale",
                    }
                ]
            ).to_csv(search_dir / "eventual_sale_check" / "not_sold_yet.csv", index=False)

            out = dataset.build_search_dataset(search_dir)
            out.index = out["Dataid"].astype(str)
            self.assertEqual(int(out.loc["1", "offline_sold_label"]), 1)
            self.assertTrue(bool(out.loc["1", "offline_label_eligible"]))
            self.assertEqual(int(out.loc["1", "fast_sale_2d"]), 1)
            self.assertTrue(bool(out.loc["1", "primary_label_eligible"]))
            self.assertEqual(int(out.loc["2", "offline_sold_label"]), 0)
            self.assertTrue(bool(out.loc["2", "offline_label_eligible"]))
            self.assertEqual(int(out.loc["2", "fast_sale_2d"]), 0)
            self.assertTrue(bool(out.loc["2", "primary_label_eligible"]))
            self.assertFalse(bool(out.loc["3", "offline_label_eligible"]))

    def test_safe_feature_columns_excludes_future_columns(self):
        df = pd.DataFrame(columns=["Price", "MarketStatus", "LastCheckStatus", "PriorityQueueEnqueuedAt", "fast_sale_2d", "offline_sold_label"])
        cols = dataset.safe_feature_columns(df)
        self.assertIn("Price", cols)
        self.assertNotIn("MarketStatus", cols)
        self.assertNotIn("LastCheckStatus", cols)
        self.assertNotIn("PriorityQueueEnqueuedAt", cols)
        self.assertNotIn("fast_sale_2d", cols)
        self.assertNotIn("offline_sold_label", cols)


class DealFinderModelingTests(unittest.TestCase):
    def test_time_split_uses_60_20_20(self):
        df = pd.DataFrame(
            {
                "_split_ts": pd.date_range("2026-01-01", periods=10, tz="UTC"),
                "fast_sale_2d": [0, 1] * 5,
            }
        )
        split = modeling.time_split(df)
        self.assertEqual((len(split.train), len(split.validation), len(split.test)), (6, 2, 2))

    def test_choose_threshold_respects_min_count_and_precision(self):
        y = pd.Series([1] * 8 + [0] * 2 + [0] * 10).to_numpy()
        scores = pd.Series([0.9] * 10 + [0.1] * 10).to_numpy()
        chosen = modeling.choose_threshold(y, scores, min_precision=0.6, min_count=10)
        self.assertEqual(chosen["count"], 10)
        self.assertAlmostEqual(chosen["precision"], 0.8)

    def test_promotion_rules_require_conservative_metrics(self):
        metrics = {
            "validation": {
                "base_rate": 0.2,
                "threshold": {"precision": 0.7, "count": 20},
                "precision_at": {"p@10": {"precision": 0.7}},
            },
            "test": {
                "base_rate": 0.2,
                "threshold": {"precision": 0.7, "count": 10},
                "precision_at": {"p@10": {"precision": 0.7}},
            },
        }
        ok, failures = modeling.promoted(metrics)
        self.assertTrue(ok)
        self.assertEqual(failures, [])

    def test_promotion_rule_allows_strong_high_base_rate_search(self):
        metrics = {
            "validation": {
                "base_rate": 0.7,
                "threshold": {"precision": 1.0, "count": 25},
                "precision_at": {"p@10": {"precision": 1.0}},
            },
            "test": {
                "base_rate": 0.7,
                "threshold": {"precision": 1.0, "count": 20},
                "precision_at": {"p@10": {"precision": 1.0}},
            },
        }
        ok, failures = modeling.promoted(metrics)
        self.assertTrue(ok)
        self.assertEqual(failures, [])


class DealFinderModelSweepTests(unittest.TestCase):
    def test_stratified_random_split_uses_60_20_20_and_preserves_classes(self):
        df = pd.DataFrame(
            {
                "item_id": [str(i) for i in range(100)],
                "offline_sold_label": [1] * 40 + [0] * 60,
            }
        )
        split = model_sweep.stratified_random_split(df, seed=42)
        self.assertEqual((len(split.train), len(split.validation), len(split.test)), (60, 20, 20))
        for part in (split.train, split.validation, split.test):
            self.assertEqual(set(part["offline_sold_label"].unique()), {0, 1})

    def test_promotion_80_gate(self):
        metrics = {
            "validation": {
                "threshold": {"precision": 0.8, "count": 20},
                "precision_at": {"p@10": {"precision": 0.8}},
            },
            "test": {
                "threshold": {"precision": 0.8, "count": 10},
                "precision_at": {"p@10": {"precision": 0.8}},
            },
        }
        ok, failures = model_sweep.promoted_80(metrics, live_ready=True, feature_leakage=False)
        self.assertTrue(ok)
        self.assertEqual(failures, [])

        metrics["test"]["threshold"]["precision"] = 0.79
        ok, failures = model_sweep.promoted_80(metrics, live_ready=True, feature_leakage=False)
        self.assertFalse(ok)
        self.assertIn("test threshold precision >= 80%", failures)

    def test_sweep_leakage_feature_detection(self):
        self.assertTrue(model_sweep.has_leakage_features(["Price", "MarketStatus"]))
        self.assertTrue(model_sweep.has_leakage_features(["LocalPrimaryImagePath"]))
        self.assertFalse(model_sweep.has_leakage_features(["price_num", "image_sharpness"]))

    def test_visual_sample_is_bounded_and_keeps_classes(self):
        df = pd.DataFrame(
            {
                "item_id": [str(i) for i in range(100)],
                "offline_sold_label": [1] * 30 + [0] * 70,
            }
        )
        sampled = model_sweep.stratified_sample_frame(df, max_rows=20, seed=42)
        self.assertEqual(len(sampled), 20)
        self.assertEqual(set(sampled["offline_sold_label"].unique()), {0, 1})

    def test_basic_image_feature_extraction(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.png"
            Image.new("RGB", (20, 10), color=(120, 80, 40)).save(path)
            features = model_sweep.compute_basic_image_features(path)
            self.assertEqual(features["image_width"], 20.0)
            self.assertEqual(features["image_height"], 10.0)
            self.assertAlmostEqual(features["image_aspect_ratio"], 2.0)
            self.assertIn("image_brightness", features)


class Basic5VotingTests(unittest.TestCase):
    def test_accuracy_threshold_uses_validation_cutoff_with_best_accuracy(self):
        labels = pd.Series([1, 1, 0, 0]).to_numpy()
        scores = pd.Series([0.9, 0.8, 0.7, 0.1]).to_numpy()

        threshold, accuracy = basic_5_voting.tune_accuracy_threshold(labels, scores)

        self.assertEqual(threshold, 0.8)
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(basic_5_voting.threshold_accuracy(labels, scores, threshold), 1.0)


class Basic5StackingTests(unittest.TestCase):
    def test_top_components_and_score_sum_use_validation_scores(self):
        frame = pd.DataFrame(
            {
                "offline_sold_label": [1, 1, 0, 0],
                "score__strong": [0.9, 0.8, 0.2, 0.1],
                "score__weak": [0.1, 0.8, 0.7, 0.2],
            }
        )

        self.assertEqual(basic_5_stacking.select_top_approaches(frame, ("weak", "strong"), 1), ("strong",))
        self.assertEqual(
            basic_5_stacking.component_score(frame, ("strong", "weak"), reducer="sum").tolist(),
            [1.0, 1.6, 0.8999999999999999, 0.30000000000000004],
        )


class DealFinderPaperTradingTests(unittest.TestCase):
    def test_threshold_override_updates_metadata_without_mutating_original(self):
        original = {"threshold": 0.9656, "artifact_path": "model.pkl"}
        updated = paper_trading.apply_threshold_override(original, "gucci", {"gucci": 0.92})
        self.assertEqual(original["threshold"], 0.9656)
        self.assertEqual(updated["offline_threshold"], 0.9656)
        self.assertEqual(updated["threshold"], 0.92)
        self.assertEqual(updated["threshold_override_source"], "threshold_overrides.json")

    def test_load_threshold_overrides_accepts_search_threshold_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "threshold_overrides.json").write_text(
                '{"threshold_overrides": {"gucci": 0.92, "bad": 2.0, "skip": "x"}}',
                encoding="utf-8",
            )
            self.assertEqual(paper_trading.load_threshold_overrides(root), {"gucci": 0.92})

    def test_outcome_window_columns_include_fast_checkpoints(self):
        df = pd.DataFrame([{"item_id": "1"}])
        out = paper_trading.ensure_outcome_columns(df)
        for col in [
            "sold_within_2h",
            "evaluated_2h_at",
            "status_at_2h",
            "sold_within_12h",
            "evaluated_12h_at",
            "status_at_12h",
            "sold_within_2d",
            "sold_within_7d",
        ]:
            self.assertIn(col, out.columns)

    def test_ensure_outcome_columns_allows_timestamp_assignment_after_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracked.csv"
            pd.DataFrame(
                [
                    {
                        "item_id": "1",
                        "last_rechecked_at": pd.NA,
                        "evaluated_2h_at": pd.NA,
                    }
                ]
            ).to_csv(path, index=False)
            out = paper_trading.ensure_outcome_columns(pd.read_csv(path))
            out.at[0, "last_rechecked_at"] = "2026-01-01T03:00:00+00:00"
            out.at[0, "evaluated_2h_at"] = "2026-01-01T03:00:00+00:00"
            self.assertEqual(out.at[0, "last_rechecked_at"], "2026-01-01T03:00:00+00:00")

    def test_update_outcome_windows_marks_hourly_checkpoint_negative(self):
        tracked = paper_trading.ensure_outcome_columns(
            pd.DataFrame([{"item_id": "1"}])
        )
        first = pd.Timestamp("2026-01-01T00:00:00Z")
        recheck = pd.Timestamp("2026-01-01T03:00:00Z")
        paper_trading.update_outcome_windows(
            tracked,
            0,
            status="On Sale",
            first_ts=first,
            recheck_ts=recheck,
        )
        self.assertFalse(bool(tracked.at[0, "sold_within_2h"]))
        self.assertTrue(pd.isna(tracked.at[0, "sold_within_12h"]))

    def test_recheck_dry_run_writes_only_live_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "deal_finder"
            live = root / "live_runs" / "paper_test"
            live.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "item_id": "1",
                        "Dataid": "1",
                        "Link": "https://example.test/item/1",
                        "first_tracked_at": "2026-01-01T00:00:00+00:00",
                        "last_rechecked_at": "",
                    }
                ]
            ).to_csv(live / "tracked_items.csv", index=False)
            with patch("experiments.deal_finder.paper_trading.LIVE_RUNS_DIR", root / "live_runs"), patch(
                "experiments.deal_finder.paths.EXPERIMENT_ROOT", root
            ):
                result = paper_trading.recheck_due(live_run=live, dry_run=True)
            self.assertEqual(result["due_count"], 1)
            self.assertTrue((live / "recheck_plan.json").exists())

    def test_recheck_above_threshold_only_filters_due_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "deal_finder"
            live = root / "live_runs" / "paper_test"
            live.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "item_id": "1",
                        "Dataid": "1",
                        "Link": "https://example.test/item/1",
                        "above_threshold": True,
                        "first_tracked_at": "2026-01-01T00:00:00+00:00",
                        "last_rechecked_at": "",
                    },
                    {
                        "item_id": "2",
                        "Dataid": "2",
                        "Link": "https://example.test/item/2",
                        "above_threshold": False,
                        "first_tracked_at": "2026-01-01T00:00:00+00:00",
                        "last_rechecked_at": "",
                    },
                ]
            ).to_csv(live / "tracked_items.csv", index=False)
            with patch("experiments.deal_finder.paper_trading.LIVE_RUNS_DIR", root / "live_runs"), patch(
                "experiments.deal_finder.paths.EXPERIMENT_ROOT", root
            ):
                result = paper_trading.recheck_due(live_run=live, due_hours=1, above_threshold_only=True, dry_run=True)
            self.assertEqual(result["due_count"], 1)


class DealFinderModelBenchmarkTests(unittest.TestCase):
    def test_threshold_variants_create_three_ordered_thresholds(self):
        variants = paper_trade_model_benchmark.threshold_variants(0.9656)
        self.assertEqual([row["threshold_label"] for row in variants], ["strict", "medium", "loose"])
        self.assertEqual(variants[0]["threshold"], 0.9656)
        self.assertLess(variants[1]["threshold"], variants[0]["threshold"])
        self.assertLess(variants[2]["threshold"], variants[1]["threshold"])

    def test_image_model_is_skipped_when_live_image_features_are_missing(self):
        candidates = pd.DataFrame([{"Dataid": "1", "Link": "https://example.test/items/1", "Price": 10}])
        metadata = {
            "search_name": "gucci",
            "approach": "visual_basic_v1",
            "requires_images": True,
            "numeric_features": ["price_num", "image_brightness"],
            "artifact_path": "model.pkl",
            "metadata_path": "metadata.json",
        }
        self.assertEqual(paper_trade_model_benchmark.image_feature_status(candidates, metadata), "missing")

    def test_image_model_is_available_when_live_image_features_exist(self):
        candidates = pd.DataFrame([{"Dataid": "1", "image_brightness": 0.5, "image_contrast": 0.1}])
        metadata = {
            "requires_images": True,
            "numeric_features": ["price_num", "image_brightness", "image_contrast"],
        }
        self.assertEqual(paper_trade_model_benchmark.image_feature_status(candidates, metadata), "available")

    def test_primary_image_url_reads_first_snapshot_image(self):
        row = pd.Series({"Images": "https://images.example.test/a.webp"})
        self.assertEqual(paper_trade_model_benchmark.primary_image_url(row), "https://images.example.test/a.webp")

    def test_benchmark_key_is_model_and_threshold_specific(self):
        frame = pd.DataFrame(
            [
                {"SearchName": "nike", "item_id": "1", "approach": "a", "threshold_label": "strict"},
                {"SearchName": "nike", "item_id": "1", "approach": "a", "threshold_label": "loose"},
            ]
        )
        keys = paper_trade_model_benchmark.make_benchmark_key(frame).tolist()
        self.assertNotEqual(keys[0], keys[1])

    def test_blank_category_is_normalized_without_editing_original_config(self):
        config = SimpleNamespace(search="Nike", category="", prezzoDa="", prezzoA="", condition="", colore="", brands="")
        normalized = paper_trade_model_benchmark.normalized_search_config(config)
        self.assertEqual(config.category, "")
        self.assertEqual(normalized.category, " ")
        self.assertTrue(paper_trade_model_benchmark.has_collectable_search_settings(config))

    def test_empty_synthetic_config_is_not_collectable(self):
        config = SimpleNamespace(search="", category="", prezzoDa="", prezzoA="", condition="", colore="", brands="")
        self.assertFalse(paper_trade_model_benchmark.has_collectable_search_settings(config))

    def test_benchmark_recheck_does_not_check_new_rows_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "deal_finder"
            live = root / "live_runs" / "benchmark_test"
            live.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "item_id": "1",
                        "Dataid": "1",
                        "Link": "https://example.test/item/1",
                        "first_tracked_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        "last_rechecked_at": "",
                        "SearchName": "nike",
                        "approach": "logistic",
                        "threshold_label": "strict",
                    }
                ]
            ).to_csv(live / paper_trade_model_benchmark.BENCHMARK_TRACKED_FILE, index=False)
            with patch("experiments.deal_finder.paths.EXPERIMENT_ROOT", root):
                result = paper_trade_model_benchmark.recheck_benchmark_due(out_dir=live, dry_run=True)
            self.assertEqual(result["due_rows"], 0)


class DealFinderSixSearchStrictHourlyTests(unittest.TestCase):
    def test_due_recheck_mask_waits_one_hour_for_new_rows(self):
        tracked = pd.DataFrame(
            [
                {
                    "tracking_key": "nike::1",
                    "SearchName": "nike",
                    "item_id": "1",
                    "first_seen_at": "2026-01-01T00:30:00+00:00",
                    "last_rechecked_at": pd.NA,
                    "sold_at": pd.NA,
                },
                {
                    "tracking_key": "nike::2",
                    "SearchName": "nike",
                    "item_id": "2",
                    "first_seen_at": "2026-01-01T00:00:00+00:00",
                    "last_rechecked_at": pd.NA,
                    "sold_at": pd.NA,
                },
            ]
        )
        now = pd.Timestamp("2026-01-01T01:00:00Z")
        due = paper_trade_six_search_strict_hourly.due_recheck_mask(tracked, now, 1.0)
        self.assertEqual(due.tolist(), [False, True])

    def test_merge_snapshot_selection_updates_current_state(self):
        tracked = pd.DataFrame(
            [
                {
                    "tracking_key": "gucci::1",
                    "SearchName": "gucci",
                    "item_id": "1",
                    "max_score": 0.91,
                    "score_observations": 1,
                    "times_above_threshold": 1,
                }
            ]
        )
        selected = pd.DataFrame(
            [
                {
                    "SearchName": "gucci",
                    "item_id": "1",
                    "Link": "https://example.test/item/1",
                    "Title": "Gucci Item",
                    "Brand": "Gucci",
                    "Size": "M",
                    "Price": 100,
                    "Likes": 2,
                    "MarketStatus": "On Sale",
                    "model_probability": 0.95,
                    "model_threshold": 0.93,
                    "model_version": "gucci_model",
                }
            ]
        )
        updated, history = paper_trade_six_search_strict_hourly.merge_snapshot_selection(
            tracked,
            selected,
            observed_at="2026-01-01T02:00:00+00:00",
            snapshot_path=Path("/tmp/scored.csv"),
        )
        self.assertEqual(len(updated), 1)
        self.assertEqual(float(updated.iloc[0]["current_score"]), 0.95)
        self.assertEqual(int(updated.iloc[0]["score_observations"]), 2)
        self.assertEqual(int(updated.iloc[0]["times_above_threshold"]), 2)
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["event_type"], "threshold_hit")


if __name__ == "__main__":
    unittest.main()
