import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.basic_5_giant_model import apply_to_live_collector as basic5_live
from experiments.current.basic_5_giant_model import run as basic5_run
from experiments.old.deal_finder.modeling import TARGET_COL
from experiments.current.giant_basic_visual import apply_to_live_collector as visual_live
from experiments.current.giant_basic_visual import apply_live_trained_to_live_collector as visual_live_trained
from experiments.current.giant_basic_visual import features as visual_features
from experiments.current.giant_basic_visual import live_shap_likes_report
from experiments.current.giant_basic_visual import run as visual_run


def make_image(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (40, 30), color).save(path)
    return path


class GiantBasicVisualTests(unittest.TestCase):
    def test_rows_without_primary_image_are_skipped(self):
        frame = pd.DataFrame(
            {
                "SearchName": ["ps4"],
                "item_id": ["1"],
                "LocalPrimaryImagePath": [""],
                "LocalImagePaths": [""],
            }
        )

        kept, skipped = visual_features.filter_rows_with_main_image(frame)

        self.assertEqual(len(kept), 0)
        self.assertEqual(skipped["skip_reason"].tolist(), ["missing_main_image"])

    def test_unreadable_primary_image_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "bad.jpg"
            bad_path.write_text("not an image", encoding="utf-8")
            frame = pd.DataFrame(
                {
                    "SearchName": ["ps4"],
                    "item_id": ["1"],
                    "LocalPrimaryImagePath": [str(bad_path)],
                }
            )

            kept, skipped = visual_features.filter_rows_with_main_image(frame)

        self.assertEqual(len(kept), 0)
        self.assertEqual(len(skipped), 1)
        self.assertTrue(str(skipped.iloc[0]["skip_reason"]).startswith("main_image_unreadable"))

    def test_local_image_paths_is_never_used_as_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            fallback = make_image(Path(tmp) / "fallback.jpg", (255, 255, 255))
            frame = pd.DataFrame(
                {
                    "SearchName": ["ps4"],
                    "item_id": ["1"],
                    "LocalPrimaryImagePath": [""],
                    "LocalImagePaths": [str(fallback)],
                }
            )

            kept, skipped = visual_features.filter_rows_with_main_image(frame)

        self.assertEqual(len(kept), 0)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped.iloc[0]["skip_reason"], "missing_main_image")

    def test_secondary_images_do_not_affect_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = make_image(Path(tmp) / "primary.jpg", (255, 255, 255))
            secondary = make_image(Path(tmp) / "secondary.jpg", (0, 0, 0))
            frame = pd.DataFrame(
                {
                    "SearchName": ["ps4"],
                    "item_id": ["1"],
                    "LocalPrimaryImagePath": [str(primary)],
                    "LocalImagePaths": [str(secondary)],
                }
            )

            kept, skipped = visual_features.filter_rows_with_main_image(frame)

        self.assertTrue(skipped.empty)
        self.assertEqual(len(kept), 1)
        self.assertGreater(float(kept.iloc[0]["MainImageBrightness"]), 0.95)

    def test_raw_dino_dimensions_are_excluded(self):
        for mode in visual_features.FEATURE_MODES:
            cols = visual_features.feature_columns_for_mode(mode)
            self.assertNotIn("DinoEmbedding", cols)
            self.assertNotIn("DinoEmbeddingDim", cols)
            self.assertFalse(any(col.startswith("DinoEmbedding_") for col in cols))

    def test_dino_outlier_mode_is_diagnostic_not_live_ready(self):
        self.assertTrue(visual_features.is_diagnostic_mode("main_image_dino_outlier_diagnostic"))
        self.assertFalse(visual_features.mode_live_ready("main_image_dino_outlier_diagnostic"))

    def test_live_loader_clamps_diagnostic_mode_to_not_live_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            offline_root = tmp_path / "offline_runs"
            model_root = tmp_path / "models"
            run_dir = offline_root / "giant_basic_visual_fake"
            run_dir.mkdir(parents=True)
            model_root.mkdir()
            metadata_path = model_root / "fake_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "feature_mode": "main_image_dino_outlier_diagnostic",
                        "threshold": 0.5,
                        "live_ready": True,
                        "diagnostic": False,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "status": "trained",
                        "feature_mode": "main_image_dino_outlier_diagnostic",
                        "approach": "demo",
                        "metadata_path": str(metadata_path),
                        "artifact_path": str(model_root / "fake.pkl"),
                        "threshold": 0.5,
                    }
                ]
            ).to_csv(run_dir / "metrics_long.csv", index=False)

            old_offline = visual_live.OFFLINE_RUNS_DIR
            old_models = visual_live.MODELS_DIR
            try:
                visual_live.OFFLINE_RUNS_DIR = offline_root
                visual_live.MODELS_DIR = model_root
                records = visual_live.load_visual_model_records(
                    offline_run="giant_basic_visual_fake",
                    modes=("main_image_dino_outlier_diagnostic",),
                    approaches=("demo",),
                )
            finally:
                visual_live.OFFLINE_RUNS_DIR = old_offline
                visual_live.MODELS_DIR = old_models

        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["live_ready"])
        self.assertTrue(records[0]["diagnostic"])

    def test_feature_modes_use_same_filtered_row_set_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for search in ("gucci", "ps4"):
                for label in (0, 1):
                    for idx in range(20):
                        image_path = make_image(Path(tmp) / f"{search}_{label}_{idx}.jpg", (idx * 5, 100, 200))
                        rows.append(
                            {
                                "SearchName": search,
                                "item_id": f"{search}-{label}-{idx}",
                                "LocalPrimaryImagePath": str(image_path),
                                TARGET_COL: label,
                            }
                        )
            frame = pd.DataFrame(rows)
            filtered, _ = visual_features.filter_rows_with_main_image(frame)
            filtered, search_cols = basic5_run.add_search_onehot(filtered, ["gucci", "ps4"])
            splits = basic5_run.stratified_global_split(filtered, seed=42)
            spec = next(item for item in basic5_run.available_basic5_specs() if item.name == "random_forest_basic_v1")

            observed_ids = {
                mode: set(splits.test["item_id"])
                for mode in visual_features.FEATURE_MODES
                if visual_run.select_features(splits.train, spec, search_cols, mode)
            }

        self.assertEqual(len(observed_ids), len(visual_features.FEATURE_MODES))
        self.assertEqual(len({frozenset(value) for value in observed_ids.values()}), 1)

    def test_live_shadow_has_no_send_telegram_path(self):
        source = inspect.getsource(visual_live)

        self.assertNotIn("--send-telegram", source)
        self.assertNotIn("send_candidates_to_telegram", source)
        self.assertNotIn("append_telegram_sent_rows", source)
        self.assertNotIn("telegram_sent_items.csv", source)

    def test_current_basic5_telegram_policy_remains_unchanged(self):
        self.assertEqual(basic5_live.TELEGRAM_MIN_PRICE_EUR, 30.0)
        self.assertEqual(
            basic5_live.TELEGRAM_POLICY_DESCRIPTION,
            "any normal giant-model pass plus price > 30 EUR",
        )

    def test_live_trained_scoring_uses_model_feature_names(self):
        class FakeModel:
            feature_names_in_ = ["Price", "Likes", "search__nike", "search__ps4"]

        frame = pd.DataFrame(
            {
                "Price": [10.0],
                "Likes": [2.0],
                "search__nike": [1.0],
                "search__donna_accessori_gioielli": [1.0],
            }
        )

        aligned = visual_live_trained.align_feature_frame_to_model(frame, FakeModel())

        self.assertEqual(aligned.columns.tolist(), ["Price", "Likes", "search__nike", "search__ps4"])
        self.assertEqual(float(aligned.iloc[0]["search__ps4"]), 0.0)

    def test_live_trained_telegram_candidates_include_model_metadata(self):
        scored = pd.DataFrame(
            {
                "SearchName": ["nike"],
                "item_id": ["1"],
                "Title": ["Nike shoes"],
                "Link": ["https://www.vinted.it/items/1"],
                "Price": [50.0],
                visual_live_trained.SCORE_COL: [0.64321],
                visual_live_trained.THRESHOLD_COL: [0.58],
                visual_live_trained.PASS_COL: [True],
            }
        )

        candidates = visual_live_trained.build_telegram_candidates(scored)

        self.assertEqual(len(candidates), 1)
        row = candidates.iloc[0]
        self.assertEqual(row["TelegramModel"], visual_live_trained.MODEL_DISPLAY_NAME)
        self.assertAlmostEqual(float(row["TelegramModelScore"]), 0.64321)
        self.assertAlmostEqual(float(row["TelegramAdjustedThreshold"]), 0.58)
        self.assertAlmostEqual(float(row["GiantBestMargin"]), 0.06321)
        self.assertIn("score 0.643", row["RecommendationReason"])

    def test_live_trained_failed_candidate_can_become_eligible_later(self):
        base = {
            "SearchName": "nike",
            "item_id": "1",
            "Link": "https://www.vinted.it/items/1",
            "Price": 50.0,
            visual_live_trained.THRESHOLD_COL: 0.58,
        }
        failed = pd.DataFrame([{**base, visual_live_trained.SCORE_COL: 0.57, visual_live_trained.PASS_COL: False}])
        passed = pd.DataFrame([{**base, visual_live_trained.SCORE_COL: 0.59, visual_live_trained.PASS_COL: True}])

        self.assertTrue(visual_live_trained.build_telegram_candidates(failed).empty)
        self.assertEqual(len(visual_live_trained.build_telegram_candidates(passed)), 1)

    def test_prepare_candidates_full_scrapes_only_new_missing_rows(self):
        candidates = pd.DataFrame(
            {
                "TelegramItemKey": ["1"],
                "SearchName": ["nike"],
                "item_id": ["1"],
                "Link": ["https://www.vinted.it/items/1"],
                "Title": ["Nike shoes"],
                "Price": [50.0],
                "Description": [""],
                "Upload_date": [""],
                "SellerName": [""],
                "ReviewsCount": [pd.NA],
                "Stars": [pd.NA],
            }
        )
        successes = pd.DataFrame(
            {
                "item_id": ["1"],
                "Description": ["Full description"],
                "Upload_date": ["today"],
                "SellerName": ["seller"],
                "ReviewsCount": [12],
                "Stars": [4.8],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                visual_live_trained,
                "collect_send_full_scrape_payloads",
                return_value=(successes, pd.DataFrame()),
            ) as collect_mock, mock.patch.object(
                visual_live_trained,
                "write_send_full_scrape_outputs",
                return_value={"requested": 1, "succeeded": 1, "failed": 0},
            ):
                enriched, summary = visual_live_trained.prepare_candidates_for_telegram_send(
                    candidates,
                    out_dir=Path(tmp),
                    sent_log_path=Path(tmp) / "sent.csv",
                )

        collect_mock.assert_called_once()
        self.assertEqual(summary["status"], "attempted")
        self.assertEqual(enriched.loc[0, "Description"], "Full description")
        self.assertEqual(enriched.loc[0, "SellerName"], "seller")

    def test_prepare_candidates_does_not_full_scrape_already_sent_rows(self):
        candidates = pd.DataFrame(
            {
                "TelegramItemKey": ["1"],
                "SearchName": ["nike"],
                "item_id": ["1"],
                "Link": ["https://www.vinted.it/items/1"],
                "Description": [""],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            sent_log = Path(tmp) / "sent.csv"
            pd.DataFrame({"item_key": ["1"]}).to_csv(sent_log, index=False)
            with mock.patch.object(visual_live_trained, "collect_send_full_scrape_payloads") as collect_mock:
                _enriched, summary = visual_live_trained.prepare_candidates_for_telegram_send(
                    candidates,
                    out_dir=Path(tmp),
                    sent_log_path=sent_log,
                )

        collect_mock.assert_not_called()
        self.assertEqual(summary["requested"], 0)

    def test_prepare_candidates_keeps_row_when_full_scrape_fails(self):
        candidates = pd.DataFrame(
            {
                "TelegramItemKey": ["1"],
                "SearchName": ["nike"],
                "item_id": ["1"],
                "Link": ["https://www.vinted.it/items/1"],
                "Description": [""],
            }
        )
        failures = pd.DataFrame({"item_id": ["1"], "FullScrapeStatus": ["FetchFailed"]})

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                visual_live_trained,
                "collect_send_full_scrape_payloads",
                return_value=(pd.DataFrame(), failures),
            ), mock.patch.object(
                visual_live_trained,
                "write_send_full_scrape_outputs",
                return_value={"requested": 1, "succeeded": 0, "failed": 1},
            ):
                enriched, summary = visual_live_trained.prepare_candidates_for_telegram_send(
                    candidates,
                    out_dir=Path(tmp),
                    sent_log_path=Path(tmp) / "sent.csv",
                )

        self.assertEqual(len(enriched), 1)
        self.assertEqual(summary["status"], "attempted")
        self.assertEqual(summary["failed"], 1)

    def test_likes_summary_reports_rank_and_effect(self):
        importance = live_shap_likes_report.feature_importance_frame(
            values=np.array([[2.0, 0.1], [-1.0, 0.2]]),
            feature_names=["Likes", "Price"],
        )

        summary = live_shap_likes_report.likes_summary(importance)

        self.assertTrue(summary["present"])
        self.assertEqual(summary["rank"], 1)
        self.assertAlmostEqual(summary["mean_abs_shap"], 1.5)


if __name__ == "__main__":
    unittest.main()
