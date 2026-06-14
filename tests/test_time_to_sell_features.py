import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.time_to_sell import build_speed_datasets as datasets
from experiments.current.time_to_sell import live_bin_collector as collector
from experiments.current.time_to_sell import train_speed_models
from experiments.current.time_to_sell.text_features import add_title_features


class TimeToSellTextFeatureTests(unittest.TestCase):
    def test_full_scrape_success_merges_description_into_tracked_state(self):
        tracked = pd.DataFrame(
            {
                "SearchName": ["ps4"],
                "item_id": ["123"],
                "Title": ["PS4 slim bundle"],
                "Description": [""],
            }
        )
        selected = tracked[["SearchName", "item_id", "Title"]].copy()
        successes = pd.DataFrame(
            {
                "SearchName": ["ps4"],
                "item_id": ["123"],
                "Description": ["Console perfetta con due controller."],
                "Condition": ["Ottime condizioni"],
                "View_count": [42],
                "FullScrapedAt": ["2026-06-14T08:00:00Z"],
            }
        )

        out = collector.update_full_scrape_state(
            tracked,
            selected_for_search=selected,
            successes=successes,
            failures=pd.DataFrame(),
            visual_path=None,
            finished_at="2026-06-14T08:01:00Z",
        )

        self.assertEqual(out.loc[0, "FullScrapeStatus"], "success")
        self.assertEqual(out.loc[0, "Description"], "Console perfetta con due controller.")
        self.assertEqual(out.loc[0, "Condition"], "Ottime condizioni")
        self.assertEqual(int(out.loc[0, "View_count"]), 42)

    def test_speed_dataset_backfills_description_and_title_features_from_enrichment(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "full_items").mkdir()
            tracked = pd.DataFrame(
                {
                    "SearchName": ["prada"],
                    "item_id": ["456"],
                    "Title": ["NÚOVA Prada borsa 120€"],
                    "Brand": ["Prada"],
                    "Size": ["Unica"],
                    "Price": [120],
                    "Likes": [5],
                    "sold_at": ["2026-06-14T12:00:00Z"],
                    "first_stage1_pass_at": ["2026-06-14T08:00:00Z"],
                    "sold_within_72h": [True],
                    "evaluated_at_72h": ["2026-06-17T08:00:00Z"],
                }
            )
            enriched = pd.DataFrame(
                {
                    "SearchName": ["prada"],
                    "item_id": ["456"],
                    "Description": ["Pelle in ottime condizioni con dust bag."],
                    "Condition": ["Ottimo"],
                    "FullScrapedAt": ["2026-06-14T08:10:00Z"],
                }
            )
            enriched.to_csv(run_dir / "full_items" / "items_enriched.csv", index=False)

            labels = datasets.prepare_live_labels(tracked, [72])
            labels = add_title_features(datasets.merge_full_enrichment(labels, run_dir))
            basic5 = datasets.build_basic5(labels, [72])

        self.assertEqual(basic5.loc[0, "Description"], "Pelle in ottime condizioni con dust bag.")
        self.assertEqual(basic5.loc[0, "TitleTextNormalized"], "nuova prada borsa 120")
        self.assertEqual(int(basic5.loc[0, "title_has_new_word_tts"]), 1)
        self.assertEqual(int(basic5.loc[0, "title_has_price_like_number_tts"]), 1)

    def test_speed_model_feature_selection_keeps_description_and_title_features(self):
        frame = add_title_features(
            pd.DataFrame(
                {
                    "Title": ["NÚOVA Prada borsa 120€", "PS4 slim"],
                    "Brand": ["Prada", "Sony"],
                    "Size": ["Unica", ""],
                    "Description": ["Pelle ottima", "Con controller"],
                    "Price": [120, 80],
                    "Likes": [5, 2],
                }
            )
        )
        spec = next(spec for spec in train_speed_models.approach_specs() if spec.name == "sgd_text_numeric_v1")

        numeric, text = train_speed_models.feature_columns_for_mode(
            frame,
            mode="basic5",
            spec=spec,
            embedding_cols=[],
            include_upload_date=False,
        )

        self.assertIn("title_token_count_tts", numeric)
        self.assertIn("title_keyword_caution_count_tts", numeric)
        self.assertIn("TitleTextNormalized", text)
        self.assertIn("Description", text)


if __name__ == "__main__":
    unittest.main()
