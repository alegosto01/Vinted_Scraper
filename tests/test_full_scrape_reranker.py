import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.full_scrape_reranker import full_scrape_reranker as reranker


class FullScrapeRerankerTests(unittest.TestCase):
    def test_merge_full_scrape_enrichment_backfills_blank_description(self):
        scored = pd.DataFrame(
            [
                {
                    "tracking_key": "ps4::1",
                    "SearchName": "ps4",
                    "Description": "   ",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            live_run_dir = Path(tmp) / "collector"
            (live_run_dir / "full_items").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "tracking_key": "ps4::1",
                        "Description": "Full scrape description",
                        "Condition": "good",
                    }
                ]
            ).to_csv(live_run_dir / "full_items" / "items_enriched.csv", index=False)

            merged = reranker.merge_full_scrape_enrichment(scored, live_run_dir)

        self.assertEqual(merged.loc[0, "Description"], "Full scrape description")
        self.assertEqual(merged.loc[0, "Condition"], "good")

    def test_build_training_frame_resolves_collector_from_scoring_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector_dir = root / "bin_collector_20260602_214104"
            (collector_dir / "full_items").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "tracking_key": "ps4::1",
                        "Description": "Full scrape description",
                        "Condition": "good",
                        "Upload_date": "2 hours",
                        "Interested_count": 4,
                        "View_count": 12,
                        "SellerName": "seller",
                        "SellerId": "seller-1",
                        "Location": "Rome, Italy",
                        "ReviewsCount": 8,
                        "Stars": 4.9,
                        "VisiblePictureCount": 3,
                        "HiddenPictureCount": 1,
                        "PictureCount": 4,
                    }
                ]
            ).to_csv(collector_dir / "full_items" / "items_enriched.csv", index=False)

            scoring_dir = root / "live_scoring_20260611_191506"
            scoring_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "tracking_key": "ps4::1",
                        "SearchName": "ps4",
                        "Title": "PS4 Slim",
                        "Price": 120.0,
                        "Likes": 7,
                        "sold_within_72h": 1,
                    }
                ]
            ).to_csv(scoring_dir / "live_scored_items.csv", index=False)
            (scoring_dir / "manifest.json").write_text(
                json.dumps({"live_run_dir": str(collector_dir)}),
                encoding="utf-8",
            )

            frame = reranker.build_training_frame(
                live_run_dir=scoring_dir,
                scoring_dir=scoring_dir,
                searches=("ps4",),
                horizon_hours=72,
            )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "Description"], "Full scrape description")
        self.assertEqual(frame.loc[0, "DescriptionText"], "Full scrape description")
        self.assertEqual(frame.loc[0, "Condition"], "good")
        self.assertEqual(frame.loc[0, "LocationCountry"], "Italy")


if __name__ == "__main__":
    unittest.main()
