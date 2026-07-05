from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from experiments.old.benchmark_basic_to_full.report_live_score_distributions import (
    label_live_cohort,
    unique_tracked_items,
)


class LiveScoreDistributionReportTests(unittest.TestCase):
    def test_unique_tracked_items_keeps_latest_identity_row(self):
        tracked = pd.DataFrame(
            [
                {
                    "tracking_key": "ps4::1",
                    "SearchName": "ps4",
                    "item_id": "1",
                    "last_seen_at": "2026-05-18T00:00:00+00:00",
                    "Stage1Score": 0.81,
                },
                {
                    "tracking_key": "ps4::1",
                    "SearchName": "ps4",
                    "item_id": "1",
                    "last_seen_at": "2026-05-18T01:00:00+00:00",
                    "Stage1Score": 0.93,
                },
            ]
        )

        unique = unique_tracked_items(tracked)

        self.assertEqual(len(unique), 1)
        self.assertEqual(float(unique.iloc[0]["Stage1Score"]), 0.93)

    def test_checkpoint_cohort_uses_only_evaluated_items(self):
        tracked = pd.DataFrame(
            [
                {"item_id": "sold", "evaluated_at_24h": "2026-05-19T00:00:00Z", "sold_within_24h": "True"},
                {"item_id": "unsold", "evaluated_at_24h": "2026-05-19T00:00:00Z", "sold_within_24h": "False"},
                {"item_id": "young", "evaluated_at_24h": pd.NA, "sold_within_24h": pd.NA},
            ]
        )

        cohort, meta = label_live_cohort(tracked, window_hours=24)

        self.assertEqual(cohort["item_id"].tolist(), ["sold", "unsold"])
        self.assertEqual(cohort["SoldLabel"].tolist(), [1, 0])
        self.assertEqual(meta["sold_name"], "sold within 24h")
        self.assertEqual(meta["unsold_name"], "not sold by 24h")

    def test_all_observed_cohort_uses_detected_sold_timestamp(self):
        tracked = pd.DataFrame(
            [
                {"item_id": "sold", "sold_at": "2026-05-19T00:00:00Z"},
                {"item_id": "open", "sold_at": pd.NA},
            ]
        )

        cohort, meta = label_live_cohort(tracked, window_hours=None)

        self.assertEqual(cohort["SoldLabel"].tolist(), [1, 0])
        self.assertEqual(meta["sold_name"], "detected sold")
        self.assertEqual(meta["unsold_name"], "not sold yet")


if __name__ == "__main__":
    unittest.main()
