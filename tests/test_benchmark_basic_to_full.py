from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from experiments.benchmark_basic_to_full.cascade_runner import bool_series, merge_tracked


class CascadeTrackedStateTests(unittest.TestCase):
    def test_bool_series_counts_csv_float_booleans(self):
        parsed = bool_series(pd.Series(["True", "1", "1.0", "yes", "0.0", "", np.nan]))

        self.assertEqual(parsed.tolist(), [True, True, True, True, False, False, False])

    def test_merge_tracked_accepts_boolean_after_csv_float_state(self):
        existing = pd.DataFrame(
            [
                {
                    "tracking_key": "ps4::123",
                    "item_id": "123",
                    "SearchName": "ps4",
                    "Stage2Passed": np.nan,
                }
            ]
        )
        updates = pd.DataFrame(
            [
                {
                    "item_id": "123",
                    "SearchName": "ps4",
                    "Stage2Passed": False,
                    "Stage2Status": "scored",
                }
            ]
        )

        merged = merge_tracked(existing, updates, observed_at="2026-05-16T00:00:00+00:00")

        row = merged.loc[merged["tracking_key"] == "ps4::123"].iloc[0]
        self.assertFalse(bool(row["Stage2Passed"]))
        self.assertEqual(row["Stage2Status"], "scored")

    def test_merge_tracked_casts_legacy_numeric_column_when_needed(self):
        existing = pd.DataFrame(
            [
                {
                    "tracking_key": "ps4::123",
                    "item_id": "123",
                    "SearchName": "ps4",
                    "LegacyFlag": np.nan,
                }
            ]
        )
        updates = pd.DataFrame(
            [
                {
                    "item_id": "123",
                    "SearchName": "ps4",
                    "LegacyFlag": False,
                }
            ]
        )

        merged = merge_tracked(existing, updates, observed_at="2026-05-16T00:00:00+00:00")

        row = merged.loc[merged["tracking_key"] == "ps4::123"].iloc[0]
        self.assertFalse(bool(row["LegacyFlag"]))


if __name__ == "__main__":
    unittest.main()
