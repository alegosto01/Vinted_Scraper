from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_implementation import accountability_service as svc


class TestAccountabilityService(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.original_path = svc._TRACKER_PATH
        svc._TRACKER_PATH = Path(self.tmpdir.name) / "tracker.csv"

    def tearDown(self) -> None:
        svc._TRACKER_PATH = self.original_path
        self.tmpdir.cleanup()

    def _sample_item(self) -> dict:
        return {
            "Dataid": "12345",
            "Title": "Nike Air Max",
            "Brand": "Nike",
            "Size": "42",
            "Condition": "Very good",
            "Price": 74.2,
            "Link": "https://www.vinted.it/items/12345",
            "Images": "https://images.vinted.net/12345.jpg",
            "SellerName": "seller_01",
        }

    def test_add_and_get_item(self) -> None:
        item = self._sample_item()
        cache_key = svc.add_item(item, status="recommended")
        self.assertIsInstance(cache_key, str)
        self.assertTrue(cache_key)

        row = svc.get_item(cache_key)
        self.assertIsNotNone(row)
        self.assertEqual(row["dataid"], "12345")
        self.assertEqual(row["title"], "Nike Air Max")
        self.assertEqual(row["status"], "recommended")

    def test_add_duplicate_is_idempotent(self) -> None:
        item = self._sample_item()
        key1 = svc.add_item(item, status="recommended")
        key2 = svc.add_item(item, status="recommended")
        self.assertEqual(key1, key2)

        df = svc._ensure_tracker()
        self.assertEqual(len(df), 1)

    def test_update_item(self) -> None:
        item = self._sample_item()
        cache_key = svc.add_item(item)
        svc.update_item(cache_key, price_paid=50.0, bought_at="2026-05-16T10:00:00")

        row = svc.get_item(cache_key)
        self.assertEqual(row["price_paid"], "50.0")
        self.assertEqual(row["bought_at"], "2026-05-16T10:00:00")

    def test_transition(self) -> None:
        item = self._sample_item()
        cache_key = svc.add_item(item)
        ok = svc.transition(cache_key, "bought", price_paid=60.0)
        self.assertTrue(ok)

        row = svc.get_item(cache_key)
        self.assertEqual(row["status"], "bought")
        self.assertEqual(row["price_paid"], "60.0")

    def test_transition_unknown_item(self) -> None:
        ok = svc.transition("nonexistent", "bought")
        self.assertFalse(ok)

    def test_delete_item(self) -> None:
        item = self._sample_item()
        cache_key = svc.add_item(item)
        self.assertTrue(svc.delete_item(cache_key))
        self.assertIsNone(svc.get_item(cache_key))
        self.assertFalse(svc.delete_item(cache_key))

    def test_message_ids_round_trip(self) -> None:
        item = self._sample_item()
        cache_key = svc.add_item(item)
        svc._save_message_id(cache_key, 123456, 42)
        svc._save_message_id(cache_key, 789012, 99)

        mids = svc._load_message_ids(cache_key)
        self.assertEqual(mids["123456"], 42)
        self.assertEqual(mids["789012"], 99)

    def test_calculate_profit(self) -> None:
        self.assertAlmostEqual(svc.calculate_profit(50.0, 100.0), 50.0)
        self.assertAlmostEqual(svc.calculate_profit(100.0, 50.0), -50.0)
        self.assertAlmostEqual(svc.calculate_profit(0, 0), 0.0)

    def test_tracker_columns_complete(self) -> None:
        item = self._sample_item()
        cache_key = svc.add_item(item)
        row = svc.get_item(cache_key)
        for col in svc._TRACKER_COLUMNS:
            self.assertIn(col, row)


if __name__ == "__main__":
    unittest.main()
