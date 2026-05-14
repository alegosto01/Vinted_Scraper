import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

stub_scraping_options = sys.modules.get("scraping_options")
if stub_scraping_options is not None and not hasattr(stub_scraping_options, "_check_sold_with_own_driver"):
    sys.modules.pop("scraping_options", None)

import scraping_options
from scraping_options import _update_market_status_for_df


class SoldCsvRecheckTests(unittest.TestCase):
    def test_default_mode_skips_rows_already_marked_sold(self):
        df = pd.DataFrame(
            [
                {
                    "Dataid": 1,
                    "Link": "https://www.vinted.it/items/1-example",
                    "MarketStatus": "Sold",
                    "Price": 10.0,
                }
            ]
        )

        with patch.object(scraping_options, "_check_sold_with_own_driver") as check_mock:
            labeled_df, sold_df = _update_market_status_for_df(df, max_workers=1)

        check_mock.assert_not_called()
        self.assertEqual(labeled_df.at[0, "MarketStatus"], "Sold")
        self.assertTrue(sold_df.empty)

    def test_recheck_mode_can_downgrade_previously_sold_row_to_on_sale(self):
        df = pd.DataFrame(
            [
                {
                    "Dataid": 1,
                    "Link": "https://www.vinted.it/items/1-example",
                    "MarketStatus": "Sold",
                    "Price": 10.0,
                }
            ]
        )

        with patch.object(
            scraping_options,
            "_check_sold_with_own_driver",
            return_value=(0, "OnSale", "2 days", 12.0, 10.0, True),
        ):
            labeled_df, sold_df = _update_market_status_for_df(
                df,
                max_workers=1,
                recheck_sold_rows=True,
            )

        self.assertEqual(labeled_df.at[0, "MarketStatus"], "On Sale")
        self.assertEqual(labeled_df.at[0, "LastCheckStatus"], "OnSale")
        self.assertEqual(float(labeled_df.at[0, "ObservedPagePrice"]), 12.0)
        self.assertTrue(sold_df.empty)

    def test_sold_check_can_update_price_when_input_price_uses_string_dtype(self):
        df = pd.DataFrame(
            {
                "Dataid": [1],
                "Link": ["https://www.vinted.it/items/1-example"],
                "MarketStatus": pd.Series(["On Sale"], dtype="string"),
                "Price": pd.Series(["10.0"], dtype="string"),
            }
        )

        with patch.object(
            scraping_options,
            "_check_sold_with_own_driver",
            return_value=(0, "Sold", "2 days", 12.0, 10.0, True),
        ):
            labeled_df, sold_df = _update_market_status_for_df(df, max_workers=1)

        self.assertEqual(labeled_df.at[0, "MarketStatus"], "Sold")
        self.assertEqual(float(labeled_df.at[0, "Price"]), 12.0)
        self.assertEqual(float(labeled_df.at[0, "ObservedPagePrice"]), 12.0)
        self.assertEqual(len(sold_df), 1)


if __name__ == "__main__":
    unittest.main()
