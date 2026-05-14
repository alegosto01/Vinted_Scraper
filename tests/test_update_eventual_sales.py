import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
stub_scraping_options = sys.modules.get("scraping_options")
if stub_scraping_options is not None and not hasattr(stub_scraping_options, "_check_sold_with_own_driver"):
    sys.modules.pop("scraping_options", None)

import scraping_options
from scraping_options import update_eventual_sale_labels_for_csv


class UpdateEventualSalesTests(unittest.TestCase):
    def test_exclude_known_sold_csv_removes_overlap_from_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "deals_ranked.csv"
            known_sold_path = tmp_path / "sold_df.csv"
            out_dir = tmp_path / "eventual_sale_check"

            pd.DataFrame(
                [
                    {"Dataid": 101, "Link": "https://www.vinted.it/items/101-test", "MarketStatus": "On Sale"},
                    {"Dataid": 202, "Link": "https://www.vinted.it/items/202-test", "MarketStatus": "On Sale"},
                ]
            ).to_csv(input_path, index=False)
            pd.DataFrame(
                [
                    {"Dataid": 101, "Link": "https://www.vinted.it/items/101-test", "MarketStatus": "Sold"},
                ]
            ).to_csv(known_sold_path, index=False)

            def fake_update(df, **_kwargs):
                checked = df.copy()
                checked.loc[:, "MarketStatus"] = "Sold"
                checked.loc[:, "LastCheckStatus"] = "Sold"
                return checked, checked.copy()

            with patch.object(scraping_options, "_update_market_status_for_df", side_effect=fake_update):
                result = update_eventual_sale_labels_for_csv(
                    str(input_path),
                    out_dir=str(out_dir),
                    exclude_known_sold_csv=str(known_sold_path),
                )

            sold_df = pd.read_csv(result["sold_path"])
            labeled_df = pd.read_csv(result["labeled_path"])
            active_df = pd.read_csv(result["active_path"])

            self.assertEqual(result["n_excluded_known_sold"], 1)
            self.assertEqual(sold_df["Dataid"].astype(int).tolist(), [202])
            self.assertEqual(labeled_df["Dataid"].astype(int).tolist(), [202])
            self.assertEqual(len(active_df), 0)


if __name__ == "__main__":
    unittest.main()
