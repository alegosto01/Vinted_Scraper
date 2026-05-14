import logging
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

for module_name in list(sys.modules):
    if module_name == "utils_lib" or module_name.startswith("utils_lib."):
        sys.modules.pop(module_name, None)

scraper_module = types.ModuleType("Scraper")


class FakeBaseScraper:
    def __init__(self):
        self.logger = logging.getLogger("test-simple-scraper")


scraper_module.Scraper = FakeBaseScraper
sys.modules.setdefault("Scraper", scraper_module)
sys.modules.setdefault("requests_html", types.SimpleNamespace(HTML=lambda *args, **kwargs: None))
config_pkg = types.ModuleType("config")
config_pkg.__path__ = []
sys.modules.setdefault("config", config_pkg)
project_config_module = types.ModuleType("config.project_config")
project_config_module.settings = types.SimpleNamespace(
    paths=types.SimpleNamespace(
        simple_scrape_dir=Path("/tmp/simple-scrape-test"),
        data_dir=Path("/tmp/simple-scrape-test"),
    ),
    telegram=types.SimpleNamespace(bot_token="token", chat_id="chat"),
)
sys.modules.setdefault("config.project_config", project_config_module)
logging_config_module = types.ModuleType("config.logging_config")


@contextmanager
def _eventual_sales_log_context():
    yield


logging_config_module.eventual_sales_log_context = _eventual_sales_log_context
sys.modules.setdefault("config.logging_config", logging_config_module)

import simple_scraper as simple_scraper_module
from simple_scraper import Simple_scraper


class SimpleScraperTrimTests(unittest.TestCase):
    def setUp(self):
        self.settings_patch = patch.object(
            simple_scraper_module,
            "settings",
            types.SimpleNamespace(paths=types.SimpleNamespace(simple_scrape_dir=Path("/tmp/simple-scrape-test"))),
        )
        self.settings_patch.start()

    def tearDown(self):
        self.settings_patch.stop()

    def test_remove_and_manage_old_items_uses_search_date(self):
        scraper = Simple_scraper()
        old_df = pd.DataFrame(
            [
                {"Dataid": 1, "Link": "a", "SearchDate": "20/03/2026 10:00:00"},
                {"Dataid": 2, "Link": "b", "SearchDate": "19/03/2026 10:00:00"},
                {"Dataid": 3, "Link": "c", "SearchDate": "01/01/2020 10:00:00"},
                {"Dataid": 4, "Link": "d", "SearchDate": "21/03/2026 10:00:00"},
            ]
        )
        new_df = pd.DataFrame([
            {"Dataid": 10, "Link": "n1"},
            {"Dataid": 11, "Link": "n2"},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            unsold_path = str(Path(tmpdir) / "unsold_df.csv")
            with patch("simple_scraper.MAX_OLD_ITEMS_PER_SEARCH", 4):
                trimmed_old_df, unsold_df = scraper.remove_and_manage_old_items_in_search(old_df, new_df, unsold_path)

        self.assertEqual(set(trimmed_old_df["Dataid"].tolist()), {1, 4})
        self.assertEqual(set(unsold_df["Dataid"].tolist()), {2, 3})

    def test_attach_local_catalog_images_sets_local_columns(self):
        scraper = Simple_scraper()
        row = {
            "Dataid": "123",
            "Images": "https://example.com/a.webp",
        }

        with patch("simple_scraper.download_listing_images_to_cache", return_value=["/tmp/cache/123/image_01.webp"]) as download_mock:
            enriched = scraper._attach_local_catalog_images(row, "ps4")

        download_mock.assert_called_once()
        self.assertEqual(json.loads(enriched["LocalImagePaths"]), ["/tmp/cache/123/image_01.webp"])
        self.assertEqual(enriched["LocalPrimaryImagePath"], "/tmp/cache/123/image_01.webp")

    def test_catalog_image_cache_root_is_flat_under_image_cache(self):
        scraper = Simple_scraper()
        self.assertEqual(
            scraper._catalog_image_cache_root("ps4"),
            Path("/tmp/simple-scrape-test") / "ps4" / "image_cache",
        )

    def test_compare_and_save_queues_missing_items_for_background_check(self):
        scraper = Simple_scraper()
        old_df = pd.DataFrame(
            [
                {
                    "Dataid": 1,
                    "Link": "https://www.vinted.it/items/1-test",
                    "Title": "Old Item",
                    "MarketStatus": "On Sale",
                    "SearchDate": "20/03/2026 10:00:00",
                }
            ]
        )
        new_df = pd.DataFrame(
            [
                {
                    "Dataid": 2,
                    "Link": "https://www.vinted.it/items/2-test",
                    "Title": "New Item",
                    "MarketStatus": "On Sale",
                    "SearchDate": "20/03/2026 10:05:00",
                }
            ]
        )

        queued_rows = []
        fake_daily_module = types.SimpleNamespace(
            enqueue_priority_background_checks=lambda output_folder, rows_df, source="compare_and_save": queued_rows.append(
                (Path(output_folder), rows_df.copy(), source)
            ) or len(rows_df)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_folder = Path(tmpdir)
            unsold_path = output_folder / "unsold_df.csv"
            sold_path = output_folder / "sold_df.csv"
            non_really_path = output_folder / "non_really_sold_items_ids.csv"
            with patch.dict(sys.modules, {"daily_eventual_sales": fake_daily_module}):
                scraper.compare_and_save_df_serial(
                    new_df.copy(),
                    old_df.copy(),
                    unsold_df_path=str(unsold_path),
                    sold_df_path=str(sold_path),
                    non_really_sold_items_ids_df_path=str(non_really_path),
                    output_folder=str(output_folder),
                )

            self.assertEqual(len(queued_rows), 1)
            queued_folder, queued_df, queued_source = queued_rows[0]
            self.assertEqual(queued_folder, output_folder)
            self.assertEqual(queued_source, "compare_and_save")
            self.assertEqual(queued_df["Dataid"].tolist(), [1])

            saved_old_df = pd.read_csv(output_folder / "old_df.csv")
            self.assertEqual(set(saved_old_df["Dataid"].tolist()), {1, 2})

    def test_compare_and_save_dedupes_existing_sold_df_by_dataid(self):
        scraper = Simple_scraper()
        old_df = pd.DataFrame(
            [
                {
                    "Dataid": 10,
                    "Link": "https://www.vinted.it/items/10-test",
                    "Title": "Old Sold Item",
                    "MarketStatus": "On Sale",
                    "SearchDate": "20/03/2026 10:00:00",
                }
            ]
        )
        new_df = pd.DataFrame(columns=old_df.columns)
        existing_sold_df = pd.DataFrame(
            [
                {
                    "Dataid": 10,
                    "Link": "https://www.vinted.it/items/10-test",
                    "Title": "Old Sold Item",
                    "MarketStatus": "Sold",
                    "SearchDate": "20/03/2026 10:00:00",
                }
            ]
        )
        newly_sold_rows = [
            {
                "Dataid": 10,
                "Link": "https://www.vinted.it/items/10-test",
                "Title": "Old Sold Item Updated",
                "MarketStatus": "Sold",
                "SearchDate": "21/03/2026 10:00:00",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_folder = Path(tmpdir)
            unsold_path = output_folder / "unsold_df.csv"
            sold_path = output_folder / "sold_df.csv"
            non_really_path = output_folder / "non_really_sold_items_ids.csv"
            existing_sold_df.to_csv(sold_path, index=False)

            with patch.object(
                scraper,
                "remove_not_actually_sold_items",
                return_value=(new_df.copy(), old_df.copy(), newly_sold_rows),
            ):
                with patch.object(scraper, "remove_and_manage_old_items_in_search", return_value=(old_df.iloc[0:0].copy(), pd.DataFrame())):
                    scraper.compare_and_save_df_serial(
                        new_df.copy(),
                        old_df.copy(),
                        unsold_df_path=str(unsold_path),
                        sold_df_path=str(sold_path),
                        non_really_sold_items_ids_df_path=str(non_really_path),
                        output_folder=str(output_folder),
                    )

            saved_sold_df = pd.read_csv(sold_path)
            self.assertEqual(saved_sold_df["Dataid"].astype(int).tolist(), [10])
            self.assertEqual(saved_sold_df.loc[0, "Title"], "Old Sold Item Updated")


if __name__ == "__main__":
    unittest.main()
