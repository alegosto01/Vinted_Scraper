import logging
import os
import sys
import tempfile
import types
import unittest
import importlib
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


scraper_module = types.ModuleType("Scraper")


class FakeBaseScraper:
    def __init__(self):
        self.logger = logging.getLogger("test-full-scraper")

    def _iter_json_dicts(self, value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from self._iter_json_dicts(child)
        elif isinstance(value, list):
            for child in value:
                yield from self._iter_json_dicts(child)


scraper_module.Scraper = FakeBaseScraper
sys.modules["Scraper"] = scraper_module

sys.modules.pop("full_scraper", None)
Full_Scraper = importlib.import_module("full_scraper").Full_Scraper


class FakeElement:
    def __init__(self, *, attrs=None, text="", children=None):
        self.attrs = attrs or {}
        self.text = text
        self._children = children or []

    def find(self, selector, first=False):
        if selector == "*":
            return self._children[0] if first and self._children else list(self._children)
        return None if first else []


class FakeHtml:
    def __init__(self, mapping, html="", xpath_mapping=None):
        self.mapping = mapping
        self.html = html
        self.xpath_mapping = xpath_mapping or {}

    def find(self, selector, first=False):
        values = self.mapping.get(selector, [])
        if first:
            return values[0] if values else None
        return values

    def xpath(self, selector, first=False):
        values = self.xpath_mapping.get(selector, [])
        if first:
            return values[0] if values else None
        return values


class FullScrapeEnrichmentTests(unittest.TestCase):
    def test_picture_count_sums_visible_images_and_hidden_overlay(self):
        scraper = Full_Scraper()
        root = FakeElement(children=[FakeElement(text="+3")])
        html = FakeHtml(
            {
                'div[class*="item-photos"] img': [
                    FakeElement(attrs={"src": "https://images.example/a.webp"}),
                    FakeElement(attrs={"src": "https://images.example/b.webp"}),
                ],
                'div[class*="item-photos"]': [root],
            }
        )

        meta = scraper.extract_item_page_image_metadata(html)

        self.assertEqual(meta["VisiblePictureCount"], 2)
        self.assertEqual(meta["HiddenPictureCount"], 3)
        self.assertEqual(meta["PictureCount"], 5)
        self.assertEqual(meta["PrimaryImageUrl"], "https://images.example/a.webp")

    def test_picture_count_detects_figure_button_hidden_overlay(self):
        scraper = Full_Scraper()
        html = FakeHtml(
            {
                "section figure button div": [FakeElement(text="+4")],
                "section figure": [FakeElement(), FakeElement(), FakeElement(), FakeElement(), FakeElement()],
            }
        )

        meta = scraper.extract_item_page_image_metadata(html, fallback_primary_url="https://images.example/main.webp")

        self.assertEqual(meta["VisiblePictureCount"], 5)
        self.assertEqual(meta["HiddenPictureCount"], 4)
        self.assertEqual(meta["PictureCount"], 9)

    def test_picture_count_detects_hidden_overlay_from_xpath_nodes(self):
        scraper = Full_Scraper()
        html = FakeHtml(
            {},
            xpath_mapping={
                "//main//section//section//div//figure//button//div//div": [FakeElement(text="+7")]
            },
        )

        meta = scraper.extract_item_page_image_metadata(html, fallback_primary_url="https://images.example/main.webp")

        self.assertEqual(meta["VisiblePictureCount"], 1)
        self.assertEqual(meta["HiddenPictureCount"], 7)
        self.assertEqual(meta["PictureCount"], 8)

    def test_picture_count_detects_hidden_overlay_from_raw_html(self):
        scraper = Full_Scraper()
        html = FakeHtml(
            {},
            html="<main><section><section><div><figure><button><div><div>+6</div></div></button></figure></div></section></section></main>",
        )

        meta = scraper.extract_item_page_image_metadata(html, fallback_primary_url="https://images.example/main.webp")

        self.assertEqual(meta["VisiblePictureCount"], 1)
        self.assertEqual(meta["HiddenPictureCount"], 6)
        self.assertEqual(meta["PictureCount"], 7)

    def test_image_extraction_uses_meta_and_json_ld_without_selenium(self):
        scraper = Full_Scraper()
        html = FakeHtml(
            {
                'meta[property="og:image"]': [FakeElement(attrs={"content": "https://images.example/og.webp"})],
                'script[type="application/ld+json"]': [
                    FakeElement(
                        text='{"@type": "Product", "image": ["https://images.example/json1.jpg", "https://images.example/json2.jpg"]}'
                    )
                ],
            }
        )

        meta = scraper.extract_item_page_image_metadata(html)

        self.assertIn("https://images.example/og.webp", meta["FullImageUrls"])
        self.assertIn("https://images.example/json1.jpg", meta["FullImageUrls"])
        self.assertEqual(meta["PictureCount"], 3)

    def test_select_score_extremes_marks_low_and_high_reasons(self):
        scraper = Full_Scraper()
        scored = pd.DataFrame(
            [
                {"Dataid": "1", "DealFinderScore": 0.01},
                {"Dataid": "2", "DealFinderScore": 0.50},
                {"Dataid": "3", "DealFinderScore": 0.99},
            ]
        )

        selected = scraper.select_score_extremes(scored, low_threshold=0.05, high_threshold=0.95)

        self.assertEqual(selected["Dataid"].tolist(), ["1", "3"])
        self.assertEqual(selected["FullScrapeReason"].tolist(), ["score_low", "score_high"])

    def test_full_scrape_items_are_deduped_by_dataid(self):
        scraper = Full_Scraper()
        rows = pd.DataFrame(
            [
                {"Dataid": "1", "Link": "https://www.vinted.it/items/1-test", "Title": "First"},
                {"Dataid": "1", "Link": "https://www.vinted.it/items/1-test", "Title": "Second"},
            ]
        )

        def fake_scrape_single_product(**kwargs):
            return (
                {
                    "Description": "desc",
                    "Condition": "new",
                    "Upload_date": "today",
                    "Interested_count": 1,
                    "View_count": 2,
                    "SellerName": "seller",
                    "PrimaryImageUrl": "https://images.example/a.webp",
                    "FullImageUrls": ["https://images.example/a.webp"],
                    "VisiblePictureCount": 1,
                    "HiddenPictureCount": 0,
                    "PictureCount": 1,
                    "Images": ["https://images.example/a.webp"],
                },
                {"SellerName": "seller", "SellerId": "s1", "Location": "IT", "ReviewsCount": 3, "Stars": 5},
            )

        with tempfile.TemporaryDirectory() as tmp:
            scraper._simple_scrape_root = lambda: Path(tmp)
            with patch.object(scraper, "scrape_single_product", side_effect=fake_scrape_single_product):
                result = scraper.collect_and_store_full_items(
                    rows,
                    search_name="ps4",
                    reason="sold_backfill",
                    max_workers=1,
                    image_mode="html",
                )

            self.assertEqual(result["succeeded"], 2)
            saved = pd.read_csv(Path(tmp) / "ps4" / "full_scrape" / "items_enriched.csv")
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.iloc[0]["Title"], "Second")

    def test_confirmed_sold_hook_full_scrapes_without_breaking_sold_df(self):
        import daily_eventual_sales

        calls = []

        class FakeFullScraper:
            def collect_and_store_full_items(self, rows_df, **kwargs):
                calls.append((rows_df.copy(), kwargs))
                return {"succeeded": len(rows_df)}

        with tempfile.TemporaryDirectory() as tmp:
            output_folder = Path(tmp) / "ps4"
            output_folder.mkdir()
            pd.DataFrame(
                [
                    {
                        "Dataid": 1,
                        "Link": "https://www.vinted.it/items/1-test",
                        "Title": "Old",
                        "MarketStatus": "On Sale",
                    }
                ]
            ).to_csv(output_folder / "old_df.csv", index=False)

            checked = pd.DataFrame(
                [
                    {
                        "Dataid": 1,
                        "Link": "https://www.vinted.it/items/1-test",
                        "Title": "Old",
                        "LastCheckStatus": "Sold",
                    }
                ]
            )

            fake_scraping_options = types.ModuleType("scraping_options")

            def dedupe_market_rows(df, keep="last"):
                if df.empty:
                    return df.copy()
                subset = ["Dataid"] if "Dataid" in df.columns else ["Link"] if "Link" in df.columns else None
                return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True) if subset else df.copy()

            def ensure_search_tracking_files(folder):
                Path(folder).mkdir(parents=True, exist_ok=True)

            def write_csv_atomic(df, path):
                tmp = f"{path}.tmp"
                df.to_csv(tmp, index=False)
                os.replace(tmp, path)

            fake_scraping_options.dedupe_market_rows = dedupe_market_rows
            fake_scraping_options.ensure_search_tracking_files = ensure_search_tracking_files
            fake_scraping_options.write_csv_atomic = write_csv_atomic

            with patch.dict(sys.modules, {"scraping_options": fake_scraping_options}):
                with patch("full_scraper.Full_Scraper", return_value=FakeFullScraper()):
                    daily_eventual_sales._sync_priority_result_to_tracking_files(
                        output_folder,
                        checked,
                    )

            sold = pd.read_csv(output_folder / "sold_df.csv")
            self.assertEqual(sold["MarketStatus"].tolist(), ["Sold"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]["reason"], "sold_confirmed_live")
            self.assertNotIn("no_residential", calls[0][1])


if __name__ == "__main__":
    unittest.main()
