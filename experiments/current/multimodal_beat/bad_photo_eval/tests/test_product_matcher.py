from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import match_vinted_products as matcher
import scrape_title_candidates as title_scraper
import build_match_split_report as split_report
import bad_photo_deal_monitor as deal_monitor


class FakeTitleEncoder:
    revision = "fake-title-v1"

    def __init__(self):
        self.calls = 0
        self.roles = []

    def encode(self, titles, role):
        self.calls += len(titles)
        self.roles.extend([role] * len(titles))
        rows = []
        for title in titles:
            if "air max 90" in title:
                rows.append([1, 0, 0, 0])
            elif "samba" in title:
                rows.append([0, 1, 0, 0])
            else:
                digest = sum(map(ord, title))
                rows.append([0, 0, 1 + digest % 7, 1])
        return matcher._unit_rows(np.asarray(rows, dtype=np.float32))


class FakeImageEncoder:
    revision = "fake-image-v1"

    def __init__(self):
        self.calls = 0

    def encode(self, paths):
        self.calls += len(paths)
        rows = []
        for path in paths:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB").resize((2, 2)), dtype=np.float32)
            rows.append(rgb.reshape(-1) + 1)
        return matcher._unit_rows(np.asarray(rows))


class ProductMatcherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.target_image = self.root / "target.png"
        self.same_image = self.root / "same.png"
        self.other_image = self.root / "other.png"
        pixels = np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
            dtype=np.uint8,
        )
        Image.fromarray(pixels).save(self.target_image)
        Image.fromarray(pixels).save(self.same_image)
        Image.new("RGB", (2, 2), "black").save(self.other_image)
        self.csv = self.root / "candidates.csv"
        pd.DataFrame({
            "iid": ["20", "10", "30"],
            "name": [
                "Nike Air Max 90 shoes",
                "Scarpe Nike Air Max 90",
                "Adidas Samba",
            ],
            "photo": [str(self.same_image), str(self.same_image), ""],
            "url": ["https://example/20", "https://example/10", "https://example/30"],
            "search": ["nike", "nike", "adidas"],
        }).to_csv(self.csv, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, out_name="out"):
        return matcher.MatchConfig(
            target_item_id="target",
            target_title="  SCARPE   Nike Air Max 90  ",
            target_image=str(self.target_image),
            candidates=self.csv,
            out_dir=self.root / out_name,
            candidate_id_col="iid",
            title_col="name",
            image_col="photo",
            listing_url_col="url",
            search_name_col="search",
            batch_size=2,
        )

    def test_title_normalization_preserves_numbers_codes_and_punctuation(self):
        value = matcher.normalize_title("  Sony   WH-1000XM5, 256GB! ")
        self.assertEqual(value, "sony wh-1000xm5, 256gb!")
        self.assertEqual(
            matcher.prepare_title_text(value, "query", "intfloat/multilingual-e5-base"),
            "query: sony wh-1000xm5, 256gb!",
        )
        self.assertEqual(
            matcher.prepare_title_text(value, "passage", "intfloat/multilingual-e5-base"),
            "passage: sony wh-1000xm5, 256gb!",
        )

    def test_condition_filter_is_added_to_search_url(self):
        url = title_scraper.catalog_url("Valentino Uomo 100 ml", "6", 2)
        self.assertIn("search_text=Valentino+Uomo+100+ml", url)
        self.assertIn("status_ids[]=6", url)
        self.assertTrue(url.endswith("&page=2"))

    def test_report_rejects_mixed_conditions(self):
        split_report.validate_conditions(
            pd.DataFrame({"Condition": ["Nuovo senza cartellino"]}),
            "Nuovo senza cartellino",
        )
        with self.assertRaisesRegex(ValueError, "do not match target condition"):
            split_report.validate_conditions(
                pd.DataFrame({
                    "Condition": ["Nuovo senza cartellino", "Nuovo con cartellino"]
                }),
                "Nuovo senza cartellino",
            )

    def test_deal_monitor_uses_only_requested_searches(self):
        self.assertEqual(
            deal_monitor.SEARCHES,
            (
                "telefoni", "griffati_uomo_all", "griffati_donna_all", "gucci",
                "prada", "nike", "ps4", "donna_accessori_gioielli",
            ),
        )

    def test_provisional_split_requires_both_signals_and_combined_score(self):
        ranked = pd.DataFrame({
            "title_similarity": [0.90, 0.90, 0.70],
            "image_similarity": [0.80, 0.40, 0.90],
            "combined_score": [0.85, 0.65, 0.80],
        })
        result = deal_monitor.provisional_split(ranked, deal_monitor.SplitConfig())
        self.assertEqual(result["decision"].tolist(), ["kept", "non_kept", "non_kept"])

    def test_price_analysis_requires_three_kept_matches_for_deal_verdict(self):
        weak = deal_monitor.price_analysis(50, pd.Series([100, 110]))
        strong = deal_monitor.price_analysis(50, pd.Series([90, 100, 110]))
        self.assertIn("Weak evidence", weak["verdict"])
        self.assertIn("Possible deal", strong["verdict"])
        self.assertAlmostEqual(strong["median"], 100.0)
        self.assertAlmostEqual(strong["discount_pct"], 50.0)

    def test_identical_and_multilingual_title_variants_score_high(self):
        result = matcher.run_matching(
            self.config(), FakeTitleEncoder(), FakeImageEncoder()
        ).set_index("candidate_item_id")
        self.assertGreater(result.loc["10", "title_similarity"], 0.99)
        self.assertGreater(result.loc["20", "title_similarity"], 0.99)

    def test_identical_image_scores_high_and_missing_stays_nan(self):
        result = matcher.run_matching(
            self.config(), FakeTitleEncoder(), FakeImageEncoder()
        ).set_index("candidate_item_id")
        self.assertGreater(result.loc["10", "image_similarity"], 0.99)
        self.assertTrue(np.isnan(result.loc["30", "image_similarity"]))
        self.assertTrue(np.isnan(result.loc["30", "combined_score"]))
        self.assertEqual(result.loc["30", "image_status"], "missing")

    def test_weight_validation(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            matcher.validate_weights(0.7, 0.4)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            matcher.validate_weights(-0.1, 1.1)

    def test_deterministic_ranking_breaks_ties_by_item_id(self):
        ranks = matcher.deterministic_ranks([0.8, 0.8, np.nan], ["20", "10", "01"])
        self.assertEqual(ranks[:2].tolist(), [2.0, 1.0])
        self.assertTrue(np.isnan(ranks[2]))

    def test_target_item_is_excluded_from_candidates(self):
        candidates = pd.read_csv(self.csv, dtype={"iid": str})
        candidates.loc[0, "iid"] = "target"
        candidates.to_csv(self.csv, index=False)
        result = matcher.run_matching(
            self.config(), FakeTitleEncoder(), FakeImageEncoder()
        )
        self.assertNotIn("target", result["candidate_item_id"].tolist())
        self.assertEqual(len(result), 2)

    def test_cache_reuse(self):
        title_one, image_one = FakeTitleEncoder(), FakeImageEncoder()
        matcher.run_matching(self.config(), title_one, image_one)
        title_two, image_two = FakeTitleEncoder(), FakeImageEncoder()
        matcher.run_matching(self.config(), title_two, image_two)
        self.assertGreater(title_one.calls, 0)
        self.assertGreater(image_one.calls, 0)
        self.assertEqual(title_two.calls, 0)
        self.assertEqual(image_two.calls, 0)

    def test_output_schema_gallery_manifest_and_prefix_roles(self):
        title_encoder = FakeTitleEncoder()
        result = matcher.run_matching(
            self.config(), title_encoder, FakeImageEncoder()
        )
        self.assertEqual(result.columns.tolist(), matcher.OUTPUT_COLUMNS)
        written = pd.read_csv(self.root / "out" / "ranked_matches.csv")
        self.assertEqual(written.columns.tolist(), matcher.OUTPUT_COLUMNS)
        gallery = (self.root / "out" / "review_gallery.html").read_text()
        self.assertIn("same_product", gallery)
        self.assertIn("Download labels CSV", gallery)
        self.assertTrue((self.root / "out" / "manifest.json").exists())
        self.assertEqual(title_encoder.roles.count("query"), 1)
        self.assertEqual(title_encoder.roles.count("passage"), 3)


if __name__ == "__main__":
    unittest.main()
