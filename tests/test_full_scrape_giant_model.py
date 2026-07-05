import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.full_scrape_giant_model import model_zoo as mz
from experiments.current.full_scrape_giant_model import metrics as fm
from experiments.current.full_scrape_giant_model import image_filter as imf
from experiments.old.deal_finder.modeling import TARGET_COL, score_with_model


def _synthetic(n=240, seed=0):
    """Generate synthetic DataFrame with learnable signal and all required columns."""
    rng = np.random.RandomState(seed)

    # Target variable and label column
    y = rng.randint(0, 2, n)

    # Core numeric columns with some correlation to y
    price = rng.gamma(2, 20, n) + y * 10
    likes = rng.poisson(3, n) + y * 2
    reviews_count = rng.poisson(5, n)
    stars = rng.uniform(0, 5, n)
    visible_picture_count = rng.randint(0, 6, n)
    hidden_picture_count = rng.randint(0, 3, n)
    picture_count = visible_picture_count + hidden_picture_count
    description_char_len = rng.randint(0, 500, n)
    description_token_count = rng.randint(0, 80, n)
    title_char_len_full = rng.randint(5, 60, n)
    title_token_count_full = rng.randint(1, 12, n)
    upload_age_minutes = rng.randint(1, 5000, n)

    # Seller features
    seller_has_reviews = (reviews_count > 0).astype(int)

    # Missing flags
    stars_missing = np.zeros(n, dtype=int)
    reviews_missing = np.zeros(n, dtype=int)
    interested_missing = np.zeros(n, dtype=int)
    views_missing = np.zeros(n, dtype=int)
    picture_count_missing = np.zeros(n, dtype=int)

    # Engineered numeric features
    price_num = price
    likes_num = likes
    page_num = rng.randint(1, 10, n)

    # Categorical columns
    condition = rng.choice(["new", "very good", "good", "satisfactory"], n)
    brand = rng.choice(["nike", "gucci", "prada", "none"], n)
    location_country = rng.choice(["Italy", "France", "Spain"], n)
    search_name = rng.choice(["nike", "gucci", "prada"], n)

    # Text columns with repeated tokens for TfidfVectorizer min_df=3
    title_text = [
        "great deal rare item " + brand[i]
        if y[i] == 1
        else "common cheap basic item " + brand[i]
        for i in range(n)
    ]

    description_text = [
        "great deal rare unique nice " + brand[i]
        if y[i] == 1
        else "common cheap basic normal " + brand[i]
        for i in range(n)
    ]

    # Construct DataFrame
    df = pd.DataFrame({
        TARGET_COL: y,
        "offline_sold_label": y,
        "Price": price,
        "Likes": likes,
        "ReviewsCount": reviews_count,
        "Stars": stars,
        "VisiblePictureCount": visible_picture_count,
        "HiddenPictureCount": hidden_picture_count,
        "PictureCount": picture_count,
        "description_char_len": description_char_len,
        "description_token_count": description_token_count,
        "title_char_len_full": title_char_len_full,
        "title_token_count_full": title_token_count_full,
        "upload_age_minutes": upload_age_minutes,
        "seller_has_reviews": seller_has_reviews,
        "stars_missing": stars_missing,
        "reviews_missing": reviews_missing,
        "interested_missing": interested_missing,
        "views_missing": views_missing,
        "picture_count_missing": picture_count_missing,
        "price_num": price_num,
        "likes_num": likes_num,
        "page_num": page_num,
        "Condition": condition,
        "Brand": brand,
        "LocationCountry": location_country,
        "SearchName": search_name,
        "TitleText": title_text,
        "DescriptionText": description_text,
    })

    return df


class FullScrapeGiantModelTests(unittest.TestCase):

    def test_zoo_has_expected_names(self):
        """Verify ZOO contains all expected model specs with correct total count."""
        zoo_names = {s.name for s in mz.ZOO}

        expected = {
            "logistic_full_v1",
            "catboost_full_v1",
            "lightgbm_full_v1",
            "stacking_full_v1",
            "rules_price_v1",
        }

        for name in expected:
            self.assertIn(name, zoo_names)

        self.assertEqual(len(mz.ZOO), 14)

    def test_select_features_respects_dense_kinds(self):
        """Verify feature selection respects DENSE/non-DENSE model kinds."""
        df = _synthetic()

        # DENSE kind (HistGB) must drop the sparse TF-IDF text block.
        dense_spec = mz.zoo_by_name()["hist_gradient_full_v1"]
        numeric, categorical, text = mz.select_full_features(df, dense_spec)
        self.assertEqual(text, [])
        self.assertIn("SearchName", categorical)

        # Non-DENSE kind (logistic) keeps text and uses SearchName as categorical.
        logistic_spec = mz.zoo_by_name()["logistic_full_v1"]
        numeric, categorical, text = mz.select_full_features(df, logistic_spec)
        self.assertGreater(len(text), 0)
        self.assertIn("SearchName", categorical)

    def test_every_model_fits_and_scores(self):
        """Integration test: every model in ZOO fits and scores without error."""
        df = _synthetic()

        # 70/30 split by slicing
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        for spec in mz.ZOO:
            with self.subTest(name=spec.name):
                numeric, categorical, text = mz.select_full_features(train_df, spec)
                model = mz.make_model(spec, numeric, categorical, text)
                model.fit(train_df, train_df[TARGET_COL].astype(int))
                scores = np.asarray(score_with_model(model, test_df), dtype=float)
                self.assertEqual(len(scores), len(test_df))
                self.assertTrue(np.all(np.isfinite(scores)))

    def test_metrics_summarize_keys(self):
        """Verify metrics.summarize returns all expected keys."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_score = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
        threshold = 0.5

        result = fm.summarize(y_true, y_score, threshold)

        expected_keys = {
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "coverage",
            "selected_count",
            "lift",
            "precision_at_10",
            "precision_at_25",
            "precision_at_50",
        }

        for key in expected_keys:
            self.assertIn(key, result)

        # Coverage should be between 0 and 1
        self.assertGreaterEqual(result["coverage"], 0.0)
        self.assertLessEqual(result["coverage"], 1.0)

    def test_offline_image_filter_accounting(self):
        """Verify image filter correctly counts rows and accounting."""
        df = pd.DataFrame({
            "VisiblePictureCount": [2, 0, np.nan, 5],
            "PrimaryImageUrl": ["http://x", "", "", ""],
            "SearchName": ["nike", "nike", "nike", "nike"],
        })

        kept, acc = imf.offline_image_filter(df)

        # Should keep rows 0 and 3 (VisiblePictureCount > 0 or non-empty URL)
        self.assertEqual(len(kept), 2)

        # accounting is a DataFrame with a __TOTAL__ row
        self.assertIn("__TOTAL__", acc["search_name"].values)
        total_row = acc[acc["search_name"] == "__TOTAL__"].iloc[0]
        self.assertEqual(int(total_row["total_rows"]), 4)
        self.assertEqual(int(total_row["kept_rows"]), 2)
        self.assertEqual(int(total_row["skipped_no_picture"]), 1)
        self.assertEqual(int(total_row["skipped_picture_count_missing"]), 1)


if __name__ == "__main__":
    unittest.main()
