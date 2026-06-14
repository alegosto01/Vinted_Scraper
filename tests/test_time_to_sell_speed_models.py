import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.time_to_sell import train_speed_models as speed_models


def _spec(name: str):
    return next(spec for spec in speed_models.approach_specs() if spec.name == name)


class TimeToSellSpeedModelFeatureTests(unittest.TestCase):
    def test_basic5_selects_dataset_title_and_description_signals(self):
        frame = pd.DataFrame(
            {
                "Price": [10, 25, 40],
                "Likes": [1, 3, 7],
                "Title": ["Nike nuovo", "Prada raro", "PS4 bundle"],
                "Brand": ["Nike", "Prada", "Sony"],
                "Size": ["M", "L", "One size"],
                "Description": ["spedisco subito", "autentico con scatola", "bundle giochi"],
                "TitleTextNormalized": ["nike nuovo", "prada raro", "ps4 bundle"],
                "DescriptionText": ["spedisco subito", "autentico con scatola", "bundle giochi"],
                "title_char_len_tts": [10, 11, 12],
                "title_has_new_word_tts": [1, 0, 0],
                "description_char_len": [15, 21, 12],
                "description_token_count": [2, 3, 2],
            }
        )

        numeric, text = speed_models.feature_columns_for_mode(
            frame,
            mode="basic5",
            spec=_spec("logistic_snapshot_v2"),
            embedding_cols=[],
            include_upload_date=False,
        )

        self.assertIn("title_char_len_tts", numeric)
        self.assertIn("description_char_len", numeric)
        self.assertIn("description_token_count", numeric)
        self.assertIn("TitleTextNormalized", text)
        self.assertIn("Description", text)
        self.assertIn("DescriptionText", text)

    def test_numeric_only_models_keep_scalar_text_signals_without_text_columns(self):
        frame = pd.DataFrame(
            {
                "Price": [10, 25, 40, 55],
                "Likes": [1, 3, 7, 8],
                "Title": ["A", "B", "C", "D"],
                "Description": ["short", "medium text", "longer text here", "longest text here"],
                "description_char_len": [5, 11, 16, 17],
                "description_token_count": [1, 2, 3, 3],
                "title_keyword_positive_count_tts": [0, 1, 1, 2],
                "title_has_bundle_word_tts": [0, 0, 1, 1],
            }
        )

        numeric, text = speed_models.feature_columns_for_mode(
            frame,
            mode="full_visual",
            spec=_spec("hist_gradient_basic_numeric_v1"),
            embedding_cols=[],
            include_upload_date=False,
        )

        self.assertEqual(text, [])
        self.assertIn("description_char_len", numeric)
        self.assertIn("description_token_count", numeric)
        self.assertIn("title_keyword_positive_count_tts", numeric)
        self.assertIn("title_has_bundle_word_tts", numeric)

    def test_rules_price_baseline_does_not_claim_text_signal_features(self):
        frame = pd.DataFrame(
            {
                "Price": [10, 25, 40],
                "Likes": [1, 3, 7],
                "Description": ["spedisco subito", "autentico", "bundle giochi"],
                "description_char_len": [15, 9, 12],
                "title_char_len_tts": [10, 11, 12],
            }
        )

        numeric, text = speed_models.feature_columns_for_mode(
            frame,
            mode="basic5",
            spec=_spec("rules_price_v1"),
            embedding_cols=[],
            include_upload_date=False,
        )

        self.assertEqual(text, [])
        self.assertNotIn("description_char_len", numeric)
        self.assertNotIn("title_char_len_tts", numeric)


if __name__ == "__main__":
    unittest.main()
