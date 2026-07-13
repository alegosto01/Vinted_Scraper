import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from telegram import CopyTextButton

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_implementation.item_cache import cache_item, load_cached_item, load_description_payload, save_description_payload
from telegram_implementation.description_service import generate_description_for_item
from telegram_implementation.notify import (
    build_caption,
    build_description_actions_keyboard,
    build_generate_keyboard,
    build_model_metadata_line,
    extract_primary_image_url,
)


class TelegramImplementationTests(unittest.TestCase):
    def test_item_cache_round_trip(self):
        item = {"Title": "Zaino balenciaga", "Link": "https://example.com/item/1", "Price": 998.2}
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            cache_key = cache_item(item, cache_dir=cache_dir)
            loaded = load_cached_item(cache_key, cache_dir=cache_dir)

        self.assertEqual(loaded["Title"], "Zaino balenciaga")
        self.assertEqual(loaded["Link"], "https://example.com/item/1")

    def test_description_payload_round_trip(self):
        item = {"Title": "Zaino balenciaga", "Link": "https://example.com/item/1", "Price": 998.2}
        payload = {"generated_description": "Full draft", "generation_mode": "ollama"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            cache_key = cache_item(item, cache_dir=cache_dir)
            save_description_payload(cache_key, payload, cache_dir=cache_dir)
            loaded_payload = load_description_payload(cache_key, cache_dir=cache_dir)

        self.assertIsNotNone(loaded_payload)
        self.assertEqual(loaded_payload["generated_description"], "Full draft")

    def test_generate_keyboard_contains_callback(self):
        keyboard = build_generate_keyboard("abc123")
        button = keyboard.inline_keyboard[0][0]

        self.assertEqual(button.text, "Show Generated Description")
        self.assertEqual(button.callback_data, "generate_description:abc123")

    def test_description_actions_keyboard_has_regenerate_and_copy(self):
        keyboard = build_description_actions_keyboard("abc123", "Full generated description")
        regenerate_button = keyboard.inline_keyboard[0][0]
        copy_button = keyboard.inline_keyboard[0][1]

        self.assertEqual(regenerate_button.text, "Regenerate")
        self.assertEqual(regenerate_button.callback_data, "regenerate_description:abc123")
        self.assertEqual(copy_button.text, "Copy")
        self.assertIsInstance(copy_button.copy_text, CopyTextButton)
        self.assertEqual(copy_button.copy_text.text, "Full generated description")

    def test_description_actions_keyboard_omits_invalid_copy_text(self):
        long_description = "x" * 257

        empty_keyboard = build_description_actions_keyboard("abc123", "")
        long_keyboard = build_description_actions_keyboard("abc123", long_description)

        self.assertEqual(len(empty_keyboard.inline_keyboard[0]), 1)
        self.assertEqual(empty_keyboard.inline_keyboard[0][0].text, "Regenerate")
        self.assertEqual(len(long_keyboard.inline_keyboard[0]), 1)
        self.assertEqual(long_keyboard.inline_keyboard[0][0].text, "Regenerate")

    def test_extract_primary_image_url_from_stringified_list(self):
        raw_value = "['https://images1.vinted.net/a.webp', 'https://images1.vinted.net/b.webp']"

        resolved = extract_primary_image_url(raw_value)

        self.assertEqual(resolved, "https://images1.vinted.net/a.webp")

    def test_caption_includes_recommendation_reason(self):
        item = {
            "Title": "Prada bag",
            "Price": 50,
            "Link": "https://example.com/item/1",
            "RecommendationReason": "Giant model pass: xgboost_basic_v1",
        }

        caption = build_caption(item)

        self.assertIn("Giant model pass", caption)

    def test_caption_includes_model_score_threshold_line(self):
        item = {
            "Title": "Prada bag",
            "Price": 50,
            "Link": "https://example.com/item/1",
            "TelegramModel": "giant_basic_visual/main_image_scores_live_trained",
            "TelegramModelScore": 0.64321,
            "TelegramAdjustedThreshold": 0.58,
            "GiantBestMargin": 0.06321,
        }

        line = build_model_metadata_line(item)
        caption = build_caption(item)

        self.assertEqual(
            line,
            "Model: giant_basic_visual/main_image_scores_live_trained | score 0.643 | threshold 0.580 | margin +0.063",
        )
        self.assertIn(line, caption)

    def test_caption_includes_seller_reviews_when_rated(self):
        item = {
            "Title": "Prada bag",
            "Price": 50,
            "Link": "https://example.com/item/1",
            "ReviewsCount": 158.0,
            "Stars": 4.4,
        }

        caption = build_caption(item)

        self.assertIn("4.4", caption)
        self.assertIn("158 reviews", caption)

    def test_caption_singular_review_and_unrated_and_missing(self):
        singular = build_caption(
            {"Title": "x", "Link": "l", "ReviewsCount": 1.0, "Stars": 5.0}
        )
        self.assertIn("1 review", singular)
        self.assertNotIn("1 reviews", singular)

        unrated = build_caption(
            {"Title": "x", "Link": "l", "ReviewsCount": 0.0, "Stars": -1.0}
        )
        self.assertIn("No reviews yet", unrated)

        missing = build_caption({"Title": "x", "Link": "l"})
        self.assertNotIn("⭐", missing)

    @mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
    @mock.patch("telegram_implementation.description_service.load_example_corpus")
    @mock.patch("telegram_implementation.description_service.build_description_payload")
    def test_generate_description_for_item_prefers_gemini_when_api_key_exists(
        self,
        mock_build_payload,
        mock_load_corpus,
    ):
        item = {
            "Title": "Zaino balenciaga",
            "Brand": "Balenciaga",
            "SearchName": "bags",
            "Description": "Usato poco.",
        }
        mock_load_corpus.return_value = []
        mock_build_payload.return_value = {"generated_description": "Draft"}

        payload = generate_description_for_item(item)

        self.assertEqual(payload["generated_description"], "Draft")
        self.assertTrue(mock_build_payload.call_args.kwargs["use_gemini_api"])
        self.assertFalse(mock_build_payload.call_args.kwargs["use_ollama"])


if __name__ == "__main__":
    unittest.main()
