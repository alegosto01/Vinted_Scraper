from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telegram_implementation import accountability_handlers as handlers
from telegram_implementation import accountability_service as svc
from telegram_implementation.notify import (
    build_bought_keyboard,
    build_drafting_keyboard,
    build_recommended_keyboard,
    build_selling_keyboard,
    build_yes_no_keyboard,
)


class FakeMessage:
    def __init__(self):
        self.text = None
        self.replies = []
        self.chat_id = 999

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


class FakeCallbackQuery:
    def __init__(self):
        self.message = FakeMessage()
        self.data = None
        self.answered = False

    async def answer(self, text=None, show_alert=False):
        self.answered = True


class FakeChat:
    def __init__(self):
        self.id = 999


class FakeUpdate:
    def __init__(self):
        self.message = None
        self.callback_query = None
        self.effective_chat = FakeChat()
        self.effective_user = mock.MagicMock()
        self.effective_user.id = 123


class FakeContext:
    def __init__(self):
        self.user_data = {}
        self.bot = mock.AsyncMock()


class TestAccountabilityKeyboards(unittest.TestCase):
    def test_recommended_keyboard_has_buy_and_delete(self):
        kb = build_recommended_keyboard("abc123")
        buttons = kb.inline_keyboard[0]
        self.assertEqual(buttons[0].text, "🛒 Bought")
        self.assertEqual(buttons[0].callback_data, "accountability:buy:abc123")
        self.assertEqual(buttons[1].text, "🗑 Delete")
        self.assertEqual(buttons[1].callback_data, "accountability:delete:abc123")

    def test_bought_keyboard_has_sell(self):
        kb = build_bought_keyboard("abc123")
        self.assertEqual(kb.inline_keyboard[0][0].text, "🏷 Sell Item")
        self.assertEqual(kb.inline_keyboard[0][0].callback_data, "accountability:sell:abc123")

    def test_drafting_keyboard_has_three_rows(self):
        kb = build_drafting_keyboard("abc123")
        self.assertEqual(kb.inline_keyboard[0][0].text, "📝 Generate Description")
        self.assertEqual(kb.inline_keyboard[1][0].text, "🖼 Improve Images")
        self.assertEqual(kb.inline_keyboard[2][0].text, "👕 In the Wardrobe")

    def test_selling_keyboard_has_sold(self):
        kb = build_selling_keyboard("abc123")
        self.assertEqual(kb.inline_keyboard[0][0].text, "💰 Sold")
        self.assertEqual(kb.inline_keyboard[0][0].callback_data, "accountability:sold:abc123")

    def test_yes_no_keyboard(self):
        kb = build_yes_no_keyboard("abc123", "same_info")
        self.assertEqual(kb.inline_keyboard[0][0].text, "✅ Yes")
        self.assertEqual(kb.inline_keyboard[0][0].callback_data, "accountability:same_info:yes:abc123")
        self.assertEqual(kb.inline_keyboard[0][1].text, "❌ No")
        self.assertEqual(kb.inline_keyboard[0][1].callback_data, "accountability:same_info:no:abc123")


class TestAccountabilityStateMachine(unittest.TestCase):
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

    def test_handle_buy_callback_sets_state(self):
        update = FakeUpdate()
        update.callback_query = FakeCallbackQuery()
        update.callback_query.data = "accountability:buy:test_key"
        context = FakeContext()

        asyncio.run(handlers.handle_accountability_callback(update, context))

        self.assertEqual(context.user_data["accountability_state"]["stage"], "awaiting_price_paid")
        self.assertEqual(context.user_data["accountability_state"]["cache_key"], "test_key")
        self.assertIn("How much did you pay", update.callback_query.message.replies[0])

    def test_handle_delete_callback_removes_item(self):
        cache_key = svc.add_item(self._sample_item())
        update = FakeUpdate()
        update.callback_query = FakeCallbackQuery()
        update.callback_query.data = f"accountability:delete:{cache_key}"
        context = FakeContext()

        asyncio.run(handlers.handle_accountability_callback(update, context))

        self.assertIsNone(svc.get_item(cache_key))
        self.assertIn("removed", update.callback_query.message.replies[0])

    def test_text_input_price_paid_transitions(self):
        cache_key = svc.add_item(self._sample_item())
        update = FakeUpdate()
        update.message = FakeMessage()
        update.message.text = "55.50"
        context = FakeContext()
        context.user_data["accountability_state"] = {
            "cache_key": cache_key,
            "stage": "awaiting_price_paid",
            "data": {},
        }

        with mock.patch.object(handlers, "_send_to_stage", new=mock.AsyncMock()):
            consumed = asyncio.run(handlers.handle_accountability_text(update, context))

        self.assertTrue(consumed)
        self.assertNotIn("accountability_state", context.user_data)
        row = svc.get_item(cache_key)
        self.assertEqual(row["status"], "bought")
        self.assertEqual(row["price_paid"], "55.5")

    def test_text_input_invalid_price_rejects(self):
        update = FakeUpdate()
        update.message = FakeMessage()
        update.message.text = "not a price"
        context = FakeContext()
        context.user_data["accountability_state"] = {
            "cache_key": "whatever",
            "stage": "awaiting_price_paid",
            "data": {},
        }

        consumed = asyncio.run(handlers.handle_accountability_text(update, context))
        self.assertTrue(consumed)
        self.assertIn("Please send a valid number", update.message.replies[0])
        # State should NOT be cleared
        self.assertIn("accountability_state", context.user_data)

    def test_text_input_new_title_then_description_then_price(self):
        cache_key = svc.add_item(self._sample_item())
        context = FakeContext()

        # Step 1: new title
        update = FakeUpdate()
        update.message = FakeMessage()
        update.message.text = "My New Title"
        context.user_data["accountability_state"] = {
            "cache_key": cache_key,
            "stage": "awaiting_new_title",
            "data": {"same_info": False},
        }
        consumed = asyncio.run(handlers.handle_accountability_text(update, context))
        self.assertTrue(consumed)
        self.assertEqual(context.user_data["accountability_state"]["stage"], "awaiting_new_description")
        self.assertEqual(context.user_data["accountability_state"]["data"]["new_title"], "My New Title")

        # Step 2: new description
        update = FakeUpdate()
        update.message = FakeMessage()
        update.message.text = "Great condition, barely used."
        consumed = asyncio.run(handlers.handle_accountability_text(update, context))
        self.assertTrue(consumed)
        self.assertEqual(context.user_data["accountability_state"]["stage"], "awaiting_new_price")
        self.assertEqual(context.user_data["accountability_state"]["data"]["new_description"], "Great condition, barely used.")

        # Step 3: price
        update = FakeUpdate()
        update.message = FakeMessage()
        update.message.text = "120"
        with mock.patch.object(handlers, "_send_to_stage", new=mock.AsyncMock()):
            consumed = asyncio.run(handlers.handle_accountability_text(update, context))
        self.assertTrue(consumed)
        row = svc.get_item(cache_key)
        self.assertEqual(row["status"], "selling")
        self.assertEqual(row["new_title"], "My New Title")
        self.assertEqual(row["new_description"], "Great condition, barely used.")
        self.assertEqual(row["listed_price"], "120.0")

    def test_photo_handler_consumes_when_awaiting_photos(self):
        update = FakeUpdate()
        update.message = mock.MagicMock()
        update.message.photo = [mock.MagicMock()]
        update.message.media_group_id = None
        context = FakeContext()
        context.user_data["accountability_state"] = {
            "cache_key": "test_key",
            "stage": "awaiting_photos",
            "data": {},
        }

        with mock.patch.object(handlers, "_process_photos", new=mock.AsyncMock()):
            consumed = asyncio.run(handlers.handle_accountability_photo(update, context))

        self.assertTrue(consumed)

    def test_photo_handler_ignores_when_no_state(self):
        update = FakeUpdate()
        update.message = mock.MagicMock()
        update.message.photo = [mock.MagicMock()]
        context = FakeContext()

        consumed = asyncio.run(handlers.handle_accountability_photo(update, context))
        self.assertFalse(consumed)


if __name__ == "__main__":
    unittest.main()
