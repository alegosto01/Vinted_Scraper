from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import settings
from telegram_implementation.accountability_service import add_item, send_item_to_chat, update_tracking_message
from telegram_implementation.notify import (
    build_recommended_keyboard,
    send_item_with_button,
)


async def _send_accountability_item(token: str, rec_chat: int, item: dict) -> None:
    cache_key = add_item(item, status="recommended")
    await send_item_to_chat(
        token,
        rec_chat,
        item,
        build_recommended_keyboard(cache_key),
        cache_key=cache_key,
    )
    await update_tracking_message(token, cache_key)


def send_new_items_to_telegram(new_items: pd.DataFrame, telegram_bot: str, telegram_chat_id: str) -> None:
    if new_items.empty:
        print("No new items to send.")
        return

    use_accountability = settings.telegram.accountability_enabled
    rec_chat = settings.telegram.resolved_recommended_deals_chat_id

    for index, row in new_items.iterrows():
        item = row.to_dict()
        if use_accountability and rec_chat:
            print(f"Sending accountability item: {index + 1}/{len(new_items)}")
            asyncio.run(_send_accountability_item(telegram_bot, rec_chat, item))
        else:
            print(f"Sending message to Telegram with description button: {index + 1}/{len(new_items)}")
            asyncio.run(send_item_with_button(telegram_bot, telegram_chat_id, item, row.get("Images")))
