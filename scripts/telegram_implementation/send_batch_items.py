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

from telegram_implementation.notify import send_item_with_button


def send_new_items_to_telegram(new_items: pd.DataFrame, telegram_bot: str, telegram_chat_id: str) -> None:
    if new_items.empty:
        print("No new items to send.")
        return

    for index, row in new_items.iterrows():
        print(f"Sending message to Telegram with description button: {index + 1}/{len(new_items)}")
        asyncio.run(send_item_with_button(telegram_bot, telegram_chat_id, row.to_dict(), row.get("Images")))
