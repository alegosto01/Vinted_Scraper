import pandas as pd
import asyncio
from telegram_scripts.bot_notify import send_item

def send_new_items_to_telegram(new_items, telegram_bot, telegram_chat_id):
    """
    Sends new items to a Telegram chat.
    
    Args:
        new_items (pd.DataFrame): DataFrame containing new items.
        telegram_bot (str): Telegram bot token.
        telegram_chat_id (str): Telegram chat ID to send messages to.
    """
    if new_items.empty:
        print("No new items to send.")
        return

    for index, row in new_items.iterrows():
        # message = f"New item found:\nTitle: {row['Title']}\nPrice: {row['Price']}\nLink: {row['Link']}"
        # Here you would implement the actual sending logic using the Telegram API
        print(f"Sending message to Telegram: {index + 1}/{len(new_items)}")
        asyncio.run(send_item(telegram_bot, telegram_chat_id, row, row.get("Images")))
