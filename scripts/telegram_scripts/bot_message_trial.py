# in your scraping script
import asyncio
from bot_notify import send_item
from dotenv import load_dotenv
import os
load_dotenv("scripts/telegram_scripts/bot_env.env")  # Load environment variables from .env file

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

new_item = {
    "id": "vinted_987654321",
    "title": "Nike Air Max 97 Silver Bullet",
    "price": 79.99, "currency": "€",
    "size": "EU 42",
    "brand": "Nike",
    "condition": "Very good",
    "seller": "juan_sneaks",
    "location": "Madrid",
    "url": "https://www.vinted.it/items/6847345714-bracciale-sodini-viola?homepage_session_id=3f557381-0c74-41e9-aa82-ef66e86f6e3e",
    "image_url": "https://images1.vinted.net/t/03_00168_UWC9c9zkaUhcZ9hgXL3kUXk4/f800/1754845422.webp?s=4110f4c8013185f643cf0312e952571c1a0c150e",
    "notes": "Looks clean, original box included",
}

asyncio.run(send_item(BOT_TOKEN, CHAT_ID, new_item, new_item["image_url"]))
