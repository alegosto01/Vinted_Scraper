# telegram_notify.py
import asyncio
import html
from typing import Iterable, Mapping, Optional
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError

CAPTION_LIMIT = 1024  # Telegram caption limit for photos

def build_caption(item: Mapping) -> str:
    """
    item keys you can pass (all optional):
      id, title, price, currency, size, brand, condition, color, seller, location,
      url, created_at, shipping, notes
    """
    title = (item.get("Title") or "").strip()
    price = f'{item.get("Price")}€'.strip()
    size = item.get("Size")
    brand = item.get("Brand")
    likes = item.get("Likes")
    cond = item.get("Condition")
    url = item.get("Link") or ""


    #seller info
    seller = item.get("seller")
    location = item.get("location")


    # Escape for HTML parse mode
    def esc(x): return html.escape(str(x)) if x else None

    lines = []
    if title: lines.append(f"<b>{esc(title)}</b>")
    # compact info line
    info_bits = [price or None, size, brand, cond]
    info_bits = [esc(x) for x in info_bits if x]
    if info_bits: lines.append(" • ".join(info_bits))
    # seller/location
    sl = " · ".join([x for x in [esc(seller), esc(location)] if x])
    if sl: lines.append(sl)
    if url: lines.append(esc(url))

    caption = "\n".join(lines).strip()

    # Truncate if needed (keep the link if present)
    if len(caption) > CAPTION_LIMIT:
        if url and url in caption:
            keep = CAPTION_LIMIT - len(url) - 1
            caption = caption[:max(0, keep)].rstrip() + "\n" + esc(url)
        else:
            caption = caption[:CAPTION_LIMIT]
    return caption

async def send_item(
    token: str,
    chat_id: int | str,
    item: Mapping,
    image_url: Optional[str] = None,
):
    """
    Send one Vinted item to a Telegram chat.
    - image_url can be a direct image URL; Telegram will fetch it.
    - If image_url is None or fails, we fall back to a text message.
    """
    bot = Bot(token=token)
    caption = build_caption(item)

    try:
        if image_url:
            await bot.send_photo(chat_id=chat_id, photo=image_url,
                                 caption=caption, parse_mode=ParseMode.HTML)
        else:
            # no image: send as text
            await bot.send_message(chat_id=chat_id, text=caption, parse_mode=ParseMode.HTML)
    except RetryAfter as e:
        # Respect Telegram's backoff if we hit flood limits
        await asyncio.sleep(e.retry_after + 0.5)
        return await send_item(token, chat_id, item, image_url)
    except (TimedOut, NetworkError):
        # transient: small backoff and retry once
        await asyncio.sleep(1.0)
        if image_url:
            await bot.send_photo(chat_id=chat_id, photo=image_url,
                                 caption=caption, parse_mode=ParseMode.HTML)
        else:
            await bot.send_message(chat_id=chat_id, text=caption, parse_mode=ParseMode.HTML)

async def send_items_batch(
    token: str,
    chat_id: int | str,
    items: Iterable[Mapping],
    rate_limit_per_sec: float = 25.0,
):
    """
    Send a batch of items with gentle throttling.
    Each item dict can include 'image_url' key; otherwise pass image_url separately.
    """
    delay = 1.0 / max(1.0, rate_limit_per_sec)
    for it in items:
        await send_item(
            token=token,
            chat_id=chat_id,
            item=it,
            image_url=it.get("image_url"),
        )
        await asyncio.sleep(delay)
