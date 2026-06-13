from __future__ import annotations

import asyncio
import ast
import html
import sys
from pathlib import Path
from typing import Mapping, Optional

from telegram import Bot, CopyTextButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from telegram_implementation.event_log import log_event
from telegram_implementation.item_cache import cache_item

CAPTION_LIMIT = 1024


def extract_primary_image_url(image_value: object) -> str | None:
    if image_value in (None, ""):
        return None
    if isinstance(image_value, str):
        stripped = image_value.strip()
        if stripped.startswith("http://") or stripped.startswith("https://"):
            return stripped
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return None
            return extract_primary_image_url(parsed)
        return None
    if isinstance(image_value, (list, tuple)):
        for candidate in image_value:
            resolved = extract_primary_image_url(candidate)
            if resolved:
                return resolved
        return None
    return None


def build_caption(item: Mapping[str, object]) -> str:
    title = (str(item.get("Title") or "")).strip()
    price_value = item.get("Price")
    price = f"{price_value}€".strip() if price_value not in (None, "") else ""
    size = item.get("Size") or None
    brand = item.get("Brand") or None
    cond = item.get("Condition") or None
    url = (str(item.get("Link") or "")).strip()
    reason = (str(item.get("RecommendationReason") or "")).strip()

    def esc(value: object) -> str | None:
        if value in (None, ""):
            return None
        return html.escape(str(value))

    lines: list[str] = []
    if title:
        lines.append(f"<b>{esc(title)}</b>")
    info_bits = [esc(bit) for bit in (price or None, size, brand, cond) if bit]
    if info_bits:
        lines.append(" • ".join(info_bits))
    if reason:
        lines.append(esc(reason) or "")
    if url:
        lines.append(esc(url) or "")

    caption = "\n".join(line for line in lines if line).strip()
    if len(caption) > CAPTION_LIMIT:
        if url and url in caption:
            keep = CAPTION_LIMIT - len(url) - 1
            caption = caption[: max(0, keep)].rstrip() + "\n" + (esc(url) or "")
        else:
            caption = caption[:CAPTION_LIMIT]
    return caption


def build_generate_keyboard(cache_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Show Generated Description", callback_data=f"generate_description:{cache_key}")]]
    )


def build_description_actions_keyboard(cache_key: str, description: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Regenerate", callback_data=f"regenerate_description:{cache_key}"),
        InlineKeyboardButton("Copy", copy_text=CopyTextButton(description)),
    ]])


# ── Accountability keyboards ─────────────────────────────────────────────────

def build_recommended_keyboard(cache_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🛒 Bought", callback_data=f"accountability:buy:{cache_key}"),
            InlineKeyboardButton("🗑 Delete", callback_data=f"accountability:delete:{cache_key}"),
        ]
    ])


def build_bought_keyboard(cache_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🏷 Sell Item", callback_data=f"accountability:sell:{cache_key}")]
    ])


def build_drafting_keyboard(cache_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📝 Generate Description", callback_data=f"accountability:desc:{cache_key}"),
        ],
        [
            InlineKeyboardButton("🖼 Improve Images", callback_data=f"accountability:improve:{cache_key}"),
        ],
        [
            InlineKeyboardButton("👕 In the Wardrobe", callback_data=f"accountability:wardrobe:{cache_key}"),
        ],
    ])


def build_selling_keyboard(cache_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💰 Sold", callback_data=f"accountability:sold:{cache_key}")]
    ])


def build_yes_no_keyboard(cache_key: str, prefix: str) -> InlineKeyboardMarkup:
    """Generic Yes/No keyboard for confirmation steps."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Yes", callback_data=f"accountability:{prefix}:yes:{cache_key}"),
            InlineKeyboardButton("❌ No", callback_data=f"accountability:{prefix}:no:{cache_key}"),
        ]
    ])


def build_description_message(item: Mapping[str, object], payload: Mapping[str, object]) -> str:
    """Plain-text, copy-paste-ready message: long-press → Copy text gives the full listing."""
    title = str(item.get("Title") or "").strip()
    price = item.get("Price")
    size = str(item.get("Size") or "").strip()
    brand = str(item.get("Brand") or "").strip()
    condition = str(item.get("Condition") or "").strip()
    description = str(payload.get("generated_description") or "").strip()

    meta_parts = []
    if price not in (None, "", "0"):
        meta_parts.append(f"€{price}")
    if size and size not in ("0", "none", "nan"):
        meta_parts.append(f"Size {size}")
    if brand:
        meta_parts.append(brand)
    if condition:
        meta_parts.append(condition)

    lines = []
    if title:
        lines.append(title)
    if meta_parts:
        lines.append(" · ".join(meta_parts))
    if description:
        lines.append("")
        lines.append(description)

    return "\n".join(lines)


async def send_item_with_button(
    token: str,
    chat_id: int | str,
    item: Mapping[str, object],
    image_url: Optional[str] = None,
) -> None:
    bot = Bot(token=token)
    caption = build_caption(item)
    cache_key = cache_item(item)
    reply_markup = build_generate_keyboard(cache_key)
    resolved_image_url = extract_primary_image_url(image_url)
    log_event("item_sent", item=item, cache_key=cache_key)

    async def send_text_fallback() -> None:
        await bot.send_message(
            chat_id=chat_id,
            text=caption,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )

    try:
        if resolved_image_url:
            await bot.send_photo(
                chat_id=chat_id,
                photo=resolved_image_url,
                caption=caption,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup,
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=caption,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup,
            )
    except BadRequest as exc:
        if resolved_image_url and "Invalid file http url specified" in str(exc):
            await send_text_fallback()
        else:
            raise
    except RetryAfter as exc:
        await asyncio.sleep(exc.retry_after + 0.5)
        await send_item_with_button(token, chat_id, item, image_url=image_url)
    except (TimedOut, NetworkError):
        await asyncio.sleep(1.0)
        if resolved_image_url:
            try:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=resolved_image_url,
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
            except BadRequest as exc:
                if "Invalid file http url specified" in str(exc):
                    await send_text_fallback()
                else:
                    raise
            except (TimedOut, NetworkError):
                await send_text_fallback()
        else:
            await send_text_fallback()
