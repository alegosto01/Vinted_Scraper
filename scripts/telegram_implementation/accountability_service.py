from __future__ import annotations

import fcntl
import html
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from telegram import Bot, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import settings
from telegram_implementation.event_log import log_event
from telegram_implementation.item_cache import cache_item
from telegram_implementation.notify import build_caption, extract_primary_image_url

LOGGER = logging.getLogger(__name__)

_TRACKER_PATH = settings.paths.data_dir / "telegram_implementation" / "accountability_tracker.csv"
# Cross-process advisory lock. The bot AND the scoring/sender service both mutate
# the tracker CSV; without this they interleave read-modify-write and erase each
# other's rows (e.g. bought items reverting to recommended / vanishing).
_TRACKER_LOCK_PATH = _TRACKER_PATH.with_suffix(".lock")


@contextmanager
def _tracker_lock():
    """Hold an exclusive OS lock across a whole read-modify-write of the tracker.

    fcntl.flock is per-open-fd, so this is NOT reentrant — callers must not nest it.
    """
    _TRACKER_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(_TRACKER_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

_TRACKER_COLUMNS = [
    "cache_key", "dataid", "title", "brand", "size", "condition",
    "original_price", "original_link", "original_images", "original_seller",
    "status", "price_paid", "bought_at",
    "listed_price", "listed_at", "new_title", "new_description", "new_images",
    "sold_price", "sold_at", "profit",
    "message_ids",
]

_STATUS_FLOW = ["recommended", "bought", "drafting", "selling", "resold"]


def _ensure_tracker() -> pd.DataFrame:
    _TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _TRACKER_PATH.exists():
        df = pd.DataFrame(columns=_TRACKER_COLUMNS)
        df.to_csv(_TRACKER_PATH, index=False)
        return df
    # keep_default_na/na_filter=False → empty cells come back as "" strings,
    # never NaN floats (which would break .strip() / .lower() downstream).
    return pd.read_csv(_TRACKER_PATH, dtype=str, keep_default_na=False, na_filter=False)


def _save_tracker(df: pd.DataFrame) -> None:
    # Atomic: write a temp file then os.replace, so a mid-write kill (RuntimeMaxSec/
    # OOM) can never truncate the tracker and drop rows.
    fd, tmp = tempfile.mkstemp(dir=str(_TRACKER_PATH.parent), suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, _TRACKER_PATH)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _normalize_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def add_item(item: Mapping[str, Any], status: str = "recommended") -> str:
    """Add a new item to the tracker. Returns cache_key."""
    cache_key = cache_item(item)
    row = {
        "cache_key": cache_key,
        "dataid": _normalize_value(item.get("Dataid") or item.get("dataid") or item.get("item_id")),
        "title": _normalize_value(item.get("Title")),
        "brand": _normalize_value(item.get("Brand")),
        "size": _normalize_value(item.get("Size")),
        "condition": _normalize_value(item.get("Condition")),
        "original_price": _normalize_value(item.get("Price")),
        "original_link": _normalize_value(item.get("Link")),
        "original_images": _normalize_value(item.get("Images") or item.get("LocalImagePaths")),
        "original_seller": _normalize_value(item.get("SellerName") or item.get("seller_name")),
        "status": status,
        "price_paid": "",
        "bought_at": "",
        "listed_price": "",
        "listed_at": "",
        "new_title": "",
        "new_description": "",
        "new_images": "",
        "sold_price": "",
        "sold_at": "",
        "profit": "",
        "message_ids": "{}",
    }

    with _tracker_lock():
        df = _ensure_tracker()
        if cache_key in df["cache_key"].values:
            LOGGER.info("Item %s already in tracker", cache_key)
            return cache_key
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        _save_tracker(df)

    log_event("accountability_item_added", item=item, cache_key=cache_key,
              details={"status": status})
    return cache_key


def get_item(cache_key: str) -> dict[str, Any] | None:
    """Read a single tracker row as a dict. Returns None if not found."""
    df = _ensure_tracker()
    mask = df["cache_key"] == cache_key
    if not mask.any():
        return None
    return df[mask].iloc[0].to_dict()


def update_item(cache_key: str, **fields: Any) -> bool:
    """Update fields for an existing tracker row. Returns True if found."""
    with _tracker_lock():
        df = _ensure_tracker()
        mask = df["cache_key"] == cache_key
        if not mask.any():
            return False

        for col, val in fields.items():
            if col not in df.columns:
                LOGGER.warning("Unknown tracker column: %s", col)
                continue
            df.loc[mask, col] = _normalize_value(val)

        _save_tracker(df)
    return True


def transition(cache_key: str, new_status: str, **extra_fields: Any) -> bool:
    """Update status and any extra fields. Returns True if found."""
    if new_status not in _STATUS_FLOW:
        LOGGER.warning("Unknown status: %s", new_status)

    fields = {"status": new_status, **extra_fields}
    return update_item(cache_key, **fields)


def delete_item(cache_key: str) -> bool:
    """Remove item from tracker entirely. Returns True if found."""
    with _tracker_lock():
        df = _ensure_tracker()
        mask = df["cache_key"] == cache_key
        if not mask.any():
            return False
        df = df[~mask].reset_index(drop=True)
        _save_tracker(df)
    log_event("accountability_item_deleted", cache_key=cache_key)
    return True


def _load_message_ids(cache_key: str) -> dict[str, list[int]]:
    """Return {chat_id: [msg_id, ...]}. Accepts legacy single-int entries."""
    row = get_item(cache_key)
    if row is None:
        return {}
    raw = row.get("message_ids") or "{}"
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    result: dict[str, list[int]] = {}
    for chat_id, value in parsed.items():
        if isinstance(value, list):
            result[str(chat_id)] = [int(v) for v in value]
        elif isinstance(value, int):
            result[str(chat_id)] = [value]
    return result


def register_item_message(cache_key: str, chat_id: int | str, message_id: int) -> None:
    """Append a message_id to the list for this cache_key + chat_id."""
    LOGGER.info("register_item_message cache_key=%s chat_id=%s msg_id=%s", cache_key, chat_id, message_id)
    mids = _load_message_ids(cache_key)
    key = str(chat_id)
    mids.setdefault(key, []).append(int(message_id))
    ok = update_item(cache_key, message_ids=json.dumps(mids, ensure_ascii=False))
    if not ok:
        LOGGER.error("Failed to save message_id %s for cache_key %s", message_id, cache_key)


def _save_message_id(cache_key: str, chat_id: int | str, message_id: int) -> None:
    register_item_message(cache_key, chat_id, message_id)


async def send_item_to_chat(
    token: str,
    chat_id: int | str,
    item: Mapping[str, Any],
    reply_markup: InlineKeyboardMarkup,
    cache_key: str | None = None,
) -> int | None:
    """Send an item photo+caption+keyboard to a chat. Returns message_id or None."""
    bot = Bot(token=token)
    caption = build_caption(item)
    resolved_image_url = extract_primary_image_url(item.get("Images"))

    message_id: int | None = None

    async def _send_text() -> None:
        nonlocal message_id
        msg = await bot.send_message(
            chat_id=chat_id,
            text=caption,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
        )
        message_id = msg.message_id

    try:
        if resolved_image_url:
            msg = await bot.send_photo(
                chat_id=chat_id,
                photo=resolved_image_url,
                caption=caption,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup,
            )
            message_id = msg.message_id
        else:
            await _send_text()
    except BadRequest as exc:
        if resolved_image_url:
            LOGGER.warning("Photo send failed for %s, falling back to text: %s", resolved_image_url, exc)
            await _send_text()
        else:
            raise
    except RetryAfter as exc:
        await __import__("asyncio").sleep(exc.retry_after + 0.5)
        return await send_item_to_chat(token, chat_id, item, reply_markup, cache_key)
    except (TimedOut, NetworkError):
        await __import__("asyncio").sleep(1.0)
        if resolved_image_url:
            try:
                msg = await bot.send_photo(
                    chat_id=chat_id,
                    photo=resolved_image_url,
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                    reply_markup=reply_markup,
                )
                message_id = msg.message_id
            except BadRequest as exc:
                LOGGER.warning("Photo send failed on retry for %s, falling back to text: %s", resolved_image_url, exc)
                await _send_text()
            except (TimedOut, NetworkError):
                await _send_text()
        else:
            await _send_text()

    if message_id is not None and cache_key is not None:
        _save_message_id(cache_key, chat_id, message_id)

    return message_id


async def delete_item_message(token: str, cache_key: str, chat_id: int | str) -> bool:
    """Delete ALL tracked messages for this cache_key in this chat. Returns True if any were deleted."""
    mids = _load_message_ids(cache_key)
    ids = mids.get(str(chat_id), [])
    LOGGER.info("delete_item_message cache_key=%s chat_id=%s tracked_ids=%s", cache_key, chat_id, ids)
    if not ids:
        return False

    bot = Bot(token=token)
    deleted_any = False
    for message_id in ids:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=message_id)
            deleted_any = True
        except BadRequest as exc:
            LOGGER.warning("Failed to delete msg %s in chat %s: %s", message_id, chat_id, exc)
    mids.pop(str(chat_id), None)
    update_item(cache_key, message_ids=json.dumps(mids, ensure_ascii=False))
    return deleted_any


def calculate_profit(price_paid: float, sold_price: float) -> float:
    """Simple profit calculator."""
    try:
        return float(sold_price) - float(price_paid)
    except (ValueError, TypeError):
        return 0.0


# ── Tracking chat helpers ────────────────────────────────────────────────────

_TRACKER_STATUS_ORDER = ["recommended", "bought", "drafting", "selling", "resold"]
_TRACKER_STATUS_EMOJI = {
    "recommended": "🔍",
    "bought": "🛒",
    "drafting": "📝",
    "selling": "🏷",
    "resold": "✅",
}


def build_tracking_caption(row: dict[str, Any]) -> str:
    """Build a rich status caption for the tracking chat."""
    title = str(row.get("title") or "").strip()
    brand = str(row.get("brand") or "").strip()
    size = str(row.get("size") or "").strip()
    condition = str(row.get("condition") or "").strip()
    original_price = str(row.get("original_price") or "").strip()
    link = str(row.get("original_link") or "").strip()
    status = str(row.get("status") or "recommended").strip()

    lines: list[str] = []
    if title:
        lines.append(f"<b>{html.escape(title)}</b>")
    meta = " • ".join(p for p in [brand, size, condition] if p)
    if meta:
        lines.append(html.escape(meta))
    if original_price:
        lines.append(f"Original: €{html.escape(original_price)}")
    if link:
        lines.append(html.escape(link))
    lines.append("")

    # Pipeline visual
    current_idx = _TRACKER_STATUS_ORDER.index(status) if status in _TRACKER_STATUS_ORDER else 0
    pipeline_parts: list[str] = []
    for i, step in enumerate(_TRACKER_STATUS_ORDER):
        emoji = _TRACKER_STATUS_EMOJI.get(step, "⬜")
        if i < current_idx:
            pipeline_parts.append(f"✅ {step.capitalize()}")
        elif i == current_idx:
            pipeline_parts.append(f"{emoji} <b>{step.capitalize()}</b>")
        else:
            pipeline_parts.append(f"⬜ {step.capitalize()}")
    lines.append(" → ".join(pipeline_parts))
    lines.append("")

    # Step-specific details
    price_paid = str(row.get("price_paid") or "").strip()
    bought_at = str(row.get("bought_at") or "").strip()
    listed_price = str(row.get("listed_price") or "").strip()
    listed_at = str(row.get("listed_at") or "").strip()
    new_title = str(row.get("new_title") or "").strip()
    new_description = str(row.get("new_description") or "").strip()
    sold_price = str(row.get("sold_price") or "").strip()
    profit = str(row.get("profit") or "").strip()
    sold_at = str(row.get("sold_at") or "").strip()

    if price_paid:
        lines.append(f"💶 Price paid: €{html.escape(price_paid)}")
    if bought_at:
        lines.append(f"📅 Bought: {html.escape(bought_at[:19])}")
    if listed_price:
        lines.append(f"🏷 Listed at: €{html.escape(listed_price)}")
    if listed_at:
        lines.append(f"📅 Listed: {html.escape(listed_at[:19])}")
    if new_title:
        lines.append(f"✏️ New title: {html.escape(new_title)}")
    if new_description:
        lines.append(f"📝 Description: {html.escape(new_description[:120])}{'…' if len(new_description) > 120 else ''}")
    if sold_price:
        lines.append(f"💰 Sold: €{html.escape(sold_price)}")
    if profit:
        lines.append(f"📈 Profit: €{html.escape(profit)}")
    if sold_at:
        lines.append(f"📅 Sold: {html.escape(sold_at[:19])}")

    return "\n".join(lines)


async def update_tracking_message(token: str, cache_key: str) -> int | None:
    """Delete old tracking message(s) and send a fresh one reflecting current status."""
    from telegram import InlineKeyboardMarkup

    tracking_chat = settings.telegram.resolved_tracking_chat_id
    if tracking_chat is None:
        return None

    row = get_item(cache_key)
    if row is None:
        return None

    # Delete previous tracking messages for this item
    await delete_item_message(token, cache_key, tracking_chat)

    item = {
        "Title": row.get("title") or "",
        "Price": row.get("original_price") or "",
        "Size": row.get("size") or "",
        "Brand": row.get("brand") or "",
        "Condition": row.get("condition") or "",
        "Link": row.get("original_link") or "",
        "Images": row.get("original_images") or "",
    }
    caption = build_tracking_caption(row)

    # Send text-only (no buttons) to tracking chat
    msg_id = await send_item_to_chat(
        token,
        tracking_chat,
        item,
        InlineKeyboardMarkup([]),
        cache_key=cache_key,
    )

    # Update the caption with the rich tracking text (send_item_to_chat used build_caption)
    # We need to edit the message to use our tracking caption instead.
    if msg_id is not None:
        bot = Bot(token=token)
        try:
            await bot.edit_message_caption(
                chat_id=tracking_chat,
                message_id=msg_id,
                caption=caption,
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not edit tracking message caption: %s", exc)

    return msg_id
