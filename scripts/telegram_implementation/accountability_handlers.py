from __future__ import annotations

import asyncio
import html
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from telegram import Bot, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import settings
from telegram_implementation.accountability_service import (
    add_item,
    calculate_profit,
    delete_item,
    delete_item_message,
    get_item,
    register_item_message,
    send_item_to_chat,
    transition,
    update_item,
    update_tracking_message,
)
from telegram_implementation.description_service import generate_description_for_item
from telegram_implementation.event_log import log_event
from telegram_implementation.item_cache import load_cached_item
from telegram_implementation.notify import (
    build_bought_keyboard,
    build_caption,
    build_drafting_keyboard,
    build_recommended_keyboard,
    build_selling_keyboard,
    build_yes_no_keyboard,
    build_description_message,
    build_description_actions_keyboard,
)


def _first_improved_photo_path(new_images_raw: str | None) -> Path | None:
    """Parse the JSON-encoded list in new_images and return the first existing file."""
    if not new_images_raw:
        return None
    try:
        parsed = json.loads(new_images_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    for entry in parsed:
        p = Path(str(entry))
        if p.exists():
            return p
    return None


async def _send_local_photo_to_chat(
    token: str,
    chat_id: int | str,
    photo_path: Path,
    caption: str,
    cache_key: str,
) -> None:
    """Send a local photo + caption and persist the message_id on the tracker row."""
    bot = Bot(token=token)
    with photo_path.open("rb") as fh:
        msg = await bot.send_photo(
            chat_id=chat_id,
            photo=fh,
            caption=caption,
            parse_mode=ParseMode.HTML,
        )
    row = get_item(cache_key) or {}
    raw = row.get("message_ids") or "{}"
    try:
        mids = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        mids = {}
    mids[str(chat_id)] = msg.message_id
    update_item(cache_key, message_ids=json.dumps(mids, ensure_ascii=False))
import telegram_implementation.photo_improvement_service as photo_svc

LOGGER = logging.getLogger(__name__)

# Prefix for all accountability callbacks
ACCOUNTABILITY_PREFIX = "accountability:"

_media_group_buffer: dict[str, dict] = {}


# ── helpers ──────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _parse_float(text: str) -> float | None:
    try:
        return float(text.replace("€", "").replace(",", ".").strip())
    except (ValueError, TypeError):
        return None


def _item_from_tracker(row: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a minimal item dict from tracker row for caption building."""
    return {
        "Title": row.get("title") or row.get("new_title") or "",
        "Price": row.get("listed_price") or row.get("original_price") or "",
        "Size": row.get("size") or "",
        "Brand": row.get("brand") or "",
        "Condition": row.get("condition") or "",
        "Link": row.get("original_link") or "",
        "Images": row.get("original_images") or "",
    }


async def _send_to_stage(
    cache_key: str,
    stage_chat_id: int | None,
    keyboard_builder,
    token: str,
) -> None:
    """Send item to the next accountability chat."""
    if stage_chat_id is None:
        LOGGER.error("Chat ID not configured for stage")
        return
    row = get_item(cache_key)
    if row is None:
        return
    item = _item_from_tracker(row)
    await send_item_to_chat(token, stage_chat_id, item, keyboard_builder(cache_key), cache_key)


async def _reply_tracked(update: Update, cache_key: str, text: str, **kwargs):
    """Reply via update and register the resulting message_id under cache_key + chat_id.

    Used so every bot-sent message about an item is tracked, allowing
    delete_item_message to wipe ALL traces of the item from a chat on transition.
    """
    msg = None
    if update.callback_query and update.callback_query.message:
        msg = await update.callback_query.message.reply_text(text, **kwargs)
    elif update.message:
        msg = await update.message.reply_text(text, **kwargs)
    if msg is not None:
        register_item_message(cache_key, msg.chat_id, msg.message_id)
    return msg


# ── callback router ──────────────────────────────────────────────────────────

async def handle_accountability_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None or query.data is None or query.message is None:
        return

    data = query.data
    if not data.startswith(ACCOUNTABILITY_PREFIX):
        return

    parts = data.removeprefix(ACCOUNTABILITY_PREFIX).split(":")
    if len(parts) < 2:
        return

    action = parts[0]
    cache_key = parts[-1]

    try:
        await query.answer()
    except BadRequest:
        pass

    if action == "buy":
        await _cb_buy(update, context, cache_key)
    elif action == "delete":
        await _cb_delete(update, context, cache_key)
    elif action == "sell":
        await _cb_sell(update, context, cache_key)
    elif action == "desc":
        await _cb_desc(update, context, cache_key)
    elif action == "improve":
        await _cb_improve(update, context, cache_key)
    elif action == "wardrobe":
        await _cb_wardrobe(update, context, cache_key)
    elif action == "sold":
        await _cb_sold(update, context, cache_key)
    elif action == "same_info":
        answer = parts[1] if len(parts) > 2 else ""
        await _cb_same_info(update, context, cache_key, answer)
    elif action == "price_as_decided":
        answer = parts[1] if len(parts) > 2 else ""
        await _cb_price_as_decided(update, context, cache_key, answer)


# ── individual callbacks ─────────────────────────────────────────────────────

async def _cb_buy(update: Update, context: ContextTypes.DEFAULT_TYPE, cache_key: str) -> None:
    await _reply_tracked(update, cache_key, "💶 How much did you pay for this item? (send the amount)")
    context.user_data["accountability_state"] = {
        "cache_key": cache_key,
        "stage": "awaiting_price_paid",
        "data": {},
    }
    log_event("accountability_buy_pressed", cache_key=cache_key)


async def _cb_delete(update: Update, context: ContextTypes.DEFAULT_TYPE, cache_key: str) -> None:
    token = settings.telegram.bot_token
    chat_id = settings.telegram.resolved_recommended_deals_chat_id
    if chat_id:
        await delete_item_message(token, cache_key, chat_id)
    delete_item(cache_key)
    # Clean up any lingering state for this item
    if context.user_data.get("accountability_state", {}).get("cache_key") == cache_key:
        del context.user_data["accountability_state"]


async def _cb_sell(update: Update, context: ContextTypes.DEFAULT_TYPE, cache_key: str) -> None:
    token = settings.telegram.bot_token
    transition(cache_key, "drafting")
    await _send_to_stage(
        cache_key,
        settings.telegram.resolved_drafting_items_chat_id,
        build_drafting_keyboard,
        token,
    )
    # Optionally delete from bought chat
    bought_chat = settings.telegram.resolved_bought_items_chat_id
    if bought_chat:
        await delete_item_message(token, cache_key, bought_chat)
    await update_tracking_message(token, cache_key)
    log_event("accountability_sell_pressed", cache_key=cache_key)


async def _cb_desc(update: Update, context: ContextTypes.DEFAULT_TYPE, cache_key: str) -> None:
    row = get_item(cache_key)
    if row is None:
        await _reply_tracked(update, cache_key, "Item not found in tracker.")
        return

    try:
        item = load_cached_item(cache_key)
    except FileNotFoundError:
        # Fallback: reconstruct from tracker row
        item = _item_from_tracker(row)

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        payload = generate_description_for_item(item, ollama_language="english")
        # Save new description into tracker
        new_desc = str(payload.get("generated_description") or "").strip()
        if new_desc:
            update_item(cache_key, new_description=new_desc)
        msg = build_description_message(item, payload)
        await _reply_tracked(
            update,
            cache_key,
            msg,
            reply_markup=build_description_actions_keyboard(cache_key, new_desc),
        )
        log_event("accountability_description_generated", cache_key=cache_key,
                  details={"mode": payload.get("generation_mode"), "model": payload.get("model_name")})
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Description generation failed")
        await _reply_tracked(
            update,
            cache_key,
            f"<b>Description generation failed.</b>\n{html.escape(str(exc))}",
            parse_mode=ParseMode.HTML,
        )


async def _cb_improve(update: Update, context: ContextTypes.DEFAULT_TYPE, cache_key: str) -> None:
    await _reply_tracked(update, cache_key, "📸 Send me the photos you want to improve for this item.")
    context.user_data["accountability_state"] = {
        "cache_key": cache_key,
        "stage": "awaiting_photos",
        "data": {},
    }
    log_event("accountability_improve_photos_pressed", cache_key=cache_key)


async def _cb_wardrobe(update: Update, context: ContextTypes.DEFAULT_TYPE, cache_key: str) -> None:
    await _reply_tracked(
        update,
        cache_key,
        "Is the information the same as the original seller used?",
        reply_markup=build_yes_no_keyboard(cache_key, "same_info"),
    )


async def _cb_same_info(update: Update, context: ContextTypes.DEFAULT_TYPE, cache_key: str, answer: str) -> None:
    if answer == "yes":
        await _reply_tracked(update, cache_key, "💶 What price will you list it for? (send the amount)")
        context.user_data["accountability_state"] = {
            "cache_key": cache_key,
            "stage": "awaiting_listed_price",
            "data": {"same_info": True},
        }
    else:
        await _reply_tracked(update, cache_key, "✏️ Enter the new title:")
        context.user_data["accountability_state"] = {
            "cache_key": cache_key,
            "stage": "awaiting_new_title",
            "data": {"same_info": False},
        }


async def _cb_sold(update: Update, context: ContextTypes.DEFAULT_TYPE, cache_key: str) -> None:
    await _reply_tracked(
        update,
        cache_key,
        "Was the sold price the same as you decided?",
        reply_markup=build_yes_no_keyboard(cache_key, "price_as_decided"),
    )


async def _cb_price_as_decided(update: Update, context: ContextTypes.DEFAULT_TYPE, cache_key: str, answer: str) -> None:
    if answer == "yes":
        row = get_item(cache_key)
        listed_price = row.get("listed_price") if row else ""
        if listed_price:
            await _finish_sold(update, context, cache_key, float(listed_price))
        else:
            await _reply_tracked(update, cache_key, "⚠️ No listed price found. Please enter the sold price:")
            context.user_data["accountability_state"] = {
                "cache_key": cache_key,
                "stage": "awaiting_sold_price",
                "data": {},
            }
    else:
        await _reply_tracked(update, cache_key, "💶 What was the actual sold price? (send the amount)")
        context.user_data["accountability_state"] = {
            "cache_key": cache_key,
            "stage": "awaiting_sold_price",
            "data": {},
        }


async def _finish_sold(update: Update, context: ContextTypes.DEFAULT_TYPE, cache_key: str, sold_price: float) -> None:
    row = get_item(cache_key)
    price_paid = _parse_float(row.get("price_paid") or "0") or 0.0
    profit = calculate_profit(price_paid, sold_price)

    transition(
        cache_key,
        "resold",
        sold_price=sold_price,
        sold_at=_now_iso(),
        profit=profit,
    )

    token = settings.telegram.bot_token
    # Send photo + profit caption to the resold chat (matching every other stage).
    resold_chat = settings.telegram.resolved_resold_chat_id
    if resold_chat:
        item = _item_from_tracker(row) if row else {}
        # Append sold + profit to the Title so build_caption keeps it under the photo.
        profit_line = f"💰 Sold €{sold_price:.2f}  ·  📈 Profit €{profit:.2f}"
        item = dict(item)
        item["Title"] = f"{item.get('Title', '')}\n{profit_line}".strip()
        improved_path = _first_improved_photo_path(row.get("new_images") if row else None)
        if improved_path is not None:
            await _send_local_photo_to_chat(
                token, resold_chat, improved_path, build_caption(item), cache_key
            )
        else:
            await send_item_to_chat(
                token, resold_chat, item, InlineKeyboardMarkup([]), cache_key
            )

    # Delete from selling chat (cleans the card + all tracked replies in selling chat)
    selling_chat = settings.telegram.resolved_selling_items_chat_id
    if selling_chat:
        await delete_item_message(token, cache_key, selling_chat)

    await update_tracking_message(token, cache_key)
    log_event("accountability_item_sold", cache_key=cache_key,
              details={"sold_price": sold_price, "profit": profit})


# ── text input handler (conversation state) ──────────────────────────────────

async def handle_accountability_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Returns True if the message was consumed by accountability state."""
    state = context.user_data.get("accountability_state")
    if not state:
        return False

    msg = update.message
    if msg is None or not msg.text:
        return False

    text = msg.text.strip()
    cache_key = state["cache_key"]
    stage = state["stage"]
    data = state.get("data", {})
    token = settings.telegram.bot_token

    if stage == "awaiting_price_paid":
        price = _parse_float(text)
        if price is None:
            await _reply_tracked(update, cache_key, "❌ Please send a valid number.")
            return True
        # Register the user's own price-paid message so it disappears too.
        register_item_message(cache_key, msg.chat_id, msg.message_id)
        transition(cache_key, "bought", price_paid=price, bought_at=_now_iso())
        try:
            await _send_to_stage(
                cache_key,
                settings.telegram.resolved_bought_items_chat_id,
                build_bought_keyboard,
                token,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to send item to bought chat")
            await _reply_tracked(update, cache_key, f"⚠️ Could not send to bought chat: {exc}")
        recommended_chat = settings.telegram.resolved_recommended_deals_chat_id
        if recommended_chat:
            await delete_item_message(token, cache_key, recommended_chat)
        # Belt-and-suspenders: also delete the user's own message explicitly
        # in case register_item_message failed to persist it.
        try:
            await context.bot.delete_message(chat_id=msg.chat_id, message_id=msg.message_id)
        except BadRequest:
            pass
        await update_tracking_message(token, cache_key)
        del context.user_data["accountability_state"]
        return True

    if stage == "awaiting_listed_price":
        price = _parse_float(text)
        if price is None:
            await _reply_tracked(update, cache_key, "❌ Please send a valid number.")
            return True
        register_item_message(cache_key, msg.chat_id, msg.message_id)
        transition(cache_key, "selling", listed_price=price, listed_at=_now_iso())
        try:
            await _send_to_stage(
                cache_key,
                settings.telegram.resolved_selling_items_chat_id,
                build_selling_keyboard,
                token,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to send item to selling chat")
            await _reply_tracked(update, cache_key, f"⚠️ Could not send to selling chat: {exc}")
        drafting_chat = settings.telegram.resolved_drafting_items_chat_id
        if drafting_chat:
            await delete_item_message(token, cache_key, drafting_chat)
        try:
            await context.bot.delete_message(chat_id=msg.chat_id, message_id=msg.message_id)
        except BadRequest:
            pass
        await update_tracking_message(token, cache_key)
        del context.user_data["accountability_state"]
        return True

    if stage == "awaiting_new_title":
        register_item_message(cache_key, msg.chat_id, msg.message_id)
        data["new_title"] = text
        context.user_data["accountability_state"] = {
            "cache_key": cache_key,
            "stage": "awaiting_new_description",
            "data": data,
        }
        await _reply_tracked(update, cache_key, "✏️ Now enter the new description:")
        return True

    if stage == "awaiting_new_description":
        register_item_message(cache_key, msg.chat_id, msg.message_id)
        data["new_description"] = text
        context.user_data["accountability_state"] = {
            "cache_key": cache_key,
            "stage": "awaiting_new_brand",
            "data": data,
        }
        await _reply_tracked(update, cache_key, "🏷 What is the brand?")
        return True

    if stage == "awaiting_new_brand":
        register_item_message(cache_key, msg.chat_id, msg.message_id)
        data["new_brand"] = text
        context.user_data["accountability_state"] = {
            "cache_key": cache_key,
            "stage": "awaiting_new_size",
            "data": data,
        }
        await _reply_tracked(update, cache_key, "📏 What is the size?")
        return True

    if stage == "awaiting_new_size":
        register_item_message(cache_key, msg.chat_id, msg.message_id)
        data["new_size"] = text
        context.user_data["accountability_state"] = {
            "cache_key": cache_key,
            "stage": "awaiting_new_condition",
            "data": data,
        }
        await _reply_tracked(update, cache_key, "✨ What is the condition?")
        return True

    if stage == "awaiting_new_condition":
        register_item_message(cache_key, msg.chat_id, msg.message_id)
        data["new_condition"] = text
        context.user_data["accountability_state"] = {
            "cache_key": cache_key,
            "stage": "awaiting_new_price",
            "data": data,
        }
        await _reply_tracked(update, cache_key, "💶 What price will you list it for? (send the amount)")
        return True

    if stage == "awaiting_new_price":
        price = _parse_float(text)
        if price is None:
            await _reply_tracked(update, cache_key, "❌ Please send a valid number.")
            return True
        register_item_message(cache_key, msg.chat_id, msg.message_id)
        # "No, info is different" path writes new fields, overwriting the seller's values
        # for brand/size/condition since the user is now using their own info.
        transition(
            cache_key,
            "selling",
            new_title=data.get("new_title", ""),
            new_description=data.get("new_description", ""),
            brand=data.get("new_brand", ""),
            size=data.get("new_size", ""),
            condition=data.get("new_condition", ""),
            listed_price=price,
            listed_at=_now_iso(),
        )
        try:
            await _send_to_stage(
                cache_key,
                settings.telegram.resolved_selling_items_chat_id,
                build_selling_keyboard,
                token,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to send item to selling chat")
            await _reply_tracked(update, cache_key, f"⚠️ Could not send to selling chat: {exc}")
        drafting_chat = settings.telegram.resolved_drafting_items_chat_id
        if drafting_chat:
            await delete_item_message(token, cache_key, drafting_chat)
        try:
            await context.bot.delete_message(chat_id=msg.chat_id, message_id=msg.message_id)
        except BadRequest:
            pass
        await update_tracking_message(token, cache_key)
        del context.user_data["accountability_state"]
        return True

    if stage == "awaiting_sold_price":
        price = _parse_float(text)
        if price is None:
            await _reply_tracked(update, cache_key, "❌ Please send a valid number.")
            return True
        register_item_message(cache_key, msg.chat_id, msg.message_id)
        await _finish_sold(update, context, cache_key, price)
        try:
            await context.bot.delete_message(chat_id=msg.chat_id, message_id=msg.message_id)
        except BadRequest:
            pass
        del context.user_data["accountability_state"]
        return True

    return False


# ── photo handler (image improvement in drafting) ────────────────────────────

async def handle_accountability_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Returns True if the photo (or image-document) was consumed by accountability state."""
    state = context.user_data.get("accountability_state")
    if not state or state.get("stage") != "awaiting_photos":
        return False

    msg = update.message
    if msg is None:
        return False

    cache_key = state["cache_key"]

    # Accept either a compressed photo or an image sent as a "File" (Document.IMAGE).
    photo_obj = None
    if msg.photo:
        photo_obj = msg.photo[-1]
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        photo_obj = msg.document  # has .file_id attribute, works with bot.get_file
    if photo_obj is None:
        return False

    # Handle media groups similarly to image_bot.py
    gid = msg.media_group_id
    if gid:
        if gid not in _media_group_buffer:
            _media_group_buffer[gid] = {"photos": [], "message_ids": [], "update": update, "task": None}
        buf = _media_group_buffer[gid]
        buf["photos"].append(photo_obj)
        buf["message_ids"].append(msg.message_id)
        if buf["task"] is not None:
            buf["task"].cancel()
        buf["task"] = asyncio.create_task(_flush_photo_group(gid, cache_key, context))
        return True

    # Single photo / document
    await _process_photos([photo_obj], cache_key, update, context)
    return True


async def _flush_photo_group(gid: str, cache_key: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    await asyncio.sleep(1.5)
    group = _media_group_buffer.pop(gid, None)
    if not group:
        return
    # Register every user photo message ID so they all get deleted on transition.
    for mid in group.get("message_ids", []):
        register_item_message(cache_key, group["update"].message.chat_id, mid)
    await _process_photos(group["photos"], cache_key, group["update"], context)


async def _process_photos(
    photos: list,
    cache_key: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    n = len(photos)
    start_ts = datetime.now(tz=timezone.utc)
    LOGGER.info("photo_improve start cache_key=%s photos=%d", cache_key, n)
    # For single-photo uploads the message_id is already registered by the caller
    # (handle_accountability_photo or _flush_photo_group). For safety, register it
    # here too — duplicate IDs are harmless because delete_item_message tolerates
    # BadRequest for already-deleted messages.
    register_item_message(cache_key, update.message.chat_id, update.message.message_id)
    status_msg = await update.message.reply_text(
        f"⏳ Improving {n} photo(s) with gpt-image-2… "
        f"typical wait: 1–3 minutes per photo. I'll send each one as it finishes."
    )
    register_item_message(cache_key, status_msg.chat_id, status_msg.message_id)

    # Build item context from the tracker row so the model knows what the item is.
    row = get_item(cache_key)
    item_ctx: str | None = None
    if row:
        def _s(v) -> str:
            # Coerce NaN/None/non-string to "" before .strip().
            return str(v).strip() if v is not None and not (isinstance(v, float) and v != v) else ""
        brand = _s(row.get("brand"))
        title = _s(row.get("new_title")) or _s(row.get("title"))
        item_ctx = f"{brand} {title}".strip() or None

    # Directory for this item's improved photos (find next free index so repeated
    # batches don't overwrite earlier ones).
    out_dir = settings.paths.data_dir / "telegram_implementation" / "improved_images" / cache_key
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_files = sorted(out_dir.glob("improved_*.jpg"))
    next_index = len(existing_files) + 1

    async def _process_one(slot: int, photo_size) -> tuple[bool, int, Exception | None, str | None]:
        t0 = datetime.now(tz=timezone.utc)
        try:
            LOGGER.info("photo_improve[%d] downloading file_id=%s", slot, photo_size.file_id)
            file = await context.bot.get_file(photo_size.file_id)
            img_bytes = bytes(await file.download_as_bytearray())
            LOGGER.info("photo_improve[%d] downloaded %d bytes, calling gpt-image-2…", slot, len(img_bytes))

            improved = await photo_svc.improve_photo(img_bytes, item_ctx)
            elapsed = (datetime.now(tz=timezone.utc) - t0).total_seconds()
            LOGGER.info("photo_improve[%d] received improved bytes=%d in %.1fs", slot, len(improved), elapsed)

            saved_path = out_dir / f"improved_{slot:03d}.jpg"
            saved_path.write_bytes(improved)

            sent = await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=io.BytesIO(improved),
                filename=saved_path.name,
                caption=f"✅ Improved photo {slot - next_index + 1}/{n} ({elapsed:.0f}s)",
            )
            register_item_message(cache_key, sent.chat_id, sent.message_id)
            return True, slot, None, str(saved_path)
        except Exception as exc:  # noqa: BLE001
            elapsed = (datetime.now(tz=timezone.utc) - t0).total_seconds()
            LOGGER.exception("photo_improve[%d] FAILED after %.1fs: %s", slot, elapsed, exc)
            return False, slot, exc, None

    tasks = [
        asyncio.create_task(_process_one(next_index + i, ps))
        for i, ps in enumerate(photos)
    ]
    success = 0
    saved_paths: list[str] = []
    for coro in asyncio.as_completed(tasks):
        ok, slot, exc, path = await coro
        if ok:
            success += 1
            if path:
                saved_paths.append(path)
        else:
            err_msg = await update.message.reply_text(f"❌ Photo: {exc}")
            register_item_message(cache_key, err_msg.chat_id, err_msg.message_id)

    total_elapsed = (datetime.now(tz=timezone.utc) - start_ts).total_seconds()
    LOGGER.info("photo_improve done cache_key=%s success=%d failed=%d total=%.1fs",
                cache_key, success, n - success, total_elapsed)
    # Photos arrive with per-photo captions; no separate Done summary needed.

    # Persist the list of saved improved-photo paths into the tracker.
    if saved_paths:
        existing_raw = (row.get("new_images") if row else "") or ""
        existing: list[str] = []
        if existing_raw:
            try:
                parsed = json.loads(existing_raw)
                if isinstance(parsed, list):
                    existing = [str(p) for p in parsed]
            except json.JSONDecodeError:
                existing = []
        merged = existing + saved_paths
        update_item(cache_key, new_images=json.dumps(merged, ensure_ascii=False))

    log_event("accountability_photos_improved", cache_key=cache_key,
              details={"success_count": success, "failure_count": n - success,
                       "saved_paths": saved_paths})

    # Clear state so the next photo upload outside this flow isn't consumed.
    if "accountability_state" in context.user_data:
        del context.user_data["accountability_state"]
