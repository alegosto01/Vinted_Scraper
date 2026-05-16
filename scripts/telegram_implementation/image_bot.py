"""
Separate Telegram bot for the "Vinted Image Improving" chat.

Runs as its own process with its own bot token (IMAGE_BOT_TOKEN). Handles
text/screenshot context and photo improvement via gpt-image-2.
"""
from __future__ import annotations

import asyncio
import io
import logging
import sys
import time
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import settings
from telegram_implementation.event_log import log_event
import telegram_implementation.photo_improvement_service as photo_svc

LOGGER = logging.getLogger(__name__)

_media_group_buffer: dict[str, dict] = {}
_ITEM_CONTEXT_TTL = 600  # seconds


def is_authorized_user(update: Update) -> bool:
    allowed = settings.telegram.resolved_image_allowed_user_id
    if allowed is None:
        return True
    user = update.effective_user
    return user is not None and user.id == allowed


def _get_stored_item_context(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    entry = context.user_data.get("item_context")
    if entry and (time.time() - entry["ts"]) < _ITEM_CONTEXT_TTL:
        return entry["text"]
    return None


async def _process_photos(
    photos: list,
    item_context: str | None,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.message is None or update.effective_chat is None:
        return

    n = len(photos)
    ctx_note = f" for: {item_context}" if item_context else ""
    await update.message.reply_text(f"⏳ Improving {n} photo(s){ctx_note}…")

    log_event("photo_improvement_requested", details={
        "photo_count": n,
        "item_context": item_context or "",
        "chat_id": update.effective_chat.id,
    })

    async def _process_one(i: int, photo_size) -> tuple[bool, int, Exception | None]:
        try:
            file = await context.bot.get_file(photo_size.file_id)
            img_bytes = bytes(await file.download_as_bytearray())
            improved = await photo_svc.improve_photo(img_bytes, item_context)

            caption = f"✅ Photo {i}/{n}"
            if item_context and i == 1:
                caption += f"\nItem: {item_context}"

            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=io.BytesIO(improved),
                filename=f"improved_{i}.jpg",
                caption=caption,
            )
            return True, i, None
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Photo improvement failed for photo %d", i)
            return False, i, exc

    tasks = [asyncio.create_task(_process_one(i, ps)) for i, ps in enumerate(photos, 1)]
    success = 0
    for coro in asyncio.as_completed(tasks):
        ok, i, exc = await coro
        if ok:
            success += 1
        else:
            await update.message.reply_text(f"❌ Photo {i}/{n}: {exc}")

    log_event("photo_improvement_done", details={
        "success_count": success,
        "failure_count": n - success,
        "item_context": item_context or "",
    })


async def _flush_media_group(gid: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    await asyncio.sleep(1.5)
    group = _media_group_buffer.pop(gid, None)
    if not group:
        return
    item_context = group["caption"] or _get_stored_item_context(context)
    await _process_photos(group["photos"], item_context, group["update"], context)


async def handle_text_context(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized_user(update):
        return
    msg = update.message
    if msg is None or not msg.text:
        return
    text = msg.text.strip()
    if not text:
        return
    context.user_data["item_context"] = {"text": text, "ts": time.time()}
    await msg.reply_text(f"Got it — {text}\nNow send your photos.")


async def handle_document_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized_user(update):
        return
    msg = update.message
    if msg is None or msg.document is None:
        return

    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

    file = await context.bot.get_file(msg.document.file_id)
    img_bytes = bytes(await file.download_as_bytearray())

    item_context = await photo_svc.extract_item_context_from_screenshot(img_bytes)
    if item_context:
        context.user_data["item_context"] = {"text": item_context, "ts": time.time()}
        await msg.reply_text(f"Got it — {item_context}\nNow send your photos.")
    else:
        await msg.reply_text("Couldn’t read the screenshot. Send the item title as text instead.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized_user(update):
        return
    msg = update.message
    if msg is None or not msg.photo:
        return

    photo_size = msg.photo[-1]

    if not msg.media_group_id:
        item_context = msg.caption or _get_stored_item_context(context)
        await _process_photos([photo_size], item_context, update, context)
        return

    gid = msg.media_group_id
    if gid not in _media_group_buffer:
        _media_group_buffer[gid] = {"photos": [], "caption": "", "update": update, "task": None}

    buf = _media_group_buffer[gid]
    buf["photos"].append(photo_size)
    if not buf["caption"] and msg.caption:
        buf["caption"] = msg.caption

    if buf["task"] is not None:
        buf["task"].cancel()
    buf["task"] = asyncio.create_task(_flush_media_group(gid, context))


async def log_application_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    LOGGER.exception("Image bot update failed", exc_info=context.error)
    log_event("image_bot_error", details={"error": str(context.error)})


def build_application() -> Application:
    token = settings.telegram.image_bot_token
    if not token:
        raise RuntimeError(
            "IMAGE_BOT_TOKEN is not configured. Create a new bot via @BotFather, "
            "then add IMAGE_BOT_TOKEN=<token> to your .env."
        )

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_context))
    app.add_error_handler(log_application_error)
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = build_application()
    LOGGER.info("Vinted Image Improving bot starting — polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
