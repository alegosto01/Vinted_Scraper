from __future__ import annotations

import html
import sys
from pathlib import Path
import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import settings
from telegram_implementation.description_service import generate_description_for_item
from telegram_implementation.event_log import log_event
from telegram_implementation.item_cache import load_cached_item, load_description_payload, save_description_payload
from telegram_implementation.notify import build_description_actions_keyboard, build_description_message

LOGGER = logging.getLogger(__name__)
CALLBACK_PREFIX = "generate_description:"
REGENERATE_PREFIX = "regenerate_description:"


def is_authorized_user(update: Update) -> bool:
    allowed_user_id = settings.telegram.resolved_allowed_user_id
    if allowed_user_id is None:
        return True

    effective_user = update.effective_user
    return effective_user is not None and effective_user.id == allowed_user_id


async def reject_unauthorized_update(update: Update) -> None:
    log_event(
        "unauthorized_telegram_access",
        details={
            "user_id": getattr(update.effective_user, "id", None),
            "chat_id": getattr(update.effective_chat, "id", None),
        },
    )

    if update.callback_query and update.callback_query.message:
        try:
            await update.callback_query.answer("Not authorized.", show_alert=True)
        except BadRequest:
            pass
        await update.callback_query.message.reply_text("This bot is restricted to one authorized Telegram account.")
        return

    if update.message:
        await update.message.reply_text("This bot is restricted to one authorized Telegram account.")



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized_user(update):
        await reject_unauthorized_update(update)
        return
    if update.message:
        await update.message.reply_text("Telegram description bot is running. Use the Generate Description button on item messages.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized_user(update):
        await reject_unauthorized_update(update)
        return
    if update.message:
        await update.message.reply_text("Supported flow: press Show Generated Description on an item notification, then use Regenerate or Copy on the returned draft.")


async def chatid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        chat = update.message.chat
        await update.message.reply_text(f"chat_id: {chat.id}\ntype: {chat.type}\ntitle: {getattr(chat, 'title', '-')}")


async def generate_description_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized_user(update):
        await reject_unauthorized_update(update)
        return

    query = update.callback_query
    if query is None or query.message is None or query.data is None:
        return

    is_regenerate = query.data.startswith(REGENERATE_PREFIX)
    try:
        await query.answer("Regenerating description..." if is_regenerate else "Opening generated description...")
    except BadRequest as exc:
        if "Query is too old" in str(exc) or "query id is invalid" in str(exc):
            await query.message.reply_text("That button expired. Use a fresh item message and try again.")
            return
        raise
    if is_regenerate:
        cache_key = query.data.removeprefix(REGENERATE_PREFIX)
    else:
        cache_key = query.data.removeprefix(CALLBACK_PREFIX)

    try:
        item = load_cached_item(cache_key)
    except FileNotFoundError:
        await query.message.reply_text("I could not find the cached item for this button. Send the item again and retry.")
        log_event("button_error", cache_key=cache_key, details={"reason": "cache_miss", "action": "regenerate" if is_regenerate else "generate"})
        return

    event_type = "description_regenerated" if is_regenerate else "description_requested"
    log_event(event_type, item=item, cache_key=cache_key)

    payload = load_description_payload(cache_key)
    if is_regenerate or payload is None:
        await context.bot.send_chat_action(chat_id=query.message.chat_id, action=ChatAction.TYPING)

        try:
            payload = generate_description_for_item(
                item,
                ollama_language="english",
            )
            save_description_payload(cache_key, payload)
            log_event("description_generated", item=item, cache_key=cache_key,
                      details={"mode": payload.get("generation_mode"), "model": payload.get("model_name")})
        except Exception as exc:  # noqa: BLE001
            error_text = html.escape(str(exc))
            await query.message.reply_text(
                f"<b>Description generation failed.</b>\n{error_text}",
                parse_mode=ParseMode.HTML,
            )
            log_event("description_error", item=item, cache_key=cache_key, details={"error": str(exc)})
            return

    description = str(payload.get("generated_description") or "").strip()
    await query.message.reply_text(
        build_description_message(item, payload),
        reply_markup=build_description_actions_keyboard(cache_key, description),
        reply_to_message_id=query.message.message_id,
    )


async def log_application_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    LOGGER.exception("Telegram bot update failed", exc_info=context.error)
    log_event("bot_error", details={"error": str(context.error)})


def build_application() -> Application:
    token = settings.telegram.bot_token
    if not token:
        raise RuntimeError("BOT_TOKEN is not configured. Check scripts/telegram_scripts/bot_env.env or .env.")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("chatid", chatid_cmd))
    app.add_handler(CallbackQueryHandler(generate_description_callback, pattern=rf"^({CALLBACK_PREFIX}|{REGENERATE_PREFIX})"))
    app.add_error_handler(log_application_error)
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = build_application()
    LOGGER.info("Bot starting — polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
