import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv("/home/ale/Desktop/vinted/Vinted_New_Version/scripts/telegram_scripts/bot_env.env")  # Load environment variables from .env file
TOKEN = os.getenv("BOT_TOKEN")  # put BOT_TOKEN=123:ABC in a .env file

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(update.effective_chat.id)  # <- prints to your terminal

    await update.message.reply_text("Hi! I'm alive. Try /help")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Available commands: /start, /help, /echo <text>")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # If user typed: /echo hello
    args = context.args
    if args:
        await update.message.reply_text(" ".join(args))
    else:
        await update.message.reply_text("Usage: /echo <text>")

async def fallback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Replies to any non-command message
    await update.message.reply_text(f"You said: {update.message.text}")

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("echo", echo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fallback))

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
