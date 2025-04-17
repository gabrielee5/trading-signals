
import asyncio
import telegram

token = "7764583995:AAFIbsZUG0Lg_8GDLkdU20y7QY3WGaFikUs"
chat_id = "697220790"

async def send_telegram_message():
    
    telegram_bot = telegram.Bot(token=token)
    await telegram_bot.send_message(
        chat_id=chat_id,
        text="Hello, this is a test message from your bot!",
    )

if __name__ == "__main__":
    asyncio.run(send_telegram_message())