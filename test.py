
import asyncio
import telegram

token = "abs"
chat_id = "sdjfn"

async def send_telegram_message():
    
    telegram_bot = telegram.Bot(token=token)
    await telegram_bot.send_message(
        chat_id=chat_id,
        text="Hello, this is a test message from your bot!",
    )

if __name__ == "__main__":
    asyncio.run(send_telegram_message())