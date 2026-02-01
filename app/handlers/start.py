from __future__ import annotations

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

router = Router()

DISCLAIMER = "Дисклеймер: исключительно в учебных целях, не является финансовым советом."


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    text = (
        "Отправьте тикер и сумму, чтобы получить прогноз на 30 торговых дней.\n"
        "Форматы:\n"
        "- AAPL 1000\n"
        "- или сначала тикер, затем сумма\n\n"
        f"{DISCLAIMER}"
    )
    await message.answer(text)
