from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.bot import create_bot
from app.handlers.predict import router as predict_router
from app.handlers.start import router as start_router
from app.services.logging_service import setup_logging


def run_bot() -> None:
    logger = setup_logging()
    bot, dispatcher = create_bot()
    dispatcher.include_router(start_router)
    dispatcher.include_router(predict_router)
    logger.info("Bot started")
    asyncio.run(dispatcher.start_polling(bot))


if __name__ == "__main__":
    run_bot()
