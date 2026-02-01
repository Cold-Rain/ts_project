from __future__ import annotations

import asyncio
import logging
import re

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import BufferedInputFile, Message

from app.config import DATE_FORMAT
from app.services.forecasting import run_forecast
from app.services.logging_service import LOGGER_NAME

router = Router()
logger = logging.getLogger(LOGGER_NAME)

DISCLAIMER = "Дисклеймер: исключительно в учебных целях, не является финансовым советом."


class PredictStates(StatesGroup):
    waiting_for_ticker = State()
    waiting_for_amount = State()


def _parse_ticker(raw: str) -> str | None:
    candidate = raw.strip().upper()
    if re.fullmatch(r"[A-Z0-9][A-Z0-9\-\.]{0,14}", candidate):
        return candidate
    return None


def _parse_amount(raw: str) -> float | None:
    cleaned = raw.strip().replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _parse_ticker_amount(text: str) -> tuple[str, float] | None:
    parts = text.strip().split()
    if len(parts) < 2:
        return None
    ticker = _parse_ticker(parts[0])
    amount = _parse_amount(parts[1])
    if ticker and amount:
        return ticker, amount
    return None


def _format_signals(signals, limit: int = 6) -> str:
    if not signals:
        return "Сигналы в окне прогноза не найдены."

    lines = []
    for signal in signals[:limit]:
        date_str = signal.date.strftime(DATE_FORMAT)
        action = "ПОКУПКА" if signal.action == "BUY" else "ПРОДАЖА"
        lines.append(f"{action} {date_str} по {signal.price:.2f}")

    if len(signals) > limit:
        lines.append("...")
    return "\n".join(lines)


def _format_results(result) -> str:
    forecast_end_date = result.forecast.index[-1].strftime(DATE_FORMAT)
    forecast_end_price = float(result.forecast.iloc[-1])
    signals_text = _format_signals(result.signals)

    text = (
        f"Тикер: {result.ticker}\n"
        f"Текущая цена: {result.current_price:.2f}\n"
        f"Лучшая модель: {result.best_model}\n"
        f"RMSE (ср., {result.folds} фолда): {result.best_rmse:.4f} | MAPE (ср.): {result.best_mape:.2f}%\n"
        f"Конец прогноза (30 торговых дней, {forecast_end_date}): {forecast_end_price:.2f}\n"
        f"Изменение: {result.change_abs:.2f} ({result.change_pct:.2f}%)\n"
        f"Оценка прибыли: {result.profit_abs:.2f} ({result.profit_pct:.2f}%)\n\n"
        f"Сигналы:\n{signals_text}\n\n"
        f"{DISCLAIMER}"
    )
    return text


async def _run_forecast(message: Message, ticker: str, amount: float) -> None:
    await message.answer("Запускаю анализ. Это может занять минуту...")

    try:
        result = await asyncio.to_thread(
            run_forecast,
            ticker,
            amount,
            str(message.from_user.id) if message.from_user else "unknown",
            logger,
        )
    except ValueError as exc:
        logger.warning("User input error for %s: %s", ticker, exc)
        await message.answer(f"Ошибка ввода: {exc}")
        return
    except RuntimeError as exc:
        logger.error("Data error for %s: %s", ticker, exc)
        await message.answer(f"Ошибка данных: {exc}")
        return
    except Exception as exc:
        logger.error("Forecast failed for %s: %s", ticker, exc)
        await message.answer("Не удалось построить прогноз. Попробуйте позже.")
        return

    photo = BufferedInputFile(result.plot_png.getvalue(), filename=f"{ticker}.png")
    await message.answer_photo(photo)
    await message.answer(_format_results(result))


@router.message(F.text)
async def text_handler(message: Message, state: FSMContext) -> None:
    text = message.text.strip()

    parsed = _parse_ticker_amount(text)
    if parsed:
        ticker, amount = parsed
        await state.clear()
        await _run_forecast(message, ticker, amount)
        return

    current_state = await state.get_state()
    if current_state == PredictStates.waiting_for_ticker.state:
        ticker = _parse_ticker(text)
        if not ticker:
            await message.answer("Некорректный тикер. Попробуйте ещё раз (например, AAPL).")
            return
        await state.update_data(ticker=ticker)
        await state.set_state(PredictStates.waiting_for_amount)
        await message.answer("Принято. Теперь отправьте сумму (например, 1000).")
        return

    if current_state == PredictStates.waiting_for_amount.state:
        amount = _parse_amount(text)
        if amount is None:
            await message.answer("Некорректная сумма. Введите число больше 0.")
            return
        data = await state.get_data()
        ticker = data.get("ticker")
        await state.clear()
        if not ticker:
            await message.answer("Тикер не найден. Пожалуйста, отправьте его ещё раз.")
            return
        await _run_forecast(message, ticker, amount)
        return

    ticker = _parse_ticker(text)
    if ticker:
        await state.update_data(ticker=ticker)
        await state.set_state(PredictStates.waiting_for_amount)
        await message.answer("Отправьте сумму для инвестиции (например, 1000).")
        return

    await state.set_state(PredictStates.waiting_for_ticker)
    await message.answer("Сначала отправьте тикер (например, AAPL).")
