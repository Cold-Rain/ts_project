# Телеграм-бот для прогноза акций

Телеграм-бот, который загружает последние 2 года дневных цен из Yahoo Finance, обучает три модели временных рядов по запросу, сравнивает их по метрикам, выбирает лучшую и строит прогноз на 30 торговых дней с сигналами покупки/продажи и простой симуляцией прибыли.

## Возможности
- Источник данных: Yahoo Finance через `yfinance` (Adj Close при наличии, иначе Close).
- Модели (обучаются на каждый запрос):
  - Ridge Regression с лаговыми признаками.
  - ARIMA (statsmodels).
  - LSTM (TensorFlow/Keras).
- Ridge и LSTM обучаются на лог-доходностях и восстанавливаются в цены для метрик и прогнозов.
- Выбор модели: минимальный средний RMSE по walk-forward фолдам (tie-breaker: минимальный средний MAPE).
- Горизонт прогноза: 30 торговых дней.
- Рекомендации: точки BUY/SELL по локальным минимумам/максимумам прогноза.
- Логирование: CSV для успешных запросов + технический лог.

## Требования
- Python 3.11 (3.10+ тоже подходит). Для Python 3.12 автоматически используется TensorFlow 2.16+.
- Токен Telegram-бота в `.env`.

## Установка
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Создайте `.env`:
```
TELEGRAM_BOT_TOKEN=your_token_here
```

## Запуск бота
```bash
python -m app.main
# или
python app/main.py
```

## Локальный запуск прогноза (без Telegram)
```bash
python -m app.services.forecasting --ticker AAPL --amount 1000 --output forecast.png
```

## Форматы ввода
- Одним сообщением: `AAPL 1000`
- Пошагово: сначала `AAPL`, затем `1000`

## Оценка качества
- Walk-forward оценка до 3 фолдов.
- Каждый фолд тестирует следующие 60 торговых дней.

## Логи
- `logs/requests.csv` — успешные запросы.
- `logs/app.log` — технические логи INFO/WARNING/ERROR.

Поля CSV (минимум):
```
user_id,timestamp,ticker,amount,best_model,metric_name,metric_value,estimated_profit,current_price,forecast_end_price,change_pct
```

Пример строки:
```
123456789,2026-01-26T12:34:56+00:00,AAPL,1000,RidgeLag,RMSE_WF,2.3456,35.12,190.22,198.10,4.14
```

## Дисклеймер
Этот бот предназначен исключительно для учебных целей и не является финансовым советом.
