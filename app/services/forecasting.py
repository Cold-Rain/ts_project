from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List

import numpy as np
import pandas as pd

from app.config import DATE_FORMAT, FORECAST_HORIZON
from app.models.ml_ridge import RidgeLagModel
from app.models.stats_ets_or_arima import ARIMAModel
from app.services.data_loader import load_price_series
from app.services.logging_service import log_request
from app.services.plotting import plot_forecast
from app.services.preprocessing import generate_walk_forward_folds
from app.services.recommendations import Signal, generate_signals, simulate_strategy
from app.utils.dates import future_business_dates
from app.utils.metrics import mape, rmse


@dataclass
class ModelResult:
    name: str
    rmse: float
    mape: float
    predictions: np.ndarray
    folds: int


@dataclass
class ForecastResult:
    ticker: str
    current_price: float
    best_model: str
    best_rmse: float
    best_mape: float
    folds: int
    forecast: pd.Series
    signals: List[Signal]
    profit_abs: float
    profit_pct: float
    change_abs: float
    change_pct: float
    plot_png: BytesIO
    test_results: List[ModelResult]


def _evaluate_models(series: pd.Series, logger) -> list[tuple[object, ModelResult]]:
    folds = generate_walk_forward_folds(series)
    candidates: list[object] = [RidgeLagModel, ARIMAModel]
    try:
        from app.models.nn_lstm import LSTMForecastModel

        candidates.append(LSTMForecastModel)
    except Exception as exc:
        logger.warning("LSTM model unavailable: %s", exc)
    results: list[tuple[object, ModelResult]] = []

    for model_factory in candidates:
        try:
            fold_rmses: list[float] = []
            fold_mapes: list[float] = []
            last_preds: np.ndarray = np.array([])
            model_name: str | None = None
            for train, test in folds:
                model = model_factory()
                model.fit(train)
                model_name = model.name
                preds = model.predict(len(test))
                last_preds = preds
                fold_rmses.append(rmse(test.values, preds))
                fold_mapes.append(mape(test.values, preds))

            score_rmse = float(np.mean(fold_rmses))
            score_mape = float(np.mean(fold_mapes))
            result = ModelResult(
                name=model_name or model_factory().name,
                rmse=score_rmse,
                mape=score_mape,
                predictions=last_preds,
                folds=len(folds),
            )
            results.append((model_factory, result))
        except Exception as exc:
            logger.warning("Model %s failed: %s", model_factory.__name__, exc)
            continue

    if not results:
        raise RuntimeError("All models failed to train or predict.")

    return results


def _pick_best(results: list[tuple[object, ModelResult]]) -> tuple[object, ModelResult]:
    sorted_results = sorted(results, key=lambda item: (item[1].rmse, item[1].mape))
    return sorted_results[0]


def run_forecast(
    ticker: str, amount: float, user_id: str, logger
) -> ForecastResult:
    """Run data loading, model selection, forecasting, and logging."""
    series = load_price_series(ticker, logger=logger)
    results = _evaluate_models(series, logger)
    best_model_factory, best_result = _pick_best(results)

    best_model = best_model_factory()
    best_model.fit(series)
    best_model_name = best_model.name
    forecast_values = best_model.predict(FORECAST_HORIZON)

    future_dates = future_business_dates(series.index[-1], FORECAST_HORIZON)
    forecast = pd.Series(forecast_values, index=future_dates)

    signals = generate_signals(forecast, order=2, smooth_window=3)
    simulation = simulate_strategy(amount, forecast, signals)

    current_price = float(series.iloc[-1])
    forecast_end = float(forecast.iloc[-1])
    change_abs = forecast_end - current_price
    change_pct = (change_abs / current_price) * 100.0 if current_price else 0.0

    plot_png = plot_forecast(ticker, series, forecast, signals, best_model_name)

    log_request(
        {
            "user_id": user_id,
            "ticker": ticker,
            "amount": amount,
            "best_model": best_model_name,
            "metric_name": "RMSE_WF",
            "metric_value": round(best_result.rmse, 6),
            "estimated_profit": round(simulation["profit_abs"], 6),
            "current_price": round(current_price, 6),
            "forecast_end_price": round(forecast_end, 6),
            "change_pct": round(change_pct, 6),
        }
    )

    return ForecastResult(
        ticker=ticker,
        current_price=current_price,
        best_model=best_model_name,
        best_rmse=best_result.rmse,
        best_mape=best_result.mape,
        folds=best_result.folds,
        forecast=forecast,
        signals=signals,
        profit_abs=simulation["profit_abs"],
        profit_pct=simulation["profit_pct"],
        change_abs=change_abs,
        change_pct=change_pct,
        plot_png=plot_png,
        test_results=[result for _, result in results],
    )


def _print_summary(result: ForecastResult) -> None:
    print(f"Ticker: {result.ticker}")
    print(f"Current price: {result.current_price:.2f}")
    print(
        f"Best model: {result.best_model} | RMSE(avg, {result.folds} folds)={result.best_rmse:.4f} | MAPE(avg)={result.best_mape:.2f}%"
    )
    end_price = float(result.forecast.iloc[-1])
    print(f"Forecast end ({result.forecast.index[-1].strftime(DATE_FORMAT)}): {end_price:.2f}")
    print(
        f"Change: {result.change_abs:.2f} ({result.change_pct:.2f}%)"
    )
    print(
        f"Estimated profit: {result.profit_abs:.2f} ({result.profit_pct:.2f}%)"
    )


def main() -> None:
    import argparse
    from app.services.logging_service import setup_logging

    parser = argparse.ArgumentParser(description="Run local forecast without Telegram.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--amount", required=True, type=float, help="Investment amount")
    parser.add_argument(
        "--output",
        default="forecast.png",
        help="Output PNG path for the chart",
    )
    args = parser.parse_args()

    logger = setup_logging()
    result = run_forecast(args.ticker.upper(), args.amount, user_id="cli", logger=logger)
    _print_summary(result)

    with open(args.output, "wb") as file:
        file.write(result.plot_png.getbuffer())
    print(f"Chart saved to {args.output}")


if __name__ == "__main__":
    main()
