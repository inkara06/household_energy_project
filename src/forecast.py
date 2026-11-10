"""Forecasting: naive / seasonal naive baselines vs moving average vs linear regression."""
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import config


@dataclass(frozen=True)
class SeriesSetup:
    """Parameters that depend on the series frequency."""
    season_length: int
    moving_average_window: int
    lags: tuple[int, ...]


DAILY_SETUP = SeriesSetup(
    season_length=config.WEEK_DAYS,
    moving_average_window=config.MOVING_AVERAGE_WINDOW_DAYS,
    lags=tuple(config.LAG_FEATURE_DAYS),
)
HOURLY_SETUP = SeriesSetup(
    season_length=config.HOURS_PER_DAY,
    moving_average_window=config.HOURS_PER_DAY,
    lags=(1, 2, 3, config.HOURS_PER_DAY, config.HOURS_PER_DAY * config.WEEK_DAYS),
)


def train_test_split_by_date(series: pd.Series, test_start: str = config.TEST_START_DATE):
    """Chronological split without shuffling."""
    train = series[series.index < test_start]
    test = series[series.index >= test_start]
    if train.empty or test.empty:
        raise ValueError(f"Split at {test_start} leaves an empty train or test part")
    return train, test


def naive_forecast(series: pd.Series, test_index: pd.Index) -> pd.Series:
    """y_hat(t) = y(t-1)."""
    return series.shift(1).loc[test_index]


def seasonal_naive_forecast(series: pd.Series, test_index: pd.Index, season_length: int) -> pd.Series:
    """y_hat(t) = y(t - season), e.g. the same weekday one week earlier."""
    return series.shift(season_length).loc[test_index]


def moving_average_forecast(series: pd.Series, test_index: pd.Index, window: int) -> pd.Series:
    """y_hat(t) = mean of the previous `window` observations."""
    return series.shift(1).rolling(window).mean().loc[test_index]


def make_features(series: pd.Series, setup: SeriesSetup) -> pd.DataFrame:
    """Lag features, rolling means of the past and calendar dummies (no look-ahead)."""
    features = pd.DataFrame(index=series.index)
    for lag in setup.lags:
        features[f"lag_{lag}"] = series.shift(lag)
    past = series.shift(1)
    features[f"rolling_mean_{setup.moving_average_window}"] = past.rolling(setup.moving_average_window).mean()
    features[f"rolling_mean_{setup.season_length * 4}"] = past.rolling(setup.season_length * 4).mean()
    calendar = pd.DataFrame(index=series.index)
    calendar["weekday"] = series.index.day_of_week
    calendar["month"] = series.index.month
    if setup is HOURLY_SETUP:
        calendar["hour"] = series.index.hour
    dummies = pd.get_dummies(calendar.astype("category"), drop_first=True, dtype="float32")
    return pd.concat([features, dummies], axis=1)


def linear_regression_forecast(series: pd.Series, test_index: pd.Index,
                               setup: SeriesSetup) -> tuple[pd.Series, LinearRegression]:
    """Fit on the training period only, predict one step ahead over the test period."""
    features = make_features(series, setup)
    train_mask = (series.index < test_index[0]) & features.notna().all(axis=1)
    model = LinearRegression()
    model.fit(features[train_mask], series[train_mask])
    predictions = pd.Series(model.predict(features.loc[test_index]), index=test_index)
    return predictions, model


def regression_metrics(actual: pd.Series, predicted: pd.Series, naive_mae: float | None = None) -> dict:
    errors = actual - predicted
    mae = float(np.abs(errors).mean())
    metrics = {
        "MAE": mae,
        "RMSE": float(np.sqrt((errors ** 2).mean())),
        "MAPE_pct": float((np.abs(errors) / actual.abs().clip(lower=1e-6)).mean() * 100),
    }
    if naive_mae:
        metrics["MASE_vs_naive"] = mae / naive_mae
    return metrics


def _timed(func, *args):
    tracemalloc.start()
    start = time.perf_counter()
    result = func(*args)
    elapsed_s = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed_s, peak_bytes / config.BYTES_PER_MB


def compare_models(series: pd.Series, setup: SeriesSetup = DAILY_SETUP,
                   test_start: str = config.TEST_START_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all models on the hold-out period. Returns (metrics table, predictions table)."""
    _, test = train_test_split_by_date(series, test_start)
    models = {
        "Naive (baseline)": lambda: naive_forecast(series, test.index),
        "Seasonal naive": lambda: seasonal_naive_forecast(series, test.index, setup.season_length),
        f"Moving average ({setup.moving_average_window})":
            lambda: moving_average_forecast(series, test.index, setup.moving_average_window),
        "Linear regression (lags)": lambda: linear_regression_forecast(series, test.index, setup)[0],
    }
    predictions = pd.DataFrame({"actual": test})
    rows = []
    naive_mae = None
    for name, run in models.items():
        predicted, elapsed_s, peak_mb = _timed(run)
        predictions[name] = predicted
        metrics = regression_metrics(test, predicted, naive_mae)
        if naive_mae is None:
            naive_mae = metrics["MAE"]
            metrics["MASE_vs_naive"] = 1.0
        rows.append({"model": name, **metrics,
                     "fit_time_s": elapsed_s, "peak_memory_mb": peak_mb})
    metrics_table = pd.DataFrame(rows).round(4).sort_values("MAE").reset_index(drop=True)
    return metrics_table, predictions


def errors_by_period(predictions: pd.DataFrame, model: str, freq: str = "ME") -> pd.DataFrame:
    """MAE of each model per month (error analysis over time)."""
    abs_errors = predictions.drop(columns="actual").sub(predictions["actual"], axis=0).abs()
    table = abs_errors.resample(freq).mean().round(3)
    table["best_model"] = table.idxmin(axis=1)
    return table[[model] + [c for c in table.columns if c != model]]


def forecast_future(series: pd.Series, horizon: int, setup: SeriesSetup = DAILY_SETUP) -> pd.Series:
    """Recursive multi-step forecast with linear regression trained on the whole history."""
    if horizon < 1:
        raise ValueError("Horizon must be a positive number of steps")
    step = series.index[-1] - series.index[-2]
    history = series.copy()
    features = make_features(history, setup)
    mask = features.notna().all(axis=1)
    model = LinearRegression().fit(features[mask], history[mask])
    for _ in range(horizon):
        next_time = history.index[-1] + step
        history.loc[next_time] = np.nan
        next_features = make_features(history, setup).reindex(columns=features.columns, fill_value=0.0)
        history.loc[next_time] = float(model.predict(next_features.iloc[[-1]])[0])
    return history.iloc[-horizon:]
