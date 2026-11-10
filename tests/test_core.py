import numpy as np
import pandas as pd
import pytest

from src import forecast, preprocessing


@pytest.fixture
def daily_series() -> pd.Series:
    index = pd.date_range("2009-01-01", "2010-03-31", freq="D")
    weekly = np.tile([30, 28, 27, 29, 31, 40, 42], len(index) // 7 + 1)[: len(index)]
    return pd.Series(weekly.astype(float), index=index)


def test_naive_uses_previous_value(daily_series):
    _, test = forecast.train_test_split_by_date(daily_series)
    predicted = forecast.naive_forecast(daily_series, test.index)
    assert predicted.iloc[0] == daily_series.loc["2009-12-31"]


def test_seasonal_naive_is_exact_on_pure_weekly_pattern(daily_series):
    _, test = forecast.train_test_split_by_date(daily_series)
    predicted = forecast.seasonal_naive_forecast(daily_series, test.index, 7)
    assert np.allclose(predicted, test)


def test_moving_average_does_not_use_current_value(daily_series):
    _, test = forecast.train_test_split_by_date(daily_series)
    predicted = forecast.moving_average_forecast(daily_series, test.index, 7)
    expected = daily_series.loc["2009-12-25":"2009-12-31"].mean()
    assert predicted.iloc[0] == pytest.approx(expected)


def test_split_is_chronological(daily_series):
    train, test = forecast.train_test_split_by_date(daily_series)
    assert train.index.max() < test.index.min()


def test_split_raises_on_empty_part(daily_series):
    with pytest.raises(ValueError):
        forecast.train_test_split_by_date(daily_series, "2030-01-01")


def test_compare_models_returns_all_models(daily_series):
    metrics, predictions = forecast.compare_models(daily_series)
    assert len(metrics) == 4
    assert {"MAE", "RMSE", "fit_time_s"} <= set(metrics.columns)
    assert predictions.notna().all().all()


def test_forecast_future_length(daily_series):
    future = forecast.forecast_future(daily_series, 5)
    assert len(future) == 5
    assert future.index[0] == daily_series.index[-1] + pd.Timedelta(days=1)


def test_iqr_outliers_detects_extreme_value():
    frame = pd.DataFrame({"x": [1.0, 2.0, 2.0, 3.0, 2.0, 100.0]})
    report = preprocessing.iqr_outliers(frame, ["x"])
    assert report.loc[0, "outliers"] == 1


def _minute_frame(minutes: int) -> pd.DataFrame:
    index = pd.date_range("2008-01-01", periods=minutes, freq="min")
    values = {
        "Global_active_power": 1.2, "Global_reactive_power": 0.1, "Voltage": 240.0,
        "Global_intensity": 5.0, "Sub_metering_1": 1.0, "Sub_metering_2": 2.0, "Sub_metering_3": 3.0,
    }
    return pd.DataFrame(values, index=index).astype("float32")


def test_hourly_energy_equals_mean_power():
    hourly = preprocessing.to_hourly(_minute_frame(120))
    assert hourly["energy_kwh"].iloc[0] == pytest.approx(1.2)
    # 1.2 kW for 60 min = 1200 Wh; sub-meters take 6 Wh/min * 60 = 360 Wh
    assert hourly["Sub_metering_other"].iloc[0] == pytest.approx(840)


def test_hourly_fills_short_gaps():
    minute = _minute_frame(60 * 6)
    minute.iloc[60:120] = np.nan
    hourly = preprocessing.to_hourly(minute)
    assert not hourly.isna().any().any()
