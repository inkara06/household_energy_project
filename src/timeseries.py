"""Time-series analysis: load profiles, rolling means, decomposition, peaks, correlation."""
import pandas as pd
from statsmodels.tsa.seasonal import STL, DecomposeResult, seasonal_decompose

from src import config

MONTHS_PER_YEAR = 12
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def daily_profile(hourly: pd.DataFrame) -> pd.DataFrame:
    """Average consumption per hour of the day, weekdays vs weekends."""
    is_weekend = hourly.index.day_of_week >= 5
    profile = hourly["energy_kwh"].groupby([hourly.index.hour, is_weekend]).mean().unstack()
    profile.columns = ["weekday" if not col else "weekend" for col in profile.columns]
    profile.index.name = "hour"
    return profile


def weekly_profile(daily: pd.DataFrame) -> pd.Series:
    profile = daily["energy_kwh"].groupby(daily.index.day_of_week).mean()
    profile.index = WEEKDAY_NAMES
    return profile


def seasonal_profile(daily: pd.DataFrame) -> pd.DataFrame:
    seasons = daily.index.month.map(config.SEASONS).rename("season")
    return daily.groupby(seasons)["energy_kwh"].agg(["mean", "median", "std", "max"]).round(2)


def heatmap_hour_month(hourly: pd.DataFrame) -> pd.DataFrame:
    return hourly["energy_kwh"].groupby([hourly.index.hour, hourly.index.month]).mean().unstack()


def rolling_means(daily: pd.DataFrame, windows: tuple[int, ...] = (7, 30, 365)) -> pd.DataFrame:
    result = pd.DataFrame({"daily_kwh": daily["energy_kwh"]})
    for window in windows:
        result[f"rolling_{window}d"] = daily["energy_kwh"].rolling(window, min_periods=window).mean()
    return result


def decompose_weekly(daily: pd.DataFrame) -> DecomposeResult:
    """STL decomposition of daily energy with a weekly period."""
    return STL(daily["energy_kwh"], period=config.WEEK_DAYS, robust=True).fit()


def decompose_yearly(daily: pd.DataFrame) -> DecomposeResult:
    """Classical additive decomposition of monthly totals with a yearly period."""
    monthly = daily["energy_kwh"].resample("ME").sum()
    return seasonal_decompose(monthly, model="additive", period=MONTHS_PER_YEAR)


def peak_hours(hourly: pd.DataFrame, quantile: float = config.PEAK_QUANTILE) -> dict:
    """Peak load statistics: threshold, share of peaks per hour/weekday/season, top peaks."""
    threshold = float(hourly["energy_kwh"].quantile(quantile))
    peaks = hourly[hourly["energy_kwh"] >= threshold]
    by_hour = peaks.groupby(peaks.index.hour).size()
    by_weekday = peaks.groupby(peaks.index.day_of_week).size()
    by_weekday.index = [WEEKDAY_NAMES[i] for i in by_weekday.index]
    by_season = peaks.groupby(peaks.index.month.map(config.SEASONS)).size()
    top = hourly["energy_kwh"].nlargest(10).round(3)
    return {
        "threshold_kwh": round(threshold, 3),
        "peak_count": len(peaks),
        "by_hour": by_hour,
        "by_weekday": by_weekday,
        "by_season": by_season,
        "top_peaks": top,
    }


def sub_meter_shares(daily: pd.DataFrame) -> pd.Series:
    """Share of total energy consumed by every sub-meter group (%)."""
    columns = [column for column in config.SUB_METERS if column in daily.columns]
    totals = daily[columns].sum().astype("float64")
    totals.index = [config.SUB_METERS[column][0] for column in columns]
    return (totals / totals.sum() * 100).round(2)


def correlation_matrix(hourly: pd.DataFrame) -> pd.DataFrame:
    return hourly.select_dtypes("number").corr().round(3)
