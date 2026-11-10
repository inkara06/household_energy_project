"""Data-quality audit (types, missing values, outliers) and cleaning / resampling."""
import pandas as pd

from src import config

MAX_INTERPOLATION_HOURS = 3
HOURS_PER_WEEK = config.HOURS_PER_DAY * config.WEEK_DAYS


def feature_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe each column: dtype, kind of feature and number of unique values."""
    rows = []
    for column in frame.columns:
        kind = "numeric (continuous)" if pd.api.types.is_float_dtype(frame[column]) else "other"
        if column.startswith("Sub_metering"):
            kind = "numeric (discrete, Wh per minute)"
        rows.append({
            "column": column,
            "dtype": str(frame[column].dtype),
            "feature_type": kind,
            "unique_values": int(frame[column].nunique()),
        })
    return pd.DataFrame(rows)


def missing_report(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.isna().sum()
    return pd.DataFrame({
        "missing_count": counts,
        "missing_pct": (counts / len(frame) * 100).round(3),
    })


def missing_by_year(frame: pd.DataFrame) -> pd.Series:
    is_missing = frame["Global_active_power"].isna()
    return is_missing.groupby(frame.index.year).mean().mul(100).round(2)


def longest_gaps(frame: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Find the longest consecutive runs of missing minutes."""
    is_missing = frame["Global_active_power"].isna()
    run_id = (is_missing != is_missing.shift()).cumsum()
    runs = frame.index.to_series()[is_missing].groupby(run_id[is_missing]).agg(["min", "max", "count"])
    runs.columns = ["gap_start", "gap_end", "missing_minutes"]
    runs["missing_hours"] = (runs["missing_minutes"] / config.MINUTES_PER_HOUR).round(1)
    return runs.sort_values("missing_minutes", ascending=False).head(top_n).reset_index(drop=True)


def iqr_outliers(frame: pd.DataFrame, columns: list[str] | None = None,
                 multiplier: float = config.IQR_MULTIPLIER) -> pd.DataFrame:
    """Count values outside [Q1 - k*IQR, Q3 + k*IQR] for each column."""
    rows = []
    for column in columns or config.NUMERIC_COLUMNS:
        series = frame[column].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        rows.append({
            "column": column,
            "q1": round(float(q1), 3),
            "q3": round(float(q3), 3),
            "lower_bound": round(float(lower), 3),
            "upper_bound": round(float(upper), 3),
            "outliers": int(outliers),
            "outliers_pct": round(outliers / len(series) * 100, 2),
        })
    return pd.DataFrame(rows)


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add energy not covered by the three sub-meters (Wh per minute)."""
    frame = frame.copy()
    total_wh = frame["Global_active_power"] * config.WATTS_PER_KILOWATT / config.MINUTES_PER_HOUR
    sub_total = frame[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].sum(axis=1, min_count=3)
    frame["Sub_metering_other"] = (total_wh - sub_total).clip(lower=0).astype("float32")
    return frame


def to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Resample minutes to hours and fill gaps.

    - power/voltage/intensity: hourly mean (mean kW over an hour == kWh in that hour);
    - sub-meters: Wh per hour (mean Wh/min * 60);
    - gaps <= 3 h are interpolated in time, longer gaps take the value
      of the same hour one week earlier (keeps daily and weekly shape).
    """
    frame = add_derived_columns(frame)
    hourly = frame.resample("h").mean()
    sub_columns = [c for c in hourly.columns if c.startswith("Sub_metering")]
    hourly[sub_columns] = hourly[sub_columns] * config.MINUTES_PER_HOUR
    hourly = hourly.interpolate(method="time", limit=MAX_INTERPOLATION_HOURS)
    while hourly.isna().any().any():
        hourly = hourly.fillna(hourly.shift(HOURS_PER_WEEK)).fillna(hourly.shift(-HOURS_PER_WEEK))
    return hourly.rename(columns={"Global_active_power": "energy_kwh"}).astype("float32")


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily totals for energy and sub-meters, daily mean for voltage/intensity."""
    sum_columns = ["energy_kwh"] + [c for c in hourly.columns if c.startswith("Sub_metering")]
    daily = hourly[sum_columns].resample("D").sum()
    daily["Voltage"] = hourly["Voltage"].resample("D").mean()
    daily["Global_intensity"] = hourly["Global_intensity"].resample("D").mean()
    daily["Global_reactive_power"] = hourly["Global_reactive_power"].resample("D").mean()
    # NOTE: first and last day of the dataset are incomplete; drop them.
    return daily.iloc[1:-1].astype("float64")


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    frame["weekday"] = frame.index.day_of_week
    frame["hour"] = frame.index.hour
    frame["is_weekend"] = frame["weekday"] >= 5
    frame["season"] = frame["month"].map(config.SEASONS)
    return frame
