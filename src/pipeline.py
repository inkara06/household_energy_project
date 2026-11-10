"""High-level steps shared by the notebook, the console menu and the dashboard."""
import pandas as pd

from src import config, data_loader, database, forecast, preprocessing


def build_hourly(minute: pd.DataFrame | None = None) -> pd.DataFrame:
    """Minute data -> cleaned hourly data, cached as Parquet."""
    if minute is None:
        minute = data_loader.load_or_build_minute_data()
    hourly = preprocessing.to_hourly(minute)
    hourly.to_parquet(config.HOURLY_PARQUET_FILE)
    return hourly


def load_hourly() -> pd.DataFrame:
    try:
        return pd.read_parquet(config.HOURLY_PARQUET_FILE)
    except FileNotFoundError:
        return build_hourly()


def run_forecasts(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare models on daily data, save metrics to CSV and SQLite."""
    metrics, predictions = forecast.compare_models(daily["energy_kwh"].astype("float64"))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.METRICS_FILE, index=False)
    predictions.to_parquet(config.RESULTS_DIR / "forecast_predictions.parquet")
    if config.DB_FILE.exists():
        database.save_forecast_results(metrics)
    return metrics, predictions


def run_all() -> dict:
    """Full pipeline: load -> clean -> SQLite -> forecasts."""
    config.ensure_dirs()
    hourly = build_hourly()
    counts = database.build_database(hourly)
    daily = preprocessing.to_daily(hourly)
    metrics, _ = run_forecasts(daily)
    return {"table_counts": counts, "metrics": metrics}
