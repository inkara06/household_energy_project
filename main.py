"""Console menu for the Household Energy Consumption project.

Run:  python main.py
"""
import json
from datetime import datetime

import pandas as pd

from src import config, data_loader, database, forecast, pipeline, preprocessing, timeseries

MENU = {
    "1": "Load data (download -> chunked read -> Parquet) and show audit",
    "2": "Build SQLite database",
    "3": "Run SQL query",
    "4": "Show load profiles and peak hours",
    "5": "Compare forecasting models (MAE / RMSE / time / memory)",
    "6": "Forecast next N days",
    "7": "Statistics for a date range",
    "8": "Benchmark CSV vs Parquet",
    "0": "Exit",
}


def load_state() -> dict:
    try:
        return json.loads(config.STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"history": []}


def save_state(state: dict, action: str) -> None:
    state["history"] = (state["history"] + [{"action": action, "at": datetime.now().isoformat(timespec="seconds")}])[-20:]
    config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def ask_int(prompt: str, minimum: int, maximum: int) -> int:
    while True:
        raw = input(f"{prompt} [{minimum}-{maximum}]: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"  Value must be between {minimum} and {maximum}.")


def ask_date(prompt: str) -> str:
    while True:
        raw = input(f"{prompt} (YYYY-MM-DD): ").strip()
        try:
            return datetime.strptime(raw, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            print("  Wrong date format, example: 2008-02-15")


def action_load() -> None:
    minute = data_loader.load_or_build_minute_data()
    print(f"Rows: {len(minute):,}   Memory: {data_loader.memory_mb(minute):.1f} MB")
    print(f"Period: {minute.index.min()} -> {minute.index.max()}")
    print("\nMissing values:\n", preprocessing.missing_report(minute))
    print("\nIQR outliers:\n", preprocessing.iqr_outliers(minute).to_string(index=False))
    pipeline.build_hourly(minute)
    print(f"\nHourly data saved to {config.HOURLY_PARQUET_FILE}")


def action_build_db() -> None:
    counts = database.build_database(pipeline.load_hourly())
    for table, rows in counts.items():
        print(f"  {table:<20} {rows:>8,} rows")


def action_query() -> None:
    names = list(database.QUERIES)
    for number, name in enumerate(names, start=1):
        print(f"  {number:>2}. {database.QUERIES[name][0]}")
    choice = ask_int("Query number", 1, len(names))
    try:
        result = database.run_query(names[choice - 1])
    except Exception as error:  # NOTE: sqlite3/pandas raise several error types; show any to the user.
        print(f"  Query failed: {error}. Build the database first (menu item 2).")
        return
    with pd.option_context("display.max_rows", 40, "display.width", 140):
        print(result)


def action_profiles() -> None:
    hourly = pipeline.load_hourly()
    daily = preprocessing.to_daily(hourly)
    print("Daily profile (kWh per hour):\n", timeseries.daily_profile(hourly).round(3))
    print("\nWeekly profile (kWh per day):\n", timeseries.weekly_profile(daily).round(2))
    print("\nSeasonal profile:\n", timeseries.seasonal_profile(daily))
    peaks = timeseries.peak_hours(hourly)
    print(f"\nPeak threshold (95th percentile): {peaks['threshold_kwh']} kWh, peaks: {peaks['peak_count']}")
    print("Most frequent peak hours:\n", peaks["by_hour"].sort_values(ascending=False).head(5))


def action_compare() -> None:
    daily = preprocessing.to_daily(pipeline.load_hourly())
    metrics, _ = pipeline.run_forecasts(daily)
    print(metrics.to_string(index=False))
    best = metrics.iloc[0]
    print(f"\nBest model: {best['model']} (MAE {best['MAE']:.3f} kWh/day). Saved to {config.METRICS_FILE}")


def action_future() -> None:
    days = ask_int("How many days ahead", 1, 30)
    daily = preprocessing.to_daily(pipeline.load_hourly())
    result = forecast.forecast_future(daily["energy_kwh"].astype("float64"), days)
    print(result.round(2).to_string())


def action_range_stats() -> None:
    start = ask_date("Start date")
    end = ask_date("End date")
    if start > end:
        print("  Start date is after end date — swapping them.")
        start, end = end, start
    try:
        frame = database.query_period(start, end)
    except Exception as error:  # NOTE: database may be missing or locked.
        print(f"  Could not read database: {error}")
        return
    if frame.empty:
        print("  No data in this range (dataset covers 2006-12-16 .. 2010-11-26).")
        return
    daily_kwh = frame.groupby("date")["energy_kwh"].sum()
    print(f"Days: {len(daily_kwh)}  Total: {daily_kwh.sum():.1f} kWh  "
          f"Average: {daily_kwh.mean():.2f} kWh/day  Max day: {daily_kwh.idxmax()} ({daily_kwh.max():.2f} kWh)")


def action_benchmark() -> None:
    print("Reading the file three times, this takes about a minute...")
    print(pd.DataFrame(data_loader.benchmark_formats()).to_string(index=False))


ACTIONS = {
    "1": action_load, "2": action_build_db, "3": action_query, "4": action_profiles,
    "5": action_compare, "6": action_future, "7": action_range_stats, "8": action_benchmark,
}


def main() -> None:
    config.ensure_dirs()
    state = load_state()
    if state["history"]:
        print(f"Last action: {state['history'][-1]['action']} at {state['history'][-1]['at']}")
    while True:
        print("\n=== Household Energy Consumption ===")
        for key, label in MENU.items():
            print(f" {key} - {label}")
        choice = input("Choose: ").strip()
        if choice == "0":
            print("Bye!")
            break
        elif choice in ACTIONS:
            try:
                ACTIONS[choice]()
                save_state(state, MENU[choice])
            except FileNotFoundError as error:
                print(f"  File not found: {error}")
            except KeyboardInterrupt:
                print("\n  Cancelled.")
        else:
            print("  Unknown option, try again.")


if __name__ == "__main__":
    main()
