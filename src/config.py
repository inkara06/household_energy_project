"""Project-wide paths and constants (single source of truth)."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RAW_FILE = RAW_DIR / "household_power_consumption.txt"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
PARQUET_FILE = PROCESSED_DIR / "power_minute.parquet"
HOURLY_PARQUET_FILE = PROCESSED_DIR / "power_hourly.parquet"
DB_FILE = PROCESSED_DIR / "energy.db"
METRICS_FILE = RESULTS_DIR / "forecast_metrics.csv"
BENCHMARK_FILE = RESULTS_DIR / "format_benchmark.json"
STATE_FILE = RESULTS_DIR / "menu_state.json"

CSV_SEPARATOR = ";"
MISSING_MARKER = "?"
DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S"
CHUNK_SIZE_ROWS = 500_000

NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# NOTE: names/descriptions come from the UCI dataset documentation.
SUB_METERS = {
    "Sub_metering_1": ("kitchen", "Dishwasher, oven, microwave"),
    "Sub_metering_2": ("laundry", "Washing machine, tumble-drier, fridge, light"),
    "Sub_metering_3": ("water_heater_ac", "Electric water heater and air conditioner"),
    "Sub_metering_other": ("other", "Rest of the household (not sub-metered)"),
}

MINUTES_PER_HOUR = 60
WATTS_PER_KILOWATT = 1000
HOURS_PER_DAY = 24
WEEK_DAYS = 7
YEAR_DAYS = 365

TEST_START_DATE = "2010-01-01"
MOVING_AVERAGE_WINDOW_DAYS = 7
LAG_FEATURE_DAYS = [1, 2, 3, 7, 14]
IQR_MULTIPLIER = 1.5
PEAK_QUANTILE = 0.95

SEASONS = {12: "winter", 1: "winter", 2: "winter",
           3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer",
           9: "autumn", 10: "autumn", 11: "autumn"}
BYTES_PER_MB = 1024 ** 2


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for folder in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
