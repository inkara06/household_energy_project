"""SQLite storage: normalised schema, loading and analytical SQL queries."""
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from src import config

SCHEMA_SQL = """
DROP TABLE IF EXISTS sub_meter_readings;
DROP TABLE IF EXISTS hourly_readings;
DROP TABLE IF EXISTS sub_meters;
DROP TABLE IF EXISTS calendar;

CREATE TABLE calendar (
    date_id     INTEGER PRIMARY KEY,
    date        TEXT NOT NULL UNIQUE,
    year        INTEGER NOT NULL,
    month       INTEGER NOT NULL,
    day         INTEGER NOT NULL,
    weekday     INTEGER NOT NULL,          -- 0 = Monday
    is_weekend  INTEGER NOT NULL CHECK (is_weekend IN (0, 1)),
    season      TEXT NOT NULL
);

CREATE TABLE hourly_readings (
    reading_id      INTEGER PRIMARY KEY,
    date_id         INTEGER NOT NULL REFERENCES calendar(date_id),
    hour            INTEGER NOT NULL CHECK (hour BETWEEN 0 AND 23),
    energy_kwh      REAL NOT NULL,
    reactive_power  REAL,
    voltage         REAL,
    intensity       REAL,
    UNIQUE (date_id, hour)
);

CREATE TABLE sub_meters (
    meter_id     INTEGER PRIMARY KEY,
    code         TEXT NOT NULL UNIQUE,
    name         TEXT NOT NULL,
    description  TEXT
);

CREATE TABLE sub_meter_readings (
    date_id    INTEGER NOT NULL REFERENCES calendar(date_id),
    hour       INTEGER NOT NULL,
    meter_id   INTEGER NOT NULL REFERENCES sub_meters(meter_id),
    energy_wh  REAL NOT NULL,
    PRIMARY KEY (date_id, hour, meter_id)
);

CREATE INDEX idx_calendar_year_month ON calendar(year, month);
CREATE INDEX idx_hourly_date ON hourly_readings(date_id);
CREATE INDEX idx_sub_meter ON sub_meter_readings(meter_id);
"""

ER_DIAGRAM = """
erDiagram
    calendar ||--o{ hourly_readings : "date_id"
    calendar ||--o{ sub_meter_readings : "date_id"
    sub_meters ||--o{ sub_meter_readings : "meter_id"
    calendar {
        int date_id PK
        text date
        int year
        int month
        int day
        int weekday
        int is_weekend
        text season
    }
    hourly_readings {
        int reading_id PK
        int date_id FK
        int hour
        real energy_kwh
        real reactive_power
        real voltage
        real intensity
    }
    sub_meters {
        int meter_id PK
        text code
        text name
        text description
    }
    sub_meter_readings {
        int date_id FK
        int hour
        int meter_id FK
        real energy_wh
    }
    forecast_results {
        text model
        real mae
        real rmse
        real mape
        real fit_time_s
        real peak_memory_mb
    }
"""


def connect(db_path: Path = config.DB_FILE) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _calendar_table(hourly: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(hourly.index.normalize().unique())
    return pd.DataFrame({
        "date_id": range(1, len(dates) + 1),
        "date": dates.strftime("%Y-%m-%d"),
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "weekday": dates.day_of_week,
        "is_weekend": (dates.day_of_week >= 5).astype(int),
        "season": dates.month.map(config.SEASONS),
    })


def build_database(hourly: pd.DataFrame, db_path: Path = config.DB_FILE) -> dict:
    """Create the schema and load cleaned hourly data. Returns row counts per table."""
    calendar = _calendar_table(hourly)
    date_ids = dict(zip(calendar["date"], calendar["date_id"]))
    keys = pd.DataFrame({
        "date_id": hourly.index.strftime("%Y-%m-%d").map(date_ids),
        "hour": hourly.index.hour,
    })
    readings = keys.assign(
        energy_kwh=hourly["energy_kwh"].to_numpy(),
        reactive_power=hourly["Global_reactive_power"].to_numpy(),
        voltage=hourly["Voltage"].to_numpy(),
        intensity=hourly["Global_intensity"].to_numpy(),
    )
    meters = pd.DataFrame([
        {"meter_id": i, "code": code, "name": name, "description": description}
        for i, (code, (name, description)) in enumerate(config.SUB_METERS.items(), start=1)
    ])
    sub_frames = [
        keys.assign(meter_id=row.meter_id, energy_wh=hourly[row.code].to_numpy())
        for row in meters.itertuples()
    ]
    sub_readings = pd.concat(sub_frames, ignore_index=True)

    with closing(connect(db_path)) as connection, connection:
        connection.executescript(SCHEMA_SQL)
        calendar.to_sql("calendar", connection, if_exists="append", index=False)
        readings.to_sql("hourly_readings", connection, if_exists="append", index=False)
        meters.to_sql("sub_meters", connection, if_exists="append", index=False)
        sub_readings.to_sql("sub_meter_readings", connection, if_exists="append",
                            index=False, chunksize=50_000)
    return {
        "calendar": len(calendar),
        "hourly_readings": len(readings),
        "sub_meters": len(meters),
        "sub_meter_readings": len(sub_readings),
    }


QUERIES = {
    "1_monthly_energy_join": (
        "Monthly energy (kWh) per year — JOIN + GROUP BY",
        """
        SELECT c.year, c.month, ROUND(SUM(h.energy_kwh), 1) AS total_kwh
        FROM hourly_readings h
        JOIN calendar c ON c.date_id = h.date_id
        GROUP BY c.year, c.month
        ORDER BY c.year, c.month
        """,
    ),
    "2_high_months_having": (
        "Months whose average daily consumption exceeds the overall average — HAVING + subquery",
        """
        SELECT c.year, c.month, ROUND(SUM(h.energy_kwh) / COUNT(DISTINCT c.date_id), 2) AS avg_daily_kwh
        FROM hourly_readings h
        JOIN calendar c ON c.date_id = h.date_id
        GROUP BY c.year, c.month
        HAVING SUM(h.energy_kwh) / COUNT(DISTINCT c.date_id) >
               (SELECT SUM(energy_kwh) / COUNT(DISTINCT date_id) FROM hourly_readings)
        ORDER BY avg_daily_kwh DESC
        """,
    ),
    "3_weekday_weekend_by_season": (
        "Average hourly load: weekday vs weekend by season",
        """
        SELECT c.season,
               CASE c.is_weekend WHEN 1 THEN 'weekend' ELSE 'weekday' END AS day_type,
               ROUND(AVG(h.energy_kwh), 3) AS avg_hourly_kwh,
               ROUND(MAX(h.energy_kwh), 3) AS max_hourly_kwh
        FROM hourly_readings h
        JOIN calendar c ON c.date_id = h.date_id
        GROUP BY c.season, c.is_weekend
        ORDER BY c.season, day_type
        """,
    ),
    "4_sub_meter_share_by_season": (
        "Share of each sub-meter by season — 3-table JOIN + window SUM",
        """
        SELECT c.season, m.name,
               ROUND(SUM(r.energy_wh) / 1000, 1) AS kwh,
               ROUND(100.0 * SUM(r.energy_wh) / SUM(SUM(r.energy_wh)) OVER (PARTITION BY c.season), 1) AS share_pct
        FROM sub_meter_readings r
        JOIN sub_meters m ON m.meter_id = r.meter_id
        JOIN calendar c ON c.date_id = r.date_id
        GROUP BY c.season, m.name
        ORDER BY c.season, share_pct DESC
        """,
    ),
    "5_top_peak_days_cte": (
        "Top-10 days by consumption with their peak hour — CTE",
        """
        WITH daily AS (
            SELECT date_id, SUM(energy_kwh) AS day_kwh, MAX(energy_kwh) AS peak_kwh
            FROM hourly_readings
            GROUP BY date_id
        ),
        peak_hour AS (
            SELECT h.date_id, MIN(h.hour) AS hour
            FROM hourly_readings h
            JOIN daily d ON d.date_id = h.date_id AND h.energy_kwh = d.peak_kwh
            GROUP BY h.date_id
        )
        SELECT c.date, c.season, ROUND(d.day_kwh, 2) AS day_kwh,
               ROUND(d.peak_kwh, 3) AS peak_kwh, p.hour AS peak_hour
        FROM daily d
        JOIN calendar c ON c.date_id = d.date_id
        JOIN peak_hour p ON p.date_id = d.date_id
        ORDER BY d.day_kwh DESC
        LIMIT 10
        """,
    ),
    "6_rolling_7d_window": (
        "Daily energy with 7-day moving average — window AVG OVER ROWS",
        """
        WITH daily AS (
            SELECT c.date, SUM(h.energy_kwh) AS day_kwh
            FROM hourly_readings h
            JOIN calendar c ON c.date_id = h.date_id
            GROUP BY c.date
        )
        SELECT date, ROUND(day_kwh, 2) AS day_kwh,
               ROUND(AVG(day_kwh) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS ma_7d
        FROM daily
        ORDER BY date
        """,
    ),
    "7_month_rank_in_year": (
        "Rank of months within each year by consumption — RANK() OVER PARTITION",
        """
        WITH monthly AS (
            SELECT c.year, c.month, SUM(h.energy_kwh) AS month_kwh
            FROM hourly_readings h
            JOIN calendar c ON c.date_id = h.date_id
            GROUP BY c.year, c.month
            HAVING COUNT(DISTINCT c.date_id) >= 28  -- skip incomplete first/last month
        )
        SELECT year, month, ROUND(month_kwh, 1) AS month_kwh,
               RANK() OVER (PARTITION BY year ORDER BY month_kwh DESC) AS rank_in_year
        FROM monthly
        ORDER BY year, rank_in_year
        """,
    ),
    "8_year_over_year_lag": (
        "Year-over-year change for each month — LAG()",
        """
        WITH monthly AS (
            SELECT c.year, c.month, SUM(h.energy_kwh) AS month_kwh
            FROM hourly_readings h
            JOIN calendar c ON c.date_id = h.date_id
            GROUP BY c.year, c.month
            HAVING COUNT(DISTINCT c.date_id) >= 28  -- skip incomplete first/last month
        )
        SELECT year, month, ROUND(month_kwh, 1) AS month_kwh,
               ROUND(LAG(month_kwh) OVER (PARTITION BY month ORDER BY year), 1) AS prev_year_kwh,
               ROUND(100.0 * (month_kwh - LAG(month_kwh) OVER (PARTITION BY month ORDER BY year))
                     / LAG(month_kwh) OVER (PARTITION BY month ORDER BY year), 1) AS yoy_change_pct
        FROM monthly
        ORDER BY month, year
        """,
    ),
    "9_hours_above_hour_average": (
        "Share of hours above the average for that hour of day — correlated subquery",
        """
        SELECT h.hour,
               COUNT(*) AS hours_above_avg,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM hourly_readings x WHERE x.hour = h.hour), 1) AS pct
        FROM hourly_readings h
        WHERE h.energy_kwh > (SELECT AVG(y.energy_kwh) FROM hourly_readings y WHERE y.hour = h.hour)
        GROUP BY h.hour
        ORDER BY h.hour
        """,
    ),
    "10_hourly_profile_ntile": (
        "Hour-of-day profile split into load quartiles — NTILE()",
        """
        WITH profile AS (
            SELECT hour, AVG(energy_kwh) AS avg_kwh
            FROM hourly_readings
            GROUP BY hour
        )
        SELECT hour, ROUND(avg_kwh, 3) AS avg_kwh,
               NTILE(4) OVER (ORDER BY avg_kwh) AS load_quartile
        FROM profile
        ORDER BY hour
        """,
    ),
}


def run_query(name: str, db_path: Path = config.DB_FILE, params: tuple = ()) -> pd.DataFrame:
    if name not in QUERIES:
        raise KeyError(f"Unknown query '{name}'. Available: {', '.join(QUERIES)}")
    _, sql = QUERIES[name]
    with closing(connect(db_path)) as connection:
        return pd.read_sql_query(sql, connection, params=params)


def query_period(start_date: str, end_date: str, db_path: Path = config.DB_FILE) -> pd.DataFrame:
    """Hourly readings in a date range (parameterised query, used by the dashboard)."""
    sql = """
        SELECT c.date, h.hour, c.season, c.is_weekend, h.energy_kwh, h.voltage, h.intensity
        FROM hourly_readings h
        JOIN calendar c ON c.date_id = h.date_id
        WHERE c.date BETWEEN ? AND ?
        ORDER BY c.date, h.hour
    """
    with closing(connect(db_path)) as connection:
        frame = pd.read_sql_query(sql, connection, params=(start_date, end_date))
    frame["datetime"] = pd.to_datetime(frame["date"]) + pd.to_timedelta(frame["hour"], unit="h")
    return frame


def save_forecast_results(metrics: pd.DataFrame, db_path: Path = config.DB_FILE) -> None:
    with closing(connect(db_path)) as connection, connection:
        metrics.to_sql("forecast_results", connection, if_exists="replace", index=False)


def table_counts(db_path: Path = config.DB_FILE) -> dict:
    with closing(connect(db_path)) as connection:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        return {table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                for table in tables}
