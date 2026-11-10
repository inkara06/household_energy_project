"""Loading the raw UCI file: download, chunked reading, dtype optimisation, Parquet."""
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src import config


def download_dataset(target_dir: Path = config.RAW_DIR) -> Path:
    """Download and unzip the dataset unless it is already present."""
    if config.RAW_FILE.exists():
        return config.RAW_FILE
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "hpc.zip"
    try:
        urllib.request.urlretrieve(config.DATASET_URL, zip_path)
    except OSError as error:
        raise RuntimeError(
            f"Could not download dataset. Download it manually from {config.DATASET_URL}"
        ) from error
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)
    return config.RAW_FILE


def memory_mb(frame: pd.DataFrame) -> float:
    """Deep memory footprint of a DataFrame in MB."""
    return frame.memory_usage(deep=True).sum() / config.BYTES_PER_MB


def read_raw_naive(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the whole file with default pandas settings (baseline for memory)."""
    return pd.read_csv(path, sep=config.CSV_SEPARATOR, low_memory=False)


def _optimize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["datetime"] = pd.to_datetime(
        chunk["Date"] + " " + chunk["Time"], format=config.DATETIME_FORMAT
    )
    chunk = chunk.drop(columns=["Date", "Time"])
    for column in config.NUMERIC_COLUMNS:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("float32")
    return chunk


def read_raw_chunked(path: Path = config.RAW_FILE,
                     chunk_size: int = config.CHUNK_SIZE_ROWS) -> pd.DataFrame:
    """Read the file chunk by chunk, parse datetime and downcast numbers to float32."""
    chunks = []
    reader = pd.read_csv(
        path,
        sep=config.CSV_SEPARATOR,
        na_values=[config.MISSING_MARKER],
        dtype={"Date": "string", "Time": "string"},
        chunksize=chunk_size,
    )
    for chunk in reader:
        chunks.append(_optimize_chunk(chunk))
    frame = pd.concat(chunks, ignore_index=True)
    return frame.set_index("datetime").sort_index()


def save_parquet(frame: pd.DataFrame, path: Path = config.PARQUET_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="snappy")
    return path


def load_parquet(path: Path = config.PARQUET_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the loading step first.")
    return pd.read_parquet(path, engine="pyarrow")


def load_or_build_minute_data() -> pd.DataFrame:
    """Return the optimised minute-level frame, building the Parquet cache if needed."""
    try:
        return load_parquet()
    except FileNotFoundError:
        download_dataset()
        frame = read_raw_chunked()
        save_parquet(frame)
        return frame


BENCHMARK_SCRIPT = """
import json, resource, sys, time
from src import data_loader
start = time.perf_counter()
frame = getattr(data_loader, sys.argv[1])()
elapsed_s = time.perf_counter() - start
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = peak / 1024 ** 2 if sys.platform == "darwin" else peak / 1024
print(json.dumps({"time_s": elapsed_s, "peak_rss_mb": peak_mb,
                  "frame_mb": data_loader.memory_mb(frame), "rows": len(frame)}))
"""

BENCHMARK_APPROACHES = {
    "CSV naive (baseline)": ("read_raw_naive", config.RAW_FILE),
    "CSV chunked + float32": ("read_raw_chunked", config.RAW_FILE),
    "Parquet (pyarrow)": ("load_parquet", config.PARQUET_FILE),
}


def _measure_in_subprocess(function_name: str) -> dict:
    """Run one reader in a fresh Python process so its peak RSS is measured in isolation."""
    completed = subprocess.run(
        [sys.executable, "-c", BENCHMARK_SCRIPT, function_name],
        cwd=config.PROJECT_DIR, capture_output=True, text=True, check=True, timeout=600,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def benchmark_formats() -> list[dict]:
    """Compare reading approaches: naive CSV vs chunked+optimised CSV vs Parquet.

    Criteria: read time, peak process memory (RSS), DataFrame memory, file size.
    """
    rows = []
    for name, (function_name, path) in BENCHMARK_APPROACHES.items():
        measured = _measure_in_subprocess(function_name)
        rows.append({
            "approach": name,
            "read_time_s": round(measured["time_s"], 2),
            "peak_rss_mb": round(measured["peak_rss_mb"], 1),
            "frame_memory_mb": round(measured["frame_mb"], 1),
            "file_size_mb": round(path.stat().st_size / config.BYTES_PER_MB, 1),
            "rows": measured["rows"],
        })
    config.BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.BENCHMARK_FILE.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
