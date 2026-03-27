from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import ForecastSnapshot
from .providers.csv_forecast import CsvForecastProvider


class SlidingCsvForecastProvider:
    provider_name = "csv_forecast"

    def __init__(self, csv_path: str | Path):
        self._delegate = CsvForecastProvider(csv_path)
        snapshot = self._delegate.load_forecast(max_delay_hours=0)
        if not snapshot.intervals:
            raise ValueError(f"No forecast intervals found in '{csv_path}'.")

        self._intervals = snapshot.intervals
        self._start_offset = 0

    @property
    def interval_count(self) -> int:
        return len(self._intervals)

    @property
    def current_interval(self):
        return self._intervals[self._start_offset]

    def set_start_offset(self, start_offset: int) -> None:
        if start_offset < 0 or start_offset >= len(self._intervals):
            raise IndexError(
                f"Start offset {start_offset} is outside the available forecast range "
                f"(0-{len(self._intervals) - 1})."
            )
        self._start_offset = start_offset

    def load_forecast(self, max_delay_hours: float) -> ForecastSnapshot:
        del max_delay_hours

        intervals = self._intervals[self._start_offset :]
        if not intervals:
            return ForecastSnapshot(intervals=(), reference_time=None)

        return ForecastSnapshot(intervals=intervals, reference_time=intervals[0].starts_at)


def iter_submission_offsets(
    total_intervals: int,
    *,
    start_offset: int = 0,
    limit: int | None = None,
    step: int = 1,
) -> list[int]:
    if total_intervals <= 0:
        return []
    if start_offset < 0 or start_offset >= total_intervals:
        raise ValueError(
            f"start_offset must be between 0 and {total_intervals - 1}, got {start_offset}."
        )
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}.")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative when provided, got {limit}.")

    offsets = list(range(start_offset, total_intervals, step))
    if limit is None:
        return offsets
    return offsets[:limit]


def load_actual_intensity_mapping(csv_path: str | Path) -> dict[str, float]:
    mapping: dict[str, float] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["from"]] = float(row["intensity_actual"])
    return mapping


def format_forecast_timestamp(timestamp: datetime) -> str:
    aligned_minute = 30 if timestamp.minute >= 30 else 0
    aligned_timestamp = timestamp.replace(minute=aligned_minute, second=0, microsecond=0)
    return aligned_timestamp.strftime("%Y-%m-%dT%H:%MZ")


def lookup_actual_intensity(
    actual_mapping: dict[str, float],
    timestamp: datetime,
    *,
    fallback: float,
) -> float:
    return actual_mapping.get(format_forecast_timestamp(timestamp), fallback)


def reset_output_file(path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_path.exists():
        resolved_path.unlink()
    return resolved_path


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = Path(path)
    if not resolved_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def read_last_new_jsonl_record(
    path: str | Path,
    previous_count: int,
) -> tuple[dict[str, Any], int]:
    records = load_jsonl_records(path)
    if len(records) <= previous_count:
        raise RuntimeError(
            f"Expected a new telemetry record in '{path}', but the file did not grow."
        )
    return records[-1], len(records)


async def mock_sleep(seconds: float) -> None:
    del seconds
