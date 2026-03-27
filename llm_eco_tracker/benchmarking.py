from __future__ import annotations

import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .models import ForecastInterval, SchedulePlan
from .models import ForecastSnapshot
from .planning import build_schedule_plan
from .providers.csv_forecast import CsvForecastProvider


@dataclass(frozen=True, slots=True)
class TraceInterval:
    starts_at: datetime
    ends_at: datetime
    forecast_intensity_gco2eq_per_kwh: float
    actual_intensity_gco2eq_per_kwh: float


@dataclass(frozen=True, slots=True)
class TraceDay:
    day: date
    submission_offsets: tuple[int, ...]

    @property
    def submission_count(self) -> int:
        return len(self.submission_offsets)


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


def load_trace_intervals(csv_path: str | Path) -> tuple[TraceInterval, ...]:
    intervals: list[TraceInterval] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            intervals.append(
                TraceInterval(
                    starts_at=_parse_utc_timestamp(row["from"]),
                    ends_at=_parse_utc_timestamp(row["to"]),
                    forecast_intensity_gco2eq_per_kwh=float(row["intensity_forecast"]),
                    actual_intensity_gco2eq_per_kwh=float(row["intensity_actual"]),
                )
            )
    return tuple(intervals)


def group_complete_trace_days(
    trace_intervals: tuple[TraceInterval, ...] | list[TraceInterval],
    *,
    slots_per_day: int = 48,
    max_delay_hours: float = 0.0,
) -> tuple[TraceDay, ...]:
    if slots_per_day <= 0:
        raise ValueError("slots_per_day must be positive.")

    grouped_offsets: OrderedDict[date, list[int]] = OrderedDict()
    for offset, interval in enumerate(trace_intervals):
        grouped_offsets.setdefault(interval.starts_at.date(), []).append(offset)

    eligible_days: list[TraceDay] = []
    for day_key, submission_offsets in grouped_offsets.items():
        if len(submission_offsets) != slots_per_day:
            continue
        if not _is_complete_utc_day(trace_intervals, submission_offsets):
            continue
        if not _has_full_delay_horizon(
            trace_intervals,
            submission_offsets[-1],
            max_delay_hours=max_delay_hours,
        ):
            continue

        eligible_days.append(TraceDay(day=day_key, submission_offsets=tuple(submission_offsets)))

    return tuple(eligible_days)


def build_trace_schedule_plan(
    trace_intervals: tuple[TraceInterval, ...] | list[TraceInterval],
    start_offset: int,
    max_delay_hours: float,
    *,
    intensity_kind: str = "forecast",
) -> tuple[SchedulePlan, int]:
    if start_offset < 0 or start_offset >= len(trace_intervals):
        raise IndexError(f"start_offset {start_offset} is outside the trace interval range.")
    if intensity_kind not in {"forecast", "actual"}:
        raise ValueError("intensity_kind must be either 'forecast' or 'actual'.")

    forecast_intervals = tuple(
        ForecastInterval(
            starts_at=interval.starts_at,
            ends_at=interval.ends_at,
            carbon_intensity_gco2eq_per_kwh=(
                interval.forecast_intensity_gco2eq_per_kwh
                if intensity_kind == "forecast"
                else interval.actual_intensity_gco2eq_per_kwh
            ),
        )
        for interval in trace_intervals[start_offset:]
    )
    reference_time = trace_intervals[start_offset].starts_at
    schedule_plan = build_schedule_plan(
        forecast_intervals,
        max_delay_hours,
        reference_time=reference_time,
    )

    selected_interval = schedule_plan.selected_interval
    if selected_interval is None:
        return schedule_plan, start_offset

    selected_offset = find_trace_offset(
        trace_intervals,
        selected_interval.starts_at,
        start_offset=start_offset,
    )
    return schedule_plan, selected_offset


def find_trace_offset(
    trace_intervals: tuple[TraceInterval, ...] | list[TraceInterval],
    timestamp: datetime,
    *,
    start_offset: int = 0,
) -> int:
    for offset in range(start_offset, len(trace_intervals)):
        if trace_intervals[offset].starts_at == timestamp:
            return offset
    raise ValueError(f"Could not find a trace interval starting at {timestamp.isoformat()}.")


def summarize_trace_days(
    trace_days: tuple[TraceDay, ...] | list[TraceDay],
    *,
    start_day: date | None = None,
    end_day: date | None = None,
    limit_days: int | None = None,
) -> tuple[TraceDay, ...]:
    filtered_days = [
        trace_day
        for trace_day in trace_days
        if (start_day is None or trace_day.day >= start_day)
        and (end_day is None or trace_day.day <= end_day)
    ]
    if limit_days is not None:
        if limit_days < 0:
            raise ValueError("limit_days must be non-negative when provided.")
        filtered_days = filtered_days[:limit_days]
    return tuple(filtered_days)


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_iso_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        return _parse_utc_timestamp(value)
    return datetime.fromisoformat(value)


def format_iso_date(value: date) -> str:
    return value.isoformat()


def last_completed_utc_day(reference_time: datetime | None = None) -> date:
    now = reference_time or datetime.now(timezone.utc)
    return (now - timedelta(days=1)).date()


def contiguous_date_window(end_day: date, *, days: int) -> tuple[date, date]:
    if days <= 0:
        raise ValueError("days must be positive.")
    start_day = end_day - timedelta(days=days - 1)
    return start_day, end_day


def _parse_utc_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)


def _is_complete_utc_day(
    trace_intervals: tuple[TraceInterval, ...] | list[TraceInterval],
    submission_offsets: list[int],
) -> bool:
    first_interval = trace_intervals[submission_offsets[0]]
    last_interval = trace_intervals[submission_offsets[-1]]
    if first_interval.starts_at.hour != 0 or first_interval.starts_at.minute != 0:
        return False
    if last_interval.starts_at.hour != 23 or last_interval.starts_at.minute != 30:
        return False

    for previous_offset, next_offset in zip(submission_offsets, submission_offsets[1:]):
        if trace_intervals[next_offset].starts_at != trace_intervals[previous_offset].ends_at:
            return False

    return True


def _has_full_delay_horizon(
    trace_intervals: tuple[TraceInterval, ...] | list[TraceInterval],
    submission_offset: int,
    *,
    max_delay_hours: float,
) -> bool:
    if max_delay_hours <= 0:
        return True
    deadline = trace_intervals[submission_offset].starts_at + timedelta(hours=max_delay_hours)
    return trace_intervals[-1].starts_at >= deadline
