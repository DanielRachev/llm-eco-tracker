from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..models import ForecastInterval, ForecastSnapshot

logger = logging.getLogger(__name__)


class CsvForecastProvider:
    provider_name = "csv_forecast"

    def __init__(self, csv_path: str | Path):
        self._csv_path = Path(csv_path)

    def load_forecast(self, max_delay_hours: float) -> ForecastSnapshot:
        del max_delay_hours

        try:
            with self._csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                intervals = tuple(self._parse_row(row) for row in reader)
        except (OSError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to read CSV forecast data from '%s': %s",
                self._csv_path,
                exc,
            )
            return ForecastSnapshot(intervals=())

        if not intervals:
            return ForecastSnapshot(intervals=())

        return ForecastSnapshot(
            intervals=intervals,
            reference_time=intervals[0].starts_at,
        )

    @staticmethod
    def _parse_row(row: dict[str, str]) -> ForecastInterval:
        return ForecastInterval(
            starts_at=_parse_utc_timestamp(row["from"]),
            ends_at=_parse_utc_timestamp(row["to"]),
            carbon_intensity_gco2eq_per_kwh=float(row["intensity_forecast"]),
        )


def _parse_utc_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
