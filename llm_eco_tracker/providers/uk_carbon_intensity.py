from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

from ..models import ForecastInterval, ForecastSnapshot

logger = logging.getLogger(__name__)


class UKCarbonIntensityProvider:
    provider_name = "uk_carbon_intensity"

    def load_forecast(self, max_delay_hours: float) -> ForecastSnapshot:
        if max_delay_hours <= 0:
            return ForecastSnapshot(intervals=())

        now = datetime.now(timezone.utc)
        end_time = now + timedelta(hours=max_delay_hours)
        start_str = now.strftime("%Y-%m-%dT%H:%MZ")
        end_str = end_time.strftime("%Y-%m-%dT%H:%MZ")
        url = f"https://api.carbonintensity.org.uk/intensity/{start_str}/{end_str}"

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json().get("data", [])
        except Exception as exc:
            logger.error("Failed to fetch grid data: %s. Executing immediately.", exc)
            return ForecastSnapshot(intervals=(), reference_time=now)

        if not data:
            logger.warning("No forecast data returned. Defaulting to immediate execution.")
            return ForecastSnapshot(intervals=(), reference_time=now)

        intervals = []
        for entry in data:
            try:
                intervals.append(
                    ForecastInterval(
                        starts_at=_parse_utc_timestamp(entry["from"]),
                        ends_at=_parse_utc_timestamp(entry["to"]),
                        carbon_intensity_gco2eq_per_kwh=float(entry["intensity"]["forecast"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed live forecast row: %s", exc)

        return ForecastSnapshot(intervals=tuple(intervals), reference_time=now)


def _parse_utc_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
