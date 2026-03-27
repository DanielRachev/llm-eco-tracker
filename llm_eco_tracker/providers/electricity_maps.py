from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from ..models import ForecastInterval, ForecastSnapshot

logger = logging.getLogger(__name__)

_API_URL = "https://api.electricitymap.org/v3/carbon-intensity/forecast"
_SUPPORTED_HORIZON_HOURS = (24, 48, 72)
_GRANULARITY_DELTAS = {
    "5_minutes": timedelta(minutes=5),
    "15_minutes": timedelta(minutes=15),
    "hourly": timedelta(hours=1),
}


class ElectricityMapsProvider:
    provider_name = "electricity_maps"

    def __init__(
        self,
        *,
        zone: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
        data_center_provider: str | None = None,
        data_center_region: str | None = None,
        auth_token: str | None = None,
        emission_factor_type: str = "lifecycle",
        temporal_granularity: str = "hourly",
        disable_estimations: bool = False,
        disable_caller_lookup: bool = True,
        request_timeout_seconds: float = 5.0,
        api_url: str = _API_URL,
    ):
        self._zone = zone
        self._lat = lat
        self._lon = lon
        self._data_center_provider = data_center_provider
        self._data_center_region = data_center_region
        self._auth_token = auth_token or os.getenv("ELECTRICITY_MAPS_API_TOKEN") or os.getenv(
            "ELECTRICITY_MAPS_AUTH_TOKEN"
        )
        self._emission_factor_type = emission_factor_type
        self._temporal_granularity = temporal_granularity
        self._disable_estimations = disable_estimations
        self._disable_caller_lookup = disable_caller_lookup
        self._request_timeout_seconds = request_timeout_seconds
        self._api_url = api_url

    def load_forecast(self, max_delay_hours: float) -> ForecastSnapshot:
        if max_delay_hours <= 0:
            return ForecastSnapshot(intervals=())

        if not self._auth_token:
            logger.error(
                "Electricity Maps auth token is missing. Set ELECTRICITY_MAPS_API_TOKEN or pass auth_token=..."
            )
            return ForecastSnapshot(intervals=())

        if not self._has_location_selector():
            logger.error(
                "Electricity Maps provider requires a zone, lat/lon pair, or data center selector."
            )
            return ForecastSnapshot(intervals=())

        if self._temporal_granularity not in _GRANULARITY_DELTAS:
            logger.error(
                "Unsupported Electricity Maps temporal granularity '%s'. Supported values: %s",
                self._temporal_granularity,
                ", ".join(sorted(_GRANULARITY_DELTAS)),
            )
            return ForecastSnapshot(intervals=())

        params = self._build_query_params(max_delay_hours)
        headers = {"auth-token": self._auth_token}

        try:
            response = requests.get(
                self._api_url,
                params=params,
                headers=headers,
                timeout=self._request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.error("Failed to fetch Electricity Maps forecast data: %s", exc)
            return ForecastSnapshot(intervals=())

        intervals = self._parse_intervals(payload)
        if not intervals:
            logger.warning(
                "No usable Electricity Maps forecast rows were returned. Defaulting to immediate execution."
            )
            return ForecastSnapshot(intervals=())

        return ForecastSnapshot(
            intervals=tuple(intervals),
            reference_time=intervals[0].starts_at,
        )

    def _has_location_selector(self) -> bool:
        if self._zone:
            return True
        if self._lat is not None and self._lon is not None:
            return True
        if self._data_center_provider and self._data_center_region:
            return True
        return False

    def _build_query_params(self, max_delay_hours: float) -> dict[str, Any]:
        params: dict[str, Any] = {
            "temporalGranularity": self._temporal_granularity,
            "emissionFactorType": self._emission_factor_type,
            "disableEstimations": str(self._disable_estimations).lower(),
            "disableCallerLookup": str(self._disable_caller_lookup).lower(),
            "horizonHours": self._resolve_horizon_hours(max_delay_hours),
        }
        if self._zone:
            params["zone"] = self._zone
        if self._lat is not None and self._lon is not None:
            params["lat"] = self._lat
            params["lon"] = self._lon
        if self._data_center_provider and self._data_center_region:
            params["dataCenterProvider"] = self._data_center_provider
            params["dataCenterRegion"] = self._data_center_region
        return params

    @staticmethod
    def _resolve_horizon_hours(max_delay_hours: float) -> int:
        for horizon in _SUPPORTED_HORIZON_HOURS:
            if max_delay_hours <= horizon:
                return horizon
        return _SUPPORTED_HORIZON_HOURS[-1]

    def _parse_intervals(self, payload: Any) -> list[ForecastInterval]:
        entries = self._extract_entries(payload)
        if not entries:
            return []

        parsed_points: list[tuple[datetime, float]] = []
        for entry in entries:
            try:
                starts_at = _parse_timestamp(_extract_timestamp(entry))
                intensity = float(_extract_intensity(entry))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed Electricity Maps forecast row: %s", exc)
                continue
            parsed_points.append((starts_at, intensity))

        parsed_points.sort(key=lambda item: item[0])
        if not parsed_points:
            return []

        default_delta = _GRANULARITY_DELTAS[self._temporal_granularity]
        intervals: list[ForecastInterval] = []
        for index, (starts_at, intensity) in enumerate(parsed_points):
            if index + 1 < len(parsed_points):
                ends_at = parsed_points[index + 1][0]
            else:
                ends_at = starts_at + default_delta
            if ends_at <= starts_at:
                ends_at = starts_at + default_delta

            intervals.append(
                ForecastInterval(
                    starts_at=starts_at,
                    ends_at=ends_at,
                    carbon_intensity_gco2eq_per_kwh=intensity,
                )
            )

        return intervals

    @staticmethod
    def _extract_entries(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        if not isinstance(payload, dict):
            return []

        for key in ("data", "forecast", "history", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]

        if "datetime" in payload and any(
            key in payload for key in ("carbonIntensity", "carbonIntensityForecast", "value")
        ):
            return [payload]

        return []


def _extract_timestamp(entry: dict[str, Any]) -> str:
    for key in ("datetime", "from", "timestamp"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError("timestamp field not found")


def _extract_intensity(entry: dict[str, Any]) -> float:
    for key in ("carbonIntensity", "carbonIntensityForecast", "value"):
        value = entry.get(key)
        resolved = _coerce_numeric(value)
        if resolved is not None:
            return resolved
    raise KeyError("carbon intensity field not found")


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("value", "carbonIntensity"):
            nested = value.get(key)
            if isinstance(nested, (int, float)):
                return float(nested)
    return None


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
