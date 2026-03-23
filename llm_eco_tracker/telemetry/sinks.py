from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..models import ForecastInterval, SchedulePlan, TelemetryRecord
from .base import TelemetrySink

logger = logging.getLogger(__name__)


class NoOpTelemetrySink:
    def emit(self, record: TelemetryRecord) -> None:
        del record


class JsonlTelemetrySink:
    def __init__(self, path: str | Path):
        self._path = Path(path)

    def emit(self, record: TelemetryRecord) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_serialize_record(record)) + "\n")
        except OSError as exc:
            logger.warning("Failed to write telemetry file '%s': %s", self._path, exc)


class LoggerTelemetrySink:
    def __init__(self, logger_instance: logging.Logger | None = None, *, level: int = logging.INFO):
        self._logger = logger_instance or logger
        self._level = level

    def emit(self, record: TelemetryRecord) -> None:
        self._logger.log(self._level, "Telemetry record: %s", json.dumps(_serialize_record(record)))


class CompositeTelemetrySink:
    def __init__(self, sinks: list[TelemetrySink] | tuple[TelemetrySink, ...]):
        self._sinks = tuple(sinks)

    def emit(self, record: TelemetryRecord) -> None:
        for sink in self._sinks:
            try:
                sink.emit(record)
            except Exception as exc:
                logger.warning("Telemetry sink '%s' failed: %s", type(sink).__name__, exc)


def _serialize_record(record: TelemetryRecord) -> dict[str, Any]:
    payload = {
        "timestamp": record.timestamp.isoformat(),
        "energy_kwh": record.emissions.energy_kwh,
        "baseline_gco2eq": record.emissions.baseline_gco2eq,
        "actual_gco2eq": record.emissions.actual_gco2eq,
        "saved_gco2eq": record.emissions.saved_gco2eq,
    }

    if record.schedule_plan is not None:
        payload["schedule_plan"] = _serialize_schedule_plan(record.schedule_plan)
    if record.forecast_provider is not None:
        payload["forecast_provider"] = record.forecast_provider
    if record.llm_provider is not None:
        payload["llm_provider"] = record.llm_provider
    if record.model is not None:
        payload["model"] = record.model
    if record.metadata:
        payload["metadata"] = dict(record.metadata)

    return payload


def _serialize_schedule_plan(schedule_plan: SchedulePlan) -> dict[str, Any]:
    return {
        "baseline_interval": _serialize_forecast_interval(schedule_plan.baseline_interval),
        "selected_interval": _serialize_forecast_interval(schedule_plan.selected_interval),
        "baseline_intensity_gco2eq_per_kwh": schedule_plan.baseline_intensity_gco2eq_per_kwh,
        "optimal_intensity_gco2eq_per_kwh": schedule_plan.optimal_intensity_gco2eq_per_kwh,
        "raw_delay_seconds": schedule_plan.raw_delay_seconds,
        "execution_delay_seconds": schedule_plan.execution_delay_seconds,
    }


def _serialize_forecast_interval(interval: ForecastInterval | None) -> dict[str, Any] | None:
    if interval is None:
        return None

    return {
        "starts_at": interval.starts_at.isoformat(),
        "ends_at": interval.ends_at.isoformat(),
        "carbon_intensity_gco2eq_per_kwh": interval.carbon_intensity_gco2eq_per_kwh,
    }
