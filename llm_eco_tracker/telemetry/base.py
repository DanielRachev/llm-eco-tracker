from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ..models import TelemetryRecord


class TelemetryAdapter(Protocol):
    provider_name: str

    def install(self, record_energy_kwh: Callable[[float], None]) -> bool:
        """Install provider-specific telemetry hooks for the current process."""

    def uninstall(self) -> None:
        """Remove any previously installed telemetry hooks."""


class TelemetrySink(Protocol):
    def emit(self, record: TelemetryRecord) -> None:
        """Persist or forward a normalized telemetry record."""
