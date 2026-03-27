from __future__ import annotations

from typing import Protocol

from ..models import ModelDowngradePolicy, TelemetryRecord


class TelemetrySessionHooks(Protocol):
    def get_model_downgrade_policy(self) -> ModelDowngradePolicy:
        """Return the active per-session model downgrade policy."""

    def record_energy_kwh(self, energy_kwh: float) -> None:
        """Add energy captured for the current session."""

    def record_llm_provider(self, provider_name: str) -> None:
        """Record one LLM provider observed during the current session."""

    def record_model_usage(self, requested_model: str | None, effective_model: str | None) -> None:
        """Record one model invocation for the current session."""

    def should_warn_unmapped_model(self, requested_model: str) -> bool:
        """Return whether an unmapped-model warning should be emitted now."""


class TelemetryAdapter(Protocol):
    provider_name: str

    def is_available(self) -> bool:
        """Return whether the provider SDK needed by this adapter is installed."""

    def install(self, session_hooks: TelemetrySessionHooks) -> bool:
        """Install provider-specific telemetry hooks for the current process."""

    def uninstall(self) -> None:
        """Remove any previously installed telemetry hooks."""


class TelemetrySink(Protocol):
    def emit(self, record: TelemetryRecord) -> None:
        """Persist or forward a normalized telemetry record."""
