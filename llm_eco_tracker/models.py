from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ForecastInterval:
    """A single forecast window with its associated carbon intensity."""

    starts_at: datetime
    ends_at: datetime
    carbon_intensity_gco2eq_per_kwh: float


@dataclass(frozen=True, slots=True)
class SchedulePlan:
    """The selected execution plan for a carbon-aware invocation."""

    baseline_interval: ForecastInterval | None
    selected_interval: ForecastInterval | None
    baseline_intensity_gco2eq_per_kwh: float
    optimal_intensity_gco2eq_per_kwh: float
    raw_delay_seconds: float
    execution_delay_seconds: float


@dataclass(frozen=True, slots=True)
class EmissionSummary:
    """Normalized emissions computed for one execution."""

    energy_kwh: float
    baseline_gco2eq: float
    actual_gco2eq: float
    saved_gco2eq: float


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """A telemetry event ready to be written by a sink."""

    timestamp: datetime
    emissions: EmissionSummary
    schedule_plan: SchedulePlan | None = None
    forecast_provider: str | None = None
    llm_provider: str | None = None
    model: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
