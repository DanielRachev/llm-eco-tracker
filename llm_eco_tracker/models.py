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
class ForecastSnapshot:
    """A provider response containing forecast intervals and their reference time."""

    intervals: tuple[ForecastInterval, ...]
    reference_time: datetime | None = None


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
class ModelDowngradePolicy:
    """Execution-time downgrade policy for one carbon-aware session."""

    enabled: bool
    dirty_threshold_gco2eq_per_kwh: float
    execution_intensity_gco2eq_per_kwh: float
    fallback_map: dict[str, str] = field(default_factory=dict)

    @property
    def is_dirty(self) -> bool:
        return (
            self.enabled
            and self.execution_intensity_gco2eq_per_kwh >= self.dirty_threshold_gco2eq_per_kwh
        )


@dataclass(frozen=True, slots=True)
class ModelUsageSummary:
    """Aggregate usage for one requested/effective model pair."""

    requested_model: str
    effective_model: str
    call_count: int
    downgraded: bool


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """A telemetry event ready to be written by a sink."""

    timestamp: datetime
    emissions: EmissionSummary
    schedule_plan: SchedulePlan | None = None
    forecast_provider: str | None = None
    llm_provider: str | None = None
    model: str | None = None
    model_usage: tuple[ModelUsageSummary, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
