from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta

from .models import ForecastInterval, SchedulePlan

_DEFAULT_MAX_JITTER_SECONDS = 300.0


def immediate_schedule_plan() -> SchedulePlan:
    return SchedulePlan(
        baseline_interval=None,
        selected_interval=None,
        baseline_intensity_gco2eq_per_kwh=0.0,
        optimal_intensity_gco2eq_per_kwh=0.0,
        raw_delay_seconds=0.0,
        execution_delay_seconds=0.0,
        latest_execution_delay_seconds=0.0,
    )


def build_schedule_plan(
    forecast_intervals: Sequence[ForecastInterval],
    max_delay_hours: float,
    *,
    reference_time: datetime | None = None,
) -> SchedulePlan:
    if max_delay_hours <= 0 or not forecast_intervals:
        return immediate_schedule_plan()

    baseline_interval = forecast_intervals[0]
    search_reference = reference_time or baseline_interval.starts_at
    search_deadline = search_reference + timedelta(hours=max_delay_hours)

    candidate_intervals = [
        interval for interval in forecast_intervals if interval.starts_at <= search_deadline
    ] or [baseline_interval]

    selected_interval = min(
        candidate_intervals,
        key=lambda interval: interval.carbon_intensity_gco2eq_per_kwh,
    )
    raw_delay_seconds = max(
        0.0,
        (selected_interval.starts_at - search_reference).total_seconds(),
    )
    latest_execution_time = min(selected_interval.ends_at, search_deadline)
    latest_execution_delay_seconds = max(
        raw_delay_seconds,
        (latest_execution_time - search_reference).total_seconds(),
    )

    return SchedulePlan(
        baseline_interval=baseline_interval,
        selected_interval=selected_interval,
        baseline_intensity_gco2eq_per_kwh=baseline_interval.carbon_intensity_gco2eq_per_kwh,
        optimal_intensity_gco2eq_per_kwh=selected_interval.carbon_intensity_gco2eq_per_kwh,
        raw_delay_seconds=raw_delay_seconds,
        execution_delay_seconds=raw_delay_seconds,
        latest_execution_delay_seconds=latest_execution_delay_seconds,
    )


def apply_jitter(
    delay_seconds: float,
    *,
    latest_delay_seconds: float | None = None,
    max_jitter_seconds: float = _DEFAULT_MAX_JITTER_SECONDS,
    random_uniform: Callable[[float, float], float] | None = None,
) -> float:
    """
    Randomly spreads delayed executions after the selected greener window
    starts, without exceeding the selected forecast interval or the user's
    maximum delay budget.
    """
    if delay_seconds <= 0:
        return 0.0

    resolved_latest_delay_seconds = max(delay_seconds, float(latest_delay_seconds or delay_seconds))
    jitter_window = min(
        max(0.0, float(max_jitter_seconds)),
        resolved_latest_delay_seconds - delay_seconds,
    )
    if jitter_window <= 0:
        return delay_seconds

    uniform = random_uniform or random.uniform
    jitter = uniform(0.0, jitter_window)
    return delay_seconds + jitter


def apply_jitter_to_plan(
    schedule_plan: SchedulePlan,
    *,
    max_jitter_seconds: float = _DEFAULT_MAX_JITTER_SECONDS,
    random_uniform: Callable[[float, float], float] | None = None,
) -> SchedulePlan:
    jittered_delay = apply_jitter(
        schedule_plan.raw_delay_seconds,
        latest_delay_seconds=schedule_plan.latest_execution_delay_seconds,
        max_jitter_seconds=max_jitter_seconds,
        random_uniform=random_uniform,
    )
    return replace(schedule_plan, execution_delay_seconds=jittered_delay)


def cap_execution_delay(schedule_plan: SchedulePlan, max_delay_seconds: float | None) -> SchedulePlan:
    if max_delay_seconds is None:
        return schedule_plan

    capped_delay_seconds = max(0.0, max_delay_seconds)
    if schedule_plan.execution_delay_seconds <= capped_delay_seconds:
        return schedule_plan

    return replace(schedule_plan, execution_delay_seconds=capped_delay_seconds)
