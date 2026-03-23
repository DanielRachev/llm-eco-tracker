import functools
import inspect
import logging
from pathlib import Path

from .execution import ExecutionRunner
from .models import SchedulePlan
from .planning import (
    apply_jitter_to_plan,
    build_schedule_plan,
    cap_execution_delay,
    immediate_schedule_plan,
)
from .providers import CsvForecastProvider, UKCarbonIntensityProvider
from .providers.base import ForecastProvider
from .telemetry import EcoLogitsRuntime, JsonlTelemetrySink
from .telemetry.adapters import OpenAIChatCompletionsAdapter


logger = logging.getLogger(__name__)

_telemetry_path = Path("eco_telemetry.jsonl")
_mock_max_sleep_seconds = 1.0
_telemetry_runtime = EcoLogitsRuntime([OpenAIChatCompletionsAdapter()])
_telemetry_sink = JsonlTelemetrySink(_telemetry_path)
_execution_runner = ExecutionRunner(
    _telemetry_runtime,
    _telemetry_sink,
    llm_provider="openai",
)


def _select_forecast_provider(location: str, mock_csv: str | None) -> ForecastProvider:
    if mock_csv:
        logger.info("Mock mode enabled: reading forecast data from '%s'.", mock_csv)
        return CsvForecastProvider(mock_csv)

    if location.upper() not in {"UK", "GB"}:
        logger.warning(
            "Live scheduling currently uses the UK Carbon Intensity API. "
            "Location '%s' is not applied yet.",
            location,
        )

    logger.info("Live mode enabled: fetching grid forecast.")
    return UKCarbonIntensityProvider()


def _log_schedule_plan(provider: ForecastProvider, schedule_plan: SchedulePlan):
    if schedule_plan.baseline_interval is None or schedule_plan.selected_interval is None:
        return

    prefix = "Mock " if provider.provider_name == "csv_forecast" else ""
    logger.info(
        "%scurrent forecast: %.1f gCO2eq/kWh",
        prefix,
        schedule_plan.baseline_intensity_gco2eq_per_kwh,
    )
    logger.info(
        "%sbest forecast found: %.1f gCO2eq/kWh at %s",
        prefix,
        schedule_plan.optimal_intensity_gco2eq_per_kwh,
        schedule_plan.selected_interval.starts_at.isoformat(),
    )


def _get_schedule_plan(max_delay_hours, location, mock_csv):
    if max_delay_hours <= 0:
        return immediate_schedule_plan()

    forecast_provider = _select_forecast_provider(location, mock_csv)
    forecast_snapshot = forecast_provider.load_forecast(max_delay_hours)
    schedule_plan = build_schedule_plan(
        forecast_snapshot.intervals,
        max_delay_hours,
        reference_time=forecast_snapshot.reference_time,
    )
    _log_schedule_plan(forecast_provider, schedule_plan)
    return schedule_plan


def carbon_aware(max_delay_hours=2, location="NL", mock_csv=None):
    """
    Delay non-urgent work until a greener grid window, then record session-wide
    energy telemetry for OpenAI chat completions executed inside the function.

    Args:
        max_delay_hours (int): Maximum time to wait for a greener grid window.
        location (str): Region hint for future scheduler backends.
        mock_csv (str | None): Optional CSV path for deterministic mock forecasts.
    """

    def _log_intercept(func):
        logger.info("Intercepting call to '%s'", func.__name__)
        logger.info("Target location: %s, max delay: %sh", location, max_delay_hours)
        if mock_csv:
            logger.info("Using mock forecast data from: %s", mock_csv)

    def _build_delay_plan():
        schedule_plan = _get_schedule_plan(max_delay_hours, location, mock_csv)
        jittered_plan = apply_jitter_to_plan(schedule_plan)
        final_plan = jittered_plan

        if mock_csv and final_plan.execution_delay_seconds > _mock_max_sleep_seconds:
            logger.info(
                "Mock mode: capping actual wait from %.2fs to %.2fs.",
                final_plan.execution_delay_seconds,
                _mock_max_sleep_seconds,
            )
            final_plan = cap_execution_delay(final_plan, _mock_max_sleep_seconds)

        logger.info(
            "Scheduling plan: baseline %.1f gCO2eq/kWh, optimal %.1f gCO2eq/kWh, raw delay %.2fs, jittered delay %.2fs, execution delay %.2fs",
            final_plan.baseline_intensity_gco2eq_per_kwh,
            final_plan.optimal_intensity_gco2eq_per_kwh,
            final_plan.raw_delay_seconds,
            jittered_plan.execution_delay_seconds,
            final_plan.execution_delay_seconds,
        )

        return final_plan

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                _log_intercept(func)
                schedule_plan = _build_delay_plan()
                return await _execution_runner.run_async(func, args, kwargs, schedule_plan)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _log_intercept(func)
            schedule_plan = _build_delay_plan()
            return _execution_runner.run_sync(func, args, kwargs, schedule_plan)

        return sync_wrapper

    return decorator
