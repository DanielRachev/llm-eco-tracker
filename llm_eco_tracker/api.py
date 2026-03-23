import asyncio
import functools
import inspect
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from .emissions import summarize_emissions
from .models import SchedulePlan, TelemetryRecord
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


def _log_telemetry_summary(total_kwh, schedule_plan: SchedulePlan):
    if total_kwh <= 0:
        logger.info("Session complete. No EcoLogits energy impacts were captured.")
        return

    emission_summary = summarize_emissions(
        total_kwh,
        schedule_plan.baseline_intensity_gco2eq_per_kwh,
        schedule_plan.optimal_intensity_gco2eq_per_kwh,
    )

    logger.info("Session complete. Total energy: %.6f kWh", emission_summary.energy_kwh)
    logger.info("Carbon delta: %.4f gCO2eq", emission_summary.saved_gco2eq)
    _telemetry_sink.emit(
        TelemetryRecord(
            timestamp=datetime.now(timezone.utc),
            emissions=emission_summary,
            schedule_plan=schedule_plan,
            llm_provider="openai",
        )
    )


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

                if schedule_plan.execution_delay_seconds > 0:
                    logger.info(
                        "Carbon-aware scheduler: Awaiting %.2f seconds to reach a greener grid window.",
                        schedule_plan.execution_delay_seconds,
                    )
                    await asyncio.sleep(schedule_plan.execution_delay_seconds)
                else:
                    logger.info("Carbon-aware scheduler: Executing immediately.")

                logger.info("Greener window reached. Proceeding with execution.")

                with _telemetry_runtime.session() as telemetry_session:
                    try:
                        result = await func(*args, **kwargs)
                    finally:
                        total_kwh = telemetry_session.energy_kwh

                _log_telemetry_summary(total_kwh, schedule_plan)
                return result

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _log_intercept(func)
            schedule_plan = _build_delay_plan()

            done = threading.Event()
            outcome = {}

            def run_later():
                try:
                    logger.info("Greener window reached. Proceeding with execution.")
                    with _telemetry_runtime.session() as telemetry_session:
                        try:
                            outcome["result"] = func(*args, **kwargs)
                        finally:
                            total_kwh = telemetry_session.energy_kwh
                    _log_telemetry_summary(total_kwh, schedule_plan)
                except BaseException as exc:
                    outcome["exception"] = exc
                finally:
                    done.set()

            if schedule_plan.execution_delay_seconds > 0:
                logger.info(
                    "Carbon-aware scheduler: Scheduling execution in %.2f seconds.",
                    schedule_plan.execution_delay_seconds,
                )
                timer = threading.Timer(schedule_plan.execution_delay_seconds, run_later)
                timer.start()
            else:
                logger.info("Carbon-aware scheduler: Executing immediately.")
                timer = None
                run_later()

            try:
                done.wait()
            except KeyboardInterrupt:
                if timer is not None:
                    timer.cancel()
                raise

            if "exception" in outcome:
                raise outcome["exception"]

            return outcome.get("result")

        return sync_wrapper

    return decorator
