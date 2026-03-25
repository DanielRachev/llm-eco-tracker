import functools
import inspect
import logging
from collections.abc import Mapping
from pathlib import Path

from .downgrade import build_model_downgrade_policy
from .execution import ExecutionRunner
from .models import SchedulePlan
from .planning import (
    apply_jitter_to_plan,
    build_schedule_plan,
    immediate_schedule_plan,
)
from .providers import UKCarbonIntensityProvider
from .providers.base import ForecastProvider
from .telemetry import EcoLogitsRuntime, JsonlTelemetrySink
from .telemetry.adapters import OpenAIChatCompletionsAdapter
from .telemetry.base import TelemetrySink


logger = logging.getLogger(__name__)

_telemetry_path = Path("eco_telemetry.jsonl")
_default_telemetry_runtime = EcoLogitsRuntime([OpenAIChatCompletionsAdapter()])
_default_telemetry_sink = JsonlTelemetrySink(_telemetry_path)


def _resolve_forecast_provider(forecast_provider: ForecastProvider | None) -> ForecastProvider:
    if forecast_provider is None:
        return UKCarbonIntensityProvider()

    return forecast_provider


def _resolve_telemetry_sink(telemetry_sink: TelemetrySink | None) -> TelemetrySink:
    return telemetry_sink or _default_telemetry_sink


def _log_schedule_plan(forecast_provider: ForecastProvider, schedule_plan: SchedulePlan):
    if schedule_plan.baseline_interval is None or schedule_plan.selected_interval is None:
        return

    logger.info(
        "Current forecast from '%s': %.1f gCO2eq/kWh",
        forecast_provider.provider_name,
        schedule_plan.baseline_intensity_gco2eq_per_kwh,
    )
    logger.info(
        "Best forecast from '%s': %.1f gCO2eq/kWh at %s",
        forecast_provider.provider_name,
        schedule_plan.optimal_intensity_gco2eq_per_kwh,
        schedule_plan.selected_interval.starts_at.isoformat(),
    )


def _get_schedule_plan(max_delay_hours: float, forecast_provider: ForecastProvider) -> SchedulePlan:
    if max_delay_hours <= 0:
        return immediate_schedule_plan()

    forecast_snapshot = forecast_provider.load_forecast(max_delay_hours)
    schedule_plan = build_schedule_plan(
        forecast_snapshot.intervals,
        max_delay_hours,
        reference_time=forecast_snapshot.reference_time,
    )
    _log_schedule_plan(forecast_provider, schedule_plan)
    return schedule_plan


def carbon_aware(
    *,
    max_delay_hours: int = 2,
    forecast_provider: ForecastProvider | None = None,
    telemetry_sink: TelemetrySink | None = None,
    auto_downgrade: bool = False,
    dirty_threshold: float = 300.0,
    model_fallbacks: Mapping[str, str] | None = None,
):
    """
    Delay non-urgent work until a greener grid window, then record session-wide
    energy telemetry for OpenAI chat completions executed inside the function.

    Args:
        max_delay_hours (int): Maximum time to wait for a greener grid window.
        forecast_provider (ForecastProvider | None): Source of carbon-intensity forecasts.
        telemetry_sink (TelemetrySink | None): Destination for normalized telemetry records.
        auto_downgrade (bool): Rewrite supported model requests on dirty grids.
        dirty_threshold (float): Grid intensity threshold for model downgrading.
        model_fallbacks (Mapping[str, str] | None): Additional exact-match model fallbacks.
    """

    resolved_forecast_provider = _resolve_forecast_provider(forecast_provider)
    resolved_telemetry_sink = _resolve_telemetry_sink(telemetry_sink)
    execution_runner = ExecutionRunner(
        _default_telemetry_runtime,
        resolved_telemetry_sink,
        llm_provider="openai",
    )

    def _log_intercept(func):
        logger.info("Intercepting call to '%s'", func.__name__)
        logger.info(
            "Forecast provider: %s, max delay: %sh",
            resolved_forecast_provider.provider_name,
            max_delay_hours,
        )
        logger.info(
            "Auto downgrade: %s, dirty threshold: %.1f gCO2eq/kWh",
            auto_downgrade,
            dirty_threshold,
        )

    def _build_delay_plan():
        schedule_plan = _get_schedule_plan(max_delay_hours, resolved_forecast_provider)
        final_plan = apply_jitter_to_plan(schedule_plan)

        logger.info(
            "Scheduling plan: baseline %.1f gCO2eq/kWh, optimal %.1f gCO2eq/kWh, raw delay %.2fs, execution delay %.2fs",
            final_plan.baseline_intensity_gco2eq_per_kwh,
            final_plan.optimal_intensity_gco2eq_per_kwh,
            final_plan.raw_delay_seconds,
            final_plan.execution_delay_seconds,
        )

        return final_plan

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                _log_intercept(func)
                schedule_plan = _build_delay_plan()
                model_downgrade_policy = build_model_downgrade_policy(
                    schedule_plan,
                    auto_downgrade=auto_downgrade,
                    dirty_threshold=dirty_threshold,
                    model_fallbacks=model_fallbacks,
                )
                return await execution_runner.run_async(
                    func,
                    args,
                    kwargs,
                    schedule_plan,
                    forecast_provider=resolved_forecast_provider.provider_name,
                    model_downgrade_policy=model_downgrade_policy,
                )

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _log_intercept(func)
            schedule_plan = _build_delay_plan()
            model_downgrade_policy = build_model_downgrade_policy(
                schedule_plan,
                auto_downgrade=auto_downgrade,
                dirty_threshold=dirty_threshold,
                model_fallbacks=model_fallbacks,
            )
            return execution_runner.run_sync(
                func,
                args,
                kwargs,
                schedule_plan,
                forecast_provider=resolved_forecast_provider.provider_name,
                model_downgrade_policy=model_downgrade_policy,
            )

        return sync_wrapper

    return decorator
