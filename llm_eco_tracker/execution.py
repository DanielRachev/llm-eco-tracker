from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from .emissions import summarize_emissions
from .models import (
    CarbonBudgetPolicy,
    ModelDowngradePolicy,
    ModelUsageSummary,
    SchedulePlan,
    TelemetryRecord,
)
from .telemetry.base import TelemetrySink
from .telemetry.runtime import EcoLogitsRuntime

logger = logging.getLogger(__name__)


class ExecutionRunner:
    def __init__(
        self,
        telemetry_runtime: EcoLogitsRuntime,
        telemetry_sink: TelemetrySink,
        *,
        llm_provider: str | None = None,
    ):
        self._telemetry_runtime = telemetry_runtime
        self._telemetry_sink = telemetry_sink
        self._llm_provider = llm_provider

    async def run_async(
        self,
        func,
        args,
        kwargs,
        schedule_plan: SchedulePlan,
        *,
        forecast_provider: str | None = None,
        model: str | None = None,
        carbon_budget_policy: CarbonBudgetPolicy | None = None,
        model_downgrade_policy: ModelDowngradePolicy | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._log_async_delay(schedule_plan)

        if schedule_plan.execution_delay_seconds > 0:
            await asyncio.sleep(schedule_plan.execution_delay_seconds)

        logger.info("Greener window reached. Proceeding with execution.")

        total_kwh = 0.0
        model_usage: tuple[ModelUsageSummary, ...] = ()
        session_metadata: dict[str, Any] = {}
        try:
            with self._telemetry_runtime.session(
                carbon_budget_policy=carbon_budget_policy,
                model_downgrade_policy=model_downgrade_policy
            ) as telemetry_session:
                try:
                    result = await func(*args, **kwargs)
                finally:
                    total_kwh = telemetry_session.energy_kwh
                    model_usage = telemetry_session.model_usage
                    session_metadata = telemetry_session.session_metadata
        finally:
            self._finalize_execution(
                total_kwh,
                schedule_plan,
                forecast_provider=forecast_provider,
                model=self._resolve_legacy_model(model, model_usage),
                model_usage=model_usage,
                metadata=self._merge_metadata(metadata, session_metadata),
            )

        return result

    def run_sync(
        self,
        func,
        args,
        kwargs,
        schedule_plan: SchedulePlan,
        *,
        forecast_provider: str | None = None,
        model: str | None = None,
        carbon_budget_policy: CarbonBudgetPolicy | None = None,
        model_downgrade_policy: ModelDowngradePolicy | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        done = threading.Event()
        outcome: dict[str, Any] = {}

        def run_later():
            total_kwh = 0.0
            model_usage: tuple[ModelUsageSummary, ...] = ()
            session_metadata: dict[str, Any] = {}
            try:
                logger.info("Greener window reached. Proceeding with execution.")
                with self._telemetry_runtime.session(
                    carbon_budget_policy=carbon_budget_policy,
                    model_downgrade_policy=model_downgrade_policy
                ) as telemetry_session:
                    try:
                        outcome["result"] = func(*args, **kwargs)
                    finally:
                        total_kwh = telemetry_session.energy_kwh
                        model_usage = telemetry_session.model_usage
                        session_metadata = telemetry_session.session_metadata
            except BaseException as exc:
                outcome["exception"] = exc
            finally:
                self._finalize_execution(
                    total_kwh,
                    schedule_plan,
                    forecast_provider=forecast_provider,
                    model=self._resolve_legacy_model(model, model_usage),
                    model_usage=model_usage,
                    metadata=self._merge_metadata(metadata, session_metadata),
                )
                done.set()

        timer = self._start_sync_execution(schedule_plan, run_later)

        try:
            done.wait()
        except KeyboardInterrupt:
            if timer is not None:
                timer.cancel()
            raise

        if "exception" in outcome:
            raise outcome["exception"]

        return outcome.get("result")

    @staticmethod
    def _log_async_delay(schedule_plan: SchedulePlan) -> None:
        if schedule_plan.execution_delay_seconds > 0:
            logger.info(
                "Carbon-aware scheduler: Awaiting %.2f seconds to reach a greener grid window.",
                schedule_plan.execution_delay_seconds,
            )
        else:
            logger.info("Carbon-aware scheduler: Executing immediately.")

    @staticmethod
    def _start_sync_execution(schedule_plan: SchedulePlan, run_later):
        if schedule_plan.execution_delay_seconds > 0:
            logger.info(
                "Carbon-aware scheduler: Scheduling execution in %.2f seconds.",
                schedule_plan.execution_delay_seconds,
            )
            timer = threading.Timer(schedule_plan.execution_delay_seconds, run_later)
            timer.start()
            return timer

        logger.info("Carbon-aware scheduler: Executing immediately.")
        run_later()
        return None

    def _finalize_execution(
        self,
        total_kwh: float,
        schedule_plan: SchedulePlan,
        *,
        forecast_provider: str | None = None,
        model: str | None = None,
        model_usage: tuple[ModelUsageSummary, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
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
        self._telemetry_sink.emit(
            TelemetryRecord(
                timestamp=datetime.now(timezone.utc),
                emissions=emission_summary,
                schedule_plan=schedule_plan,
                forecast_provider=forecast_provider,
                llm_provider=self._llm_provider,
                model=model,
                model_usage=model_usage,
                metadata=dict(metadata or {}),
            )
        )

    @staticmethod
    def _resolve_legacy_model(
        model: str | None,
        model_usage: tuple[ModelUsageSummary, ...],
    ) -> str | None:
        if model is not None:
            return model
        if len(model_usage) != 1:
            return None
        return model_usage[0].effective_model

    @staticmethod
    def _merge_metadata(
        metadata: dict[str, Any] | None,
        session_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged_metadata = dict(metadata or {})
        merged_metadata.update(session_metadata or {})
        return merged_metadata
