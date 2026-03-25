from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from llm_eco_tracker import api
from llm_eco_tracker.budget import build_carbon_budget_policy
from llm_eco_tracker.errors import CarbonBudgetExceededError
from llm_eco_tracker.models import ForecastInterval, ForecastSnapshot, SchedulePlan
from llm_eco_tracker.telemetry.adapters import OpenAIChatCompletionsAdapter
from llm_eco_tracker.telemetry.runtime import EcoLogitsRuntime


class CarbonBudgetPolicyTests(unittest.TestCase):
    def test_none_budget_disables_policy(self):
        policy = build_carbon_budget_policy(_schedule_plan(120.0), max_session_gco2eq=None)

        self.assertFalse(policy.enabled)
        self.assertFalse(policy.is_enforced)

    def test_positive_budget_enables_policy(self):
        policy = build_carbon_budget_policy(_schedule_plan(120.0), max_session_gco2eq=50.0)

        self.assertTrue(policy.enabled)
        self.assertTrue(policy.is_enforced)
        self.assertEqual(policy.max_session_gco2eq, 50.0)

    def test_nonpositive_budget_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_carbon_budget_policy(_schedule_plan(120.0), max_session_gco2eq=0.0)


class RuntimeBudgetTests(unittest.TestCase):
    def test_runtime_tracks_cumulative_actual_gco2eq(self):
        runtime = _TestRuntime([_NoOpAdapter()])
        policy = build_carbon_budget_policy(_schedule_plan(100.0), max_session_gco2eq=60.0)

        with runtime.session(carbon_budget_policy=policy) as session:
            runtime._record_energy(0.20)
            runtime._record_energy(0.10)

        self.assertAlmostEqual(session.energy_kwh, 0.30)
        self.assertAlmostEqual(session.actual_gco2eq_so_far, 30.0)
        self.assertFalse(session.carbon_budget_exceeded)

    def test_runtime_raises_when_budget_is_exceeded(self):
        runtime = _TestRuntime([_NoOpAdapter()])
        policy = build_carbon_budget_policy(_schedule_plan(100.0), max_session_gco2eq=40.0)

        with self.assertRaises(CarbonBudgetExceededError) as captured:
            with runtime.session(carbon_budget_policy=policy) as session:
                runtime._record_energy(0.20)
                runtime._record_energy(0.25)

        self.assertAlmostEqual(captured.exception.actual_gco2eq, 45.0)
        self.assertAlmostEqual(captured.exception.energy_kwh, 0.45)
        self.assertAlmostEqual(session.actual_gco2eq_so_far, 45.0)
        self.assertTrue(session.carbon_budget_exceeded)


class CarbonAwareBudgetIntegrationTests(unittest.TestCase):
    def test_sync_budget_breaker_aborts_loop_and_emits_telemetry(self):
        completions_cls, async_completions_cls = _make_fake_openai_classes()
        runtime = _TestRuntime(
            [_TestOpenAIChatCompletionsAdapter(completions_cls, async_completions_cls)]
        )
        sink = _RecordingSink()
        provider = _StaticForecastProvider(100.0)

        with patch.object(api, "_default_telemetry_runtime", runtime):

            @api.carbon_aware(
                max_delay_hours=2,
                forecast_provider=provider,
                telemetry_sink=sink,
                max_session_gco2eq=40.0,
            )
            def run_job():
                completions_cls().create(model="gpt-4o", messages=[])
                completions_cls().create(model="gpt-4o", messages=[])
                completions_cls().create(model="gpt-4o", messages=[])

            with self.assertRaises(CarbonBudgetExceededError):
                run_job()

        self.assertEqual(len(completions_cls.calls), 2)
        self.assertEqual(len(sink.records), 1)
        record = sink.records[0]
        self.assertAlmostEqual(record.emissions.energy_kwh, 0.5)
        self.assertAlmostEqual(record.emissions.actual_gco2eq, 50.0)
        self.assertTrue(record.metadata["carbon_budget_exceeded"])
        self.assertEqual(record.metadata["termination_reason"], "carbon_budget_exceeded")
        self.assertAlmostEqual(record.metadata["actual_gco2eq_so_far"], 50.0)

    def test_async_budget_breaker_aborts_loop_and_emits_telemetry(self):
        completions_cls, async_completions_cls = _make_fake_openai_classes()
        runtime = _TestRuntime(
            [_TestOpenAIChatCompletionsAdapter(completions_cls, async_completions_cls)]
        )
        sink = _RecordingSink()
        provider = _StaticForecastProvider(100.0)

        with patch.object(api, "_default_telemetry_runtime", runtime):

            @api.carbon_aware(
                max_delay_hours=2,
                forecast_provider=provider,
                telemetry_sink=sink,
                max_session_gco2eq=40.0,
            )
            async def run_job():
                await async_completions_cls().create(model="gpt-4o", messages=[])
                await async_completions_cls().create(model="gpt-4o", messages=[])
                await async_completions_cls().create(model="gpt-4o", messages=[])

            with self.assertRaises(CarbonBudgetExceededError):
                asyncio.run(run_job())

        self.assertEqual(len(async_completions_cls.calls), 2)
        self.assertEqual(len(sink.records), 1)
        self.assertTrue(sink.records[0].metadata["carbon_budget_exceeded"])

    def test_budget_can_coexist_with_auto_downgrade(self):
        completions_cls, async_completions_cls = _make_fake_openai_classes()
        runtime = _TestRuntime(
            [_TestOpenAIChatCompletionsAdapter(completions_cls, async_completions_cls)]
        )
        sink = _RecordingSink()
        provider = _StaticForecastProvider(350.0)

        with patch.object(api, "_default_telemetry_runtime", runtime):

            @api.carbon_aware(
                max_delay_hours=2,
                forecast_provider=provider,
                telemetry_sink=sink,
                auto_downgrade=True,
                max_session_gco2eq=120.0,
            )
            def run_job():
                completions_cls().create(model="gpt-4o", messages=[])
                completions_cls().create(model="gpt-4o", messages=[])

            with self.assertRaises(CarbonBudgetExceededError):
                run_job()

        self.assertEqual(completions_cls.calls[0]["model"], "gpt-4o-mini")
        self.assertEqual(completions_cls.calls[1]["model"], "gpt-4o-mini")
        self.assertEqual(sink.records[0].model_usage[0].effective_model, "gpt-4o-mini")


class _TestRuntime(EcoLogitsRuntime):
    def _ensure_ecologits_initialized(self) -> bool:
        return True


class _TestOpenAIChatCompletionsAdapter(OpenAIChatCompletionsAdapter):
    def __init__(self, completions_cls, async_completions_cls):
        super().__init__()
        self._test_classes = (completions_cls, async_completions_cls)

    def _import_openai_classes(self):
        return self._test_classes


class _StaticForecastProvider:
    provider_name = "static_forecast"

    def __init__(self, intensity: float):
        self._intensity = intensity

    def load_forecast(self, max_delay_hours: float) -> ForecastSnapshot:
        del max_delay_hours
        now = datetime.now(timezone.utc)
        interval = ForecastInterval(
            starts_at=now,
            ends_at=now + timedelta(minutes=30),
            carbon_intensity_gco2eq_per_kwh=self._intensity,
        )
        return ForecastSnapshot(intervals=(interval,), reference_time=now)


class _RecordingSink:
    def __init__(self):
        self.records = []

    def emit(self, record) -> None:
        self.records.append(record)


class _NoOpAdapter:
    provider_name = "noop"

    def install(self, session_hooks) -> bool:
        del session_hooks
        return True

    def uninstall(self) -> None:
        return None


def _schedule_plan(optimal_intensity: float) -> SchedulePlan:
    return SchedulePlan(
        baseline_interval=None,
        selected_interval=None,
        baseline_intensity_gco2eq_per_kwh=optimal_intensity,
        optimal_intensity_gco2eq_per_kwh=optimal_intensity,
        raw_delay_seconds=0.0,
        execution_delay_seconds=0.0,
    )


def _make_fake_openai_classes():
    class FakeSyncCompletions:
        calls: list[dict[str, object]] = []

        def create(self, model, messages=None):
            type(self).calls.append({"model": model, "messages": messages})
            return _fake_response()

    class FakeAsyncCompletions:
        calls: list[dict[str, object]] = []

        async def create(self, model, messages=None):
            type(self).calls.append({"model": model, "messages": messages})
            return _fake_response()

    return FakeSyncCompletions, FakeAsyncCompletions


def _fake_response(energy: float = 0.25):
    return SimpleNamespace(impacts=SimpleNamespace(energy=SimpleNamespace(value=energy)))


if __name__ == "__main__":
    unittest.main()
