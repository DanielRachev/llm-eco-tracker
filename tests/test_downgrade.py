from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm_eco_tracker import api
from llm_eco_tracker.downgrade import (
    DEFAULT_OPENAI_MODEL_FALLBACKS,
    build_model_downgrade_policy,
)
from llm_eco_tracker.models import ForecastInterval, ForecastSnapshot, SchedulePlan
from llm_eco_tracker.telemetry import JsonlTelemetrySink
from llm_eco_tracker.telemetry.adapters import OpenAIChatCompletionsAdapter
from llm_eco_tracker.telemetry.runtime import EcoLogitsRuntime


class ModelDowngradePolicyTests(unittest.TestCase):
    def test_policy_marks_dirty_grid_when_threshold_is_crossed(self):
        policy = build_model_downgrade_policy(
            _schedule_plan(360.0),
            auto_downgrade=True,
            dirty_threshold=300.0,
        )

        self.assertTrue(policy.is_dirty)
        self.assertEqual(policy.fallback_map["gpt-4o"], "gpt-4o-mini")

    def test_policy_stays_inactive_when_auto_downgrade_is_disabled(self):
        policy = build_model_downgrade_policy(
            _schedule_plan(360.0),
            auto_downgrade=False,
            dirty_threshold=300.0,
        )

        self.assertFalse(policy.is_dirty)

    def test_policy_merges_default_map_with_user_overrides(self):
        policy = build_model_downgrade_policy(
            _schedule_plan(320.0),
            auto_downgrade=True,
            model_fallbacks={
                "gpt-4o": "gpt-4.1-mini",
                "custom-large": "custom-small",
            },
        )

        self.assertEqual(policy.fallback_map["gpt-4o"], "gpt-4.1-mini")
        self.assertEqual(policy.fallback_map["custom-large"], "custom-small")
        self.assertEqual(policy.fallback_map["gpt-4"], DEFAULT_OPENAI_MODEL_FALLBACKS["gpt-4"])


class OpenAIAdapterDowngradeTests(unittest.TestCase):
    def test_sync_adapter_downgrades_keyword_model_on_dirty_grid(self):
        completions_cls, async_completions_cls = _make_fake_openai_classes()
        runtime = _TestRuntime(
            [_TestOpenAIChatCompletionsAdapter(completions_cls, async_completions_cls)]
        )
        policy = build_model_downgrade_policy(_schedule_plan(350.0), auto_downgrade=True)

        with runtime.session(model_downgrade_policy=policy) as session:
            completions_cls().create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(completions_cls.calls[0]["model"], "gpt-4o-mini")
        self.assertAlmostEqual(session.energy_kwh, 0.25)
        self.assertEqual(len(session.model_usage), 1)
        self.assertEqual(session.model_usage[0].requested_model, "gpt-4o")
        self.assertEqual(session.model_usage[0].effective_model, "gpt-4o-mini")
        self.assertTrue(session.model_usage[0].downgraded)

    def test_sync_adapter_handles_positional_model_arguments(self):
        completions_cls, async_completions_cls = _make_fake_openai_classes()
        runtime = _TestRuntime(
            [_TestOpenAIChatCompletionsAdapter(completions_cls, async_completions_cls)]
        )
        policy = build_model_downgrade_policy(_schedule_plan(350.0), auto_downgrade=True)

        with runtime.session(model_downgrade_policy=policy):
            completions_cls().create("gpt-4-turbo", messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(completions_cls.calls[0]["model"], "gpt-4o-mini")

    def test_async_adapter_keeps_original_model_on_clean_grid(self):
        completions_cls, async_completions_cls = _make_fake_openai_classes()
        runtime = _TestRuntime(
            [_TestOpenAIChatCompletionsAdapter(completions_cls, async_completions_cls)]
        )
        policy = build_model_downgrade_policy(
            _schedule_plan(120.0),
            auto_downgrade=True,
            dirty_threshold=300.0,
        )

        with runtime.session(model_downgrade_policy=policy) as session:
            asyncio.run(
                async_completions_cls().create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                )
            )

        self.assertEqual(async_completions_cls.calls[0]["model"], "gpt-4o")
        self.assertEqual(session.model_usage[0].effective_model, "gpt-4o")
        self.assertFalse(session.model_usage[0].downgraded)

    def test_unmapped_model_warns_once_and_keeps_original_model(self):
        completions_cls, async_completions_cls = _make_fake_openai_classes()
        runtime = _TestRuntime(
            [_TestOpenAIChatCompletionsAdapter(completions_cls, async_completions_cls)]
        )
        policy = build_model_downgrade_policy(_schedule_plan(350.0), auto_downgrade=True)

        with self.assertLogs("llm_eco_tracker.telemetry.adapters.openai", level="WARNING") as captured:
            with runtime.session(model_downgrade_policy=policy):
                completions_cls().create(model="custom-large", messages=[])
                completions_cls().create(model="custom-large", messages=[])

        self.assertEqual(completions_cls.calls[0]["model"], "custom-large")
        self.assertEqual(completions_cls.calls[1]["model"], "custom-large")
        self.assertEqual(len(captured.output), 1)


class CarbonAwareDowngradeIntegrationTests(unittest.TestCase):
    def test_decorated_function_tracks_mixed_model_usage(self):
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
            )
            def run_job():
                completions_cls().create(model="gpt-4o", messages=[])
                completions_cls().create(model="custom-large", messages=[])

            with self.assertLogs("llm_eco_tracker.telemetry.adapters.openai", level="WARNING") as captured:
                run_job()

        self.assertEqual(len(sink.records), 1)
        self.assertEqual(len(captured.output), 1)
        record = sink.records[0]
        self.assertAlmostEqual(record.emissions.energy_kwh, 0.5)
        self.assertIsNone(record.model)

        model_usage = {
            (entry.requested_model, entry.effective_model): entry for entry in record.model_usage
        }
        self.assertIn(("gpt-4o", "gpt-4o-mini"), model_usage)
        self.assertIn(("custom-large", "custom-large"), model_usage)
        self.assertTrue(model_usage[("gpt-4o", "gpt-4o-mini")].downgraded)
        self.assertFalse(model_usage[("custom-large", "custom-large")].downgraded)

    def test_decorated_function_keeps_original_model_when_auto_downgrade_is_disabled(self):
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
                auto_downgrade=False,
            )
            def run_job():
                completions_cls().create(model="gpt-4o", messages=[])

            run_job()

        self.assertEqual(completions_cls.calls[0]["model"], "gpt-4o")
        self.assertEqual(sink.records[0].model, "gpt-4o")
        self.assertFalse(sink.records[0].model_usage[0].downgraded)

    def test_jsonl_sink_serializes_model_usage_summary(self):
        completions_cls, async_completions_cls = _make_fake_openai_classes()
        runtime = _TestRuntime(
            [_TestOpenAIChatCompletionsAdapter(completions_cls, async_completions_cls)]
        )
        provider = _StaticForecastProvider(350.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            telemetry_path = Path(temp_dir) / "eco_telemetry.jsonl"
            telemetry_sink = JsonlTelemetrySink(telemetry_path)

            with patch.object(api, "_default_telemetry_runtime", runtime):

                @api.carbon_aware(
                    max_delay_hours=2,
                    forecast_provider=provider,
                    telemetry_sink=telemetry_sink,
                    auto_downgrade=True,
                )
                def run_job():
                    completions_cls().create(model="gpt-4o", messages=[])

                run_job()

            payload = json.loads(telemetry_path.read_text(encoding="utf-8").strip())

        self.assertEqual(payload["model"], "gpt-4o-mini")
        self.assertEqual(payload["model_usage"][0]["requested_model"], "gpt-4o")
        self.assertEqual(payload["model_usage"][0]["effective_model"], "gpt-4o-mini")
        self.assertTrue(payload["model_usage"][0]["downgraded"])


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
