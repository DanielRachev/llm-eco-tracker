from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI, OpenAI

from llm_eco_tracker import api
from llm_eco_tracker.models import ForecastInterval, ForecastSnapshot
from llm_eco_tracker.telemetry import JsonlTelemetrySink
from llm_eco_tracker.telemetry.adapters import OpenAIChatCompletionsAdapter
from llm_eco_tracker.telemetry.runtime import EcoLogitsRuntime


class RealOpenAITelemetryTests(unittest.TestCase):
    def test_runtime_records_mean_energy_from_real_openai_response(self):
        client = _build_sync_client()
        runtime = EcoLogitsRuntime([OpenAIChatCompletionsAdapter()])

        try:
            with patch("ecologits.log.logger.warning_once", return_value=None):
                with runtime.session() as session:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "hello"}],
                    )
                    expected_energy_kwh = response.impacts.energy.value.mean

            self.assertGreater(session.energy_kwh, 0.0)
            self.assertAlmostEqual(session.energy_kwh, expected_energy_kwh)
        finally:
            client.close()

    def test_carbon_aware_decorated_function_writes_telemetry_for_real_openai_client(self):
        runtime = EcoLogitsRuntime([OpenAIChatCompletionsAdapter()])
        provider = _DelayedStaticForecastProvider()

        with tempfile.TemporaryDirectory() as temp_dir:
            telemetry_path = Path(temp_dir) / "eco_telemetry.jsonl"
            telemetry_sink = JsonlTelemetrySink(telemetry_path)
            client = _build_async_client()

            with patch.object(api, "_default_telemetry_runtime", runtime):

                @api.carbon_aware(
                    max_delay_hours=2,
                    forecast_provider=provider,
                    telemetry_sink=telemetry_sink,
                )
                async def run_job():
                    await client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "hello"}],
                    )

                try:
                    with (
                        patch("ecologits.log.logger.warning_once", return_value=None),
                        patch("llm_eco_tracker.execution.asyncio.sleep", side_effect=_mock_sleep),
                        patch(
                            "llm_eco_tracker.api.apply_jitter_to_plan",
                            side_effect=lambda schedule_plan: schedule_plan,
                        ),
                    ):
                        asyncio.run(run_job())
                finally:
                    asyncio.run(client.close())

            payload = json.loads(telemetry_path.read_text(encoding="utf-8").strip())

        self.assertGreater(payload["energy_kwh"], 0.0)
        self.assertGreater(payload["schedule_plan"]["execution_delay_seconds"], 0.0)
        self.assertEqual(payload["forecast_provider"], "static_forecast")
        self.assertEqual(payload["llm_provider"], "openai")
        self.assertEqual(payload["model_usage"][0]["requested_model"], "gpt-4o-mini")
        self.assertEqual(payload["model_usage"][0]["effective_model"], "gpt-4o-mini")
        self.assertFalse(payload["model_usage"][0]["downgraded"])


class _DelayedStaticForecastProvider:
    provider_name = "static_forecast"

    def load_forecast(self, max_delay_hours: float) -> ForecastSnapshot:
        del max_delay_hours
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        baseline = ForecastInterval(
            starts_at=now,
            ends_at=now + timedelta(minutes=30),
            carbon_intensity_gco2eq_per_kwh=350.0,
        )
        greener = ForecastInterval(
            starts_at=now + timedelta(hours=2),
            ends_at=now + timedelta(hours=2, minutes=30),
            carbon_intensity_gco2eq_per_kwh=120.0,
        )
        return ForecastSnapshot(intervals=(baseline, greener), reference_time=now)


def _build_sync_client() -> OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content.decode("utf-8"))
        requested_model = request_payload["model"]
        return httpx.Response(200, json=_mock_completion_payload(requested_model))

    return OpenAI(
        api_key="test-key",
        base_url="https://example.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _build_async_client() -> AsyncOpenAI:
    async def handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content.decode("utf-8"))
        requested_model = request_payload["model"]
        return httpx.Response(200, json=_mock_completion_payload(requested_model))

    return AsyncOpenAI(
        api_key="test-key",
        base_url="https://example.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _mock_completion_payload(model: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1710000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Mock reply"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
    }


async def _mock_sleep(seconds: float) -> None:
    del seconds


if __name__ == "__main__":
    unittest.main()
