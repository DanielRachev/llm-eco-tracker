from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
from anthropic import Anthropic, AsyncAnthropic
from anthropic.resources.messages import AsyncMessages, Messages

from llm_eco_tracker import api
from llm_eco_tracker.models import ForecastInterval, ForecastSnapshot
from llm_eco_tracker.telemetry import JsonlTelemetrySink
from llm_eco_tracker.telemetry.adapters import AnthropicMessagesAdapter
from llm_eco_tracker.telemetry.runtime import EcoLogitsRuntime


MODEL = "claude-sonnet-4-0"
FALLBACK_MODEL = "claude-3-haiku-20240307"


class RealAnthropicTelemetryTests(unittest.TestCase):
    def test_runtime_restores_anthropic_create_methods_after_session(self):
        runtime = EcoLogitsRuntime([AnthropicMessagesAdapter()])
        raw_sync_create = Messages.__dict__["create"]
        raw_async_create = AsyncMessages.__dict__["create"]

        with patch("ecologits.log.logger.warning_once", return_value=None):
            with runtime.session():
                self.assertIsNot(Messages.__dict__["create"], raw_sync_create)
                self.assertIsNot(AsyncMessages.__dict__["create"], raw_async_create)

        self.assertIs(Messages.__dict__["create"], raw_sync_create)
        self.assertIs(AsyncMessages.__dict__["create"], raw_async_create)

    def test_runtime_records_mean_energy_from_real_anthropic_response(self):
        client = _build_sync_client()
        runtime = EcoLogitsRuntime([AnthropicMessagesAdapter()])

        try:
            with patch("ecologits.log.logger.warning_once", return_value=None):
                with runtime.session() as session:
                    response = client.messages.create(
                        model=MODEL,
                        max_tokens=32,
                        messages=[{"role": "user", "content": "hello"}],
                    )
                    expected_energy_kwh = response.impacts.energy.value.mean

            self.assertGreater(session.energy_kwh, 0.0)
            self.assertAlmostEqual(session.energy_kwh, expected_energy_kwh)
            self.assertEqual(session.llm_provider, "anthropic")
        finally:
            client.close()

    def test_carbon_aware_decorated_function_writes_anthropic_telemetry(self):
        runtime = EcoLogitsRuntime([AnthropicMessagesAdapter()])
        provider = _DelayedStaticForecastProvider()

        with tempfile.TemporaryDirectory() as temp_dir:
            telemetry_path = Path(temp_dir) / "eco_telemetry.jsonl"
            telemetry_sink = JsonlTelemetrySink(telemetry_path)
            captured_models: list[str] = []
            client = _build_async_client(captured_models=captured_models)

            with patch.object(api, "_default_telemetry_runtime", runtime):

                @api.carbon_aware(
                    max_delay_hours=2,
                    forecast_provider=provider,
                    telemetry_sink=telemetry_sink,
                    auto_downgrade=True,
                    model_fallbacks={MODEL: FALLBACK_MODEL},
                )
                async def run_job():
                    await client.messages.create(
                        model=MODEL,
                        max_tokens=32,
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

        self.assertEqual(captured_models, [FALLBACK_MODEL])
        self.assertGreater(payload["energy_kwh"], 0.0)
        self.assertGreater(payload["schedule_plan"]["execution_delay_seconds"], 0.0)
        self.assertEqual(payload["forecast_provider"], "static_forecast")
        self.assertEqual(payload["llm_provider"], "anthropic")
        self.assertEqual(payload["model"], FALLBACK_MODEL)
        self.assertEqual(payload["model_usage"][0]["requested_model"], MODEL)
        self.assertEqual(payload["model_usage"][0]["effective_model"], FALLBACK_MODEL)
        self.assertTrue(payload["model_usage"][0]["downgraded"])


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
            carbon_intensity_gco2eq_per_kwh=320.0,
        )
        return ForecastSnapshot(intervals=(baseline, greener), reference_time=now)


def _build_sync_client() -> Anthropic:
    def handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content.decode("utf-8"))
        requested_model = request_payload["model"]
        return httpx.Response(200, json=_mock_message_payload(requested_model))

    return Anthropic(
        api_key="test-key",
        base_url="https://example.test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _build_async_client(*, captured_models: list[str]) -> AsyncAnthropic:
    async def handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content.decode("utf-8"))
        requested_model = request_payload["model"]
        captured_models.append(requested_model)
        return httpx.Response(200, json=_mock_message_payload(requested_model))

    return AsyncAnthropic(
        api_key="test-key",
        base_url="https://example.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _mock_message_payload(model: str) -> dict[str, object]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "Mock reply"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 8,
        },
    }


async def _mock_sleep(seconds: float) -> None:
    del seconds


if __name__ == "__main__":
    unittest.main()
