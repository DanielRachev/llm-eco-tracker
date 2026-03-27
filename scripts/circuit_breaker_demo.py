from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from demo_support import (
    BASE_DIR,
    build_mock_openai_http_clients,
    configure_demo_logging,
    demo_runtime_patches,
    prepare_telemetry_path,
    read_last_telemetry_record,
)
from llm_eco_tracker import CarbonBudgetExceededError, carbon_aware
from llm_eco_tracker.providers import CsvForecastProvider
from llm_eco_tracker.telemetry import CompositeTelemetrySink, JsonlTelemetrySink, LoggerTelemetrySink


TELEMETRY_PATH = BASE_DIR / "circuit_breaker_demo_telemetry.jsonl"


async def run_demo() -> None:
    configure_demo_logging()
    _, async_http_client = build_mock_openai_http_clients(canned_text_prefix="Budget demo")
    telemetry_path = prepare_telemetry_path(TELEMETRY_PATH.name)
    client = AsyncOpenAI(
        api_key="demo-key",
        base_url="https://example.test/v1",
        http_client=async_http_client,
    )
    sink = CompositeTelemetrySink([LoggerTelemetrySink(), JsonlTelemetrySink(telemetry_path)])

    @carbon_aware(
        max_delay_hours=2,
        forecast_provider=CsvForecastProvider(BASE_DIR / "tests" / "fixtures" / "demo_dirty_forecast.csv"),
        telemetry_sink=sink,
        max_session_gco2eq=0.001,
    )
    async def run_budgeted_job() -> None:
        for call_index in range(5):
            await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"Give a one-sentence sustainability tip #{call_index + 1}.",
                    }
                ],
            )

    try:
        with demo_runtime_patches():
            await run_budgeted_job()
    except CarbonBudgetExceededError as exc:
        record = read_last_telemetry_record(telemetry_path)
        print("\n=== Circuit Breaker Demo ===")
        print(f"Circuit breaker tripped at: {exc.actual_gco2eq:.6f} gCO2eq")
        print(f"Configured budget: {exc.max_session_gco2eq:.6f} gCO2eq")
        print(f"Scheduled delay: {record['schedule_plan']['execution_delay_seconds']:.1f}s")
        print(f"Telemetry file: {telemetry_path}")
        print("Final output: circuit breaker stopped the workload before the batch completed.")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run_demo())
