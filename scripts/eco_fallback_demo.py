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
from llm_eco_tracker import carbon_aware
from llm_eco_tracker.providers import CsvForecastProvider
from llm_eco_tracker.telemetry import CompositeTelemetrySink, JsonlTelemetrySink, LoggerTelemetrySink


TELEMETRY_PATH = BASE_DIR / "eco_fallback_demo_telemetry.jsonl"


async def run_demo() -> None:
    configure_demo_logging()
    request_log: list[dict] = []
    _, async_http_client = build_mock_openai_http_clients(
        request_log=request_log,
        canned_text_prefix="Fallback demo",
    )
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
        auto_downgrade=True,
        dirty_threshold=300.0,
        model_fallbacks={"gpt-4.1": "gpt-4.1-mini"},
    )
    async def run_fallback_job() -> str:
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Explain dirty-grid model downgrading in one sentence."}],
        )
        return str(response.choices[0].message.content)

    try:
        with demo_runtime_patches():
            final_output = await run_fallback_job()
    finally:
        await client.close()

    record = read_last_telemetry_record(telemetry_path)
    usage = record["model_usage"][0]

    print("\n=== Eco-Fallback Demo ===")
    print(f"Requested model: {usage['requested_model']}")
    print(f"Effective model: {usage['effective_model']}")
    print(f"Downgraded: {usage['downgraded']}")
    print(f"Scheduled delay: {record['schedule_plan']['execution_delay_seconds']:.1f}s")
    print(f"Carbon saved: {record['saved_gco2eq']:.6f} gCO2eq")
    print(f"Telemetry file: {telemetry_path}")
    print(f"Final output: {final_output}")


if __name__ == "__main__":
    asyncio.run(run_demo())
