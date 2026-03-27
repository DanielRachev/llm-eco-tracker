from __future__ import annotations

import asyncio

from langchain_core.runnables import RunnableLambda
from openai import AsyncOpenAI

from demo_support import (
    BASE_DIR,
    build_mock_openai_http_clients,
    configure_demo_logging,
    demo_runtime_patches,
    prepare_telemetry_path,
    read_last_telemetry_record,
    summarize_model_usage,
)
from llm_eco_tracker import carbon_aware
from llm_eco_tracker.providers import CsvForecastProvider
from llm_eco_tracker.telemetry import CompositeTelemetrySink, JsonlTelemetrySink, LoggerTelemetrySink


TELEMETRY_PATH = BASE_DIR / "langchain_agentic_demo_telemetry.jsonl"


async def run_demo() -> None:
    configure_demo_logging()
    request_log: list[dict] = []
    _, async_http_client = build_mock_openai_http_clients(
        request_log=request_log,
        canned_text_prefix="Agent step",
    )
    telemetry_path = prepare_telemetry_path(TELEMETRY_PATH.name)
    client = AsyncOpenAI(
        api_key="demo-key",
        base_url="https://example.test/v1",
        http_client=async_http_client,
    )
    sink = CompositeTelemetrySink([LoggerTelemetrySink(), JsonlTelemetrySink(telemetry_path)])

    async def invoke_model(prompt: str) -> str:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are concise and structured."},
                {"role": "user", "content": prompt},
            ],
        )
        return str(response.choices[0].message.content)

    async def planning_step(topic: str) -> str:
        return await invoke_model(f"Plan three steps for this workflow topic: {topic}")

    async def critique_step(plan: str) -> str:
        critique = await invoke_model(f"Critique this workflow plan:\n{plan}")
        return f"Plan:\n{plan}\n\nCritique:\n{critique}"

    async def revision_step(context: str) -> str:
        return await invoke_model(f"Revise this workflow using the critique:\n{context}")

    async def summary_step(revision: str) -> str:
        return await invoke_model(f"Summarize this final workflow in one sentence:\n{revision}")

    workflow = (
        RunnableLambda(planning_step)
        | RunnableLambda(critique_step)
        | RunnableLambda(revision_step)
        | RunnableLambda(summary_step)
    )

    @carbon_aware(
        max_delay_hours=2,
        forecast_provider=CsvForecastProvider(BASE_DIR / "tests" / "fixtures" / "demo_dirty_forecast.csv"),
        telemetry_sink=sink,
    )
    async def run_agentic_workflow() -> str:
        return await workflow.ainvoke("sustainable bug triage for a developer team")

    try:
        with demo_runtime_patches():
            final_output = await run_agentic_workflow()
    finally:
        await client.close()

    record = read_last_telemetry_record(telemetry_path)
    intercepted_calls = summarize_model_usage(record)

    print("\n=== LangChain Agentic Demo ===")
    print(f"Intercepted OpenAI calls through LangChain: {intercepted_calls}")
    print(f"Scheduled delay: {record['schedule_plan']['execution_delay_seconds']:.1f}s")
    print(f"Carbon saved: {record['saved_gco2eq']:.6f} gCO2eq")
    print(f"Telemetry file: {telemetry_path}")
    print(f"Final output: {final_output}")


if __name__ == "__main__":
    asyncio.run(run_demo())
