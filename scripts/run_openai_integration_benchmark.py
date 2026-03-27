import argparse
import asyncio
import json
import logging
import sys
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI
from openai.resources.chat.completions import AsyncCompletions, Completions

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from llm_eco_tracker.api import _default_telemetry_runtime, carbon_aware
from llm_eco_tracker.benchmarking import (
    SlidingCsvForecastProvider,
    load_actual_intensity_mapping,
    lookup_actual_intensity,
    mock_sleep,
    read_last_new_jsonl_record,
    reset_output_file,
)
from llm_eco_tracker.emissions import summarize_emissions
from llm_eco_tracker.telemetry import JsonlTelemetrySink

DEFAULT_CSV = BASE_DIR / "tests" / "fixtures" / "mock_forecast.csv"
DEFAULT_TELEMETRY_PATH = BASE_DIR / "openai_integration_telemetry.jsonl"
DEFAULT_SUMMARY_PATH = BASE_DIR / "openai_integration_summary.json"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_START_OFFSET = 38
DEFAULT_CALL_COUNT = 1
DEFAULT_MAX_DELAY_HOURS = 4
DEFAULT_PROMPT_TOKENS = 24
DEFAULT_COMPLETION_TOKENS = 32

logging.getLogger("llm_eco_tracker").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EcoTracker OpenAI integration benchmark")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--telemetry-path", type=Path, default=DEFAULT_TELEMETRY_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--start-offset", type=int, default=DEFAULT_START_OFFSET)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--call-count", type=int, default=DEFAULT_CALL_COUNT)
    parser.add_argument("--max-delay-hours", type=float, default=DEFAULT_MAX_DELAY_HOURS)
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--completion-tokens", type=int, default=DEFAULT_COMPLETION_TOKENS)
    parser.add_argument(
        "--real-sleep",
        action="store_true",
        help="Use real scheduler waiting instead of fast-forwarding through sleeps.",
    )
    parser.add_argument(
        "--keep-jitter",
        action="store_true",
        help="Keep scheduler jitter enabled instead of forcing deterministic delay selection.",
    )
    return parser.parse_args()


def identity_schedule_plan(schedule_plan, *args, **kwargs):
    del args, kwargs
    return schedule_plan


def build_mock_openai_client(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> AsyncOpenAI:
    async def handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content.decode("utf-8"))
        requested_model = request_payload.get("model", model)
        content = f"Mocked reply for model {requested_model}"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-benchmark",
                "object": "chat.completion",
                "created": 1710000000,
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AsyncOpenAI(
        api_key="benchmark-key",
        base_url="https://example.test/v1",
        http_client=http_client,
    )


async def run_baseline_job(
    client: AsyncOpenAI,
    *,
    model: str,
    call_count: int,
) -> float:
    with _default_telemetry_runtime.session() as session:
        for call_index in range(call_count):
            await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Benchmark request {call_index + 1}",
                    }
                ],
            )
        return session.energy_kwh


def build_carbon_aware_job(
    provider: SlidingCsvForecastProvider,
    telemetry_sink: JsonlTelemetrySink,
    client: AsyncOpenAI,
    *,
    model: str,
    max_delay_hours: float,
):
    @carbon_aware(
        max_delay_hours=max_delay_hours,
        forecast_provider=provider,
        telemetry_sink=telemetry_sink,
    )
    async def run_job(call_count: int) -> None:
        for call_index in range(call_count):
            await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Benchmark request {call_index + 1}",
                    }
                ],
            )

    return run_job


def write_summary(summary_path: Path, summary: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def snapshot_openai_create_methods() -> dict[str, object | None]:
    return {
        "sync_create": Completions.__dict__.get("create"),
        "async_create": AsyncCompletions.__dict__.get("create"),
    }


def methods_restored(snapshot: dict[str, object | None]) -> bool:
    current = snapshot_openai_create_methods()
    return (
        current["sync_create"] is snapshot["sync_create"]
        and current["async_create"] is snapshot["async_create"]
    )


def print_summary(summary: dict[str, object]) -> None:
    print("=" * 72)
    print("OPENAI END-TO-END INTEGRATION BENCHMARK")
    print("=" * 72)
    print(f"Submission time:         {summary['submission_time']}")
    print(f"Scheduled time:          {summary['scheduled_time']}")
    print(f"Model:                   {summary['model']}")
    print(f"Calls in session:        {summary['call_count']}")
    print(f"Telemetry captured:      {summary['telemetry_captured']}")
    print(f"Telemetry file created:  {summary['telemetry_file_created']}")
    print(f"Model usage captured:    {summary['model_usage_captured']}")
    print(f"SDK restored:            {summary['sdk_methods_restored']}")
    print(f"Baseline energy:         {summary['baseline_energy_kwh']:.10f} kWh")
    print(f"Carbon-aware energy:     {summary['carbon_aware_energy_kwh']:.10f} kWh")
    print(f"Baseline actual:         {summary['baseline_actual_gco2eq']:.8f} gCO2eq")
    print(f"Carbon-aware actual:     {summary['carbon_aware_actual_gco2eq']:.8f} gCO2eq")
    print(f"Saved:                   {summary['saved_gco2eq']:.8f} gCO2eq")
    print(f"Reduction:               {summary['saved_pct']:.4f}%")
    print(f"Delay:                   {summary['scheduled_delay_h']:.4f} h")
    print(f"Telemetry JSONL:         {summary['telemetry_path']}")
    print(f"Summary JSON:            {summary['summary_path']}")
    print("=" * 72)


async def run_benchmark() -> None:
    args = parse_args()

    provider = SlidingCsvForecastProvider(args.csv_path)
    provider.set_start_offset(args.start_offset)
    actual_mapping = load_actual_intensity_mapping(args.csv_path)
    telemetry_path = reset_output_file(args.telemetry_path)
    telemetry_sink = JsonlTelemetrySink(telemetry_path)
    client = build_mock_openai_client(
        model=args.model,
        prompt_tokens=args.prompt_tokens,
        completion_tokens=args.completion_tokens,
    )
    raw_methods = snapshot_openai_create_methods()
    stabilized_methods: dict[str, object | None] | None = None
    carbon_aware_job = build_carbon_aware_job(
        provider,
        telemetry_sink,
        client,
        model=args.model,
        max_delay_hours=args.max_delay_hours,
    )

    telemetry_count = 0
    baseline_stabilized = False
    carbon_aware_restored = False

    try:
        with ExitStack() as stack:
            stack.enter_context(patch("ecologits.log.logger.warning_once", return_value=None))
            if not args.real_sleep:
                stack.enter_context(
                    patch("llm_eco_tracker.execution.asyncio.sleep", side_effect=mock_sleep)
                )
            if not args.keep_jitter:
                stack.enter_context(
                    patch(
                        "llm_eco_tracker.api.apply_jitter_to_plan",
                        side_effect=identity_schedule_plan,
                    )
                )

            submission_interval = provider.current_interval
            submission_time = submission_interval.starts_at
            baseline_actual_intensity = actual_mapping.get(
                submission_time.strftime("%Y-%m-%dT%H:%MZ"),
                submission_interval.carbon_intensity_gco2eq_per_kwh,
            )

            baseline_energy_kwh = await run_baseline_job(
                client,
                model=args.model,
                call_count=args.call_count,
            )
            stabilized_methods = snapshot_openai_create_methods()
            baseline_stabilized = True
            baseline_summary = summarize_emissions(
                baseline_energy_kwh,
                baseline_actual_intensity,
                baseline_actual_intensity,
            )

            await carbon_aware_job(args.call_count)
            telemetry_record, telemetry_count = read_last_new_jsonl_record(
                telemetry_path,
                telemetry_count,
            )
            if stabilized_methods is None:
                raise RuntimeError("The benchmark did not capture a stabilized SDK baseline.")
            carbon_aware_restored = methods_restored(stabilized_methods)
            if not carbon_aware_restored:
                raise RuntimeError(
                    "OpenAI SDK methods were not restored to the stabilized post-initialization state."
                )

        delay_seconds = float(telemetry_record["schedule_plan"]["execution_delay_seconds"])
        scheduled_time = submission_time + timedelta(seconds=delay_seconds)
        actual_intensity_at_run = lookup_actual_intensity(
            actual_mapping,
            scheduled_time,
            fallback=baseline_actual_intensity,
        )

        carbon_aware_energy_kwh = float(telemetry_record["energy_kwh"])
        telemetry_file_created = telemetry_path.exists()
        model_usage_captured = bool(telemetry_record.get("model_usage"))
        energy_captured = carbon_aware_energy_kwh > 0.0
        if not telemetry_file_created:
            raise RuntimeError("The integration benchmark did not create the telemetry JSONL file.")
        if not energy_captured:
            raise RuntimeError("The integration benchmark captured zero energy for the mocked OpenAI call.")
        if not model_usage_captured:
            raise RuntimeError("The integration benchmark did not record model usage metadata.")

        carbon_aware_summary = summarize_emissions(
            carbon_aware_energy_kwh,
            baseline_actual_intensity,
            actual_intensity_at_run,
        )
        saved_pct = (
            (carbon_aware_summary.saved_gco2eq / baseline_summary.actual_gco2eq) * 100.0
            if baseline_summary.actual_gco2eq > 0
            else 0.0
        )

        summary = {
            "submission_time": submission_time.isoformat(),
            "scheduled_time": scheduled_time.isoformat(),
            "model": args.model,
            "call_count": args.call_count,
            "telemetry_captured": bool(energy_captured and model_usage_captured),
            "telemetry_file_created": telemetry_file_created,
            "model_usage_captured": model_usage_captured,
            "sdk_methods_restored": bool(baseline_stabilized and carbon_aware_restored),
            "baseline_session_stabilized": baseline_stabilized,
            "carbon_aware_session_restored": carbon_aware_restored,
            "ecologits_initialization_changed_sdk": not methods_restored(raw_methods),
            "telemetry_record_count": telemetry_count,
            "baseline_energy_kwh": baseline_energy_kwh,
            "carbon_aware_energy_kwh": carbon_aware_energy_kwh,
            "baseline_actual_intensity": baseline_actual_intensity,
            "actual_intensity_at_run": actual_intensity_at_run,
            "scheduled_delay_h": delay_seconds / 3600.0,
            "baseline_actual_gco2eq": baseline_summary.actual_gco2eq,
            "carbon_aware_actual_gco2eq": carbon_aware_summary.actual_gco2eq,
            "saved_gco2eq": carbon_aware_summary.saved_gco2eq,
            "saved_pct": saved_pct,
            "telemetry_path": str(telemetry_path),
            "summary_path": str(args.summary_path),
            "telemetry_record": telemetry_record,
        }
        write_summary(args.summary_path, summary)
        print_summary(summary)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
