import argparse
import asyncio
import csv
import logging
import sys
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from llm_eco_tracker.api import _default_telemetry_runtime, carbon_aware
from llm_eco_tracker.benchmarking import (
    SlidingCsvForecastProvider,
    iter_submission_offsets,
    load_actual_intensity_mapping,
    mock_sleep,
    read_last_new_jsonl_record,
    reset_output_file,
    lookup_actual_intensity,
)
from llm_eco_tracker.emissions import summarize_emissions
from llm_eco_tracker.telemetry import JsonlTelemetrySink

DEFAULT_CSV = BASE_DIR / "tests" / "fixtures" / "mock_forecast.csv"
DEFAULT_RESULTS_CSV = BASE_DIR / "trace_benchmark_results.csv"
DEFAULT_TELEMETRY_PATH = BASE_DIR / "trace_benchmark_telemetry.jsonl"

DEFAULT_CALL_COUNT = 50
DEFAULT_SWEEP_LIMIT = 48
DEFAULT_MAX_DELAY_HOURS = 4
DEFAULT_ENERGY_PER_CALL_KWH = 0.00001

# Keep benchmark output readable.
logging.getLogger("llm_eco_tracker").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EcoTracker trace-driven scheduler benchmark")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--telemetry-path", type=Path, default=DEFAULT_TELEMETRY_PATH)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=DEFAULT_SWEEP_LIMIT)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--call-count", type=int, default=DEFAULT_CALL_COUNT)
    parser.add_argument("--max-delay-hours", type=float, default=DEFAULT_MAX_DELAY_HOURS)
    parser.add_argument("--energy-per-call-kwh", type=float, default=DEFAULT_ENERGY_PER_CALL_KWH)
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


async def simulate_llm_job(call_count: int, energy_per_call_kwh: float) -> None:
    for _ in range(call_count):
        _default_telemetry_runtime._record_energy(energy_per_call_kwh)


async def run_baseline_job(call_count: int, energy_per_call_kwh: float) -> float:
    with _default_telemetry_runtime.session() as session:
        await simulate_llm_job(call_count, energy_per_call_kwh)
        return session.energy_kwh


def build_carbon_aware_job(
    provider: SlidingCsvForecastProvider,
    telemetry_sink: JsonlTelemetrySink,
    *,
    max_delay_hours: float,
):
    @carbon_aware(
        max_delay_hours=max_delay_hours,
        forecast_provider=provider,
        telemetry_sink=telemetry_sink,
    )
    async def run_job(call_count: int, energy_per_call_kwh: float) -> None:
        await simulate_llm_job(call_count, energy_per_call_kwh)

    return run_job


def identity_schedule_plan(schedule_plan, *args, **kwargs):
    del args, kwargs
    return schedule_plan


def write_results(results_csv: Path, results: list[dict[str, object]]) -> None:
    if not results:
        raise RuntimeError("No benchmark results were produced.")

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def print_summary(
    results: list[dict[str, object]],
    *,
    results_csv: Path,
    telemetry_path: Path,
    call_count: int,
    max_delay_hours: float,
    fast_forwarded: bool,
    jitter_enabled: bool,
) -> None:
    baseline_total = sum(float(row["baseline_actual_gco2eq"]) for row in results)
    carbon_aware_total = sum(float(row["carbon_aware_actual_gco2eq"]) for row in results)
    total_reduction_pct = (
        ((baseline_total - carbon_aware_total) / baseline_total) * 100.0 if baseline_total > 0 else 0.0
    )
    average_reduction_pct = (
        sum(float(row["saved_pct"]) for row in results) / len(results) if results else 0.0
    )
    average_delay_h = (
        sum(float(row["scheduled_delay_h"]) for row in results) / len(results) if results else 0.0
    )
    immediate_runs = sum(1 for row in results if float(row["scheduled_delay_h"]) == 0.0)

    print("=" * 72)
    print("TRACE-DRIVEN SCHEDULER BENCHMARK")
    print("=" * 72)
    print(f"Scenarios swept:         {len(results)} submission times")
    print(f"Calls per job:           {call_count}")
    print(f"Max delay budget:        {max_delay_hours:.2f} h")
    print(f"Fast-forward sleep:      {fast_forwarded}")
    print(f"Deterministic jitter:    {not jitter_enabled}")
    print(f"Immediate executions:    {immediate_runs}")
    print(f"Average delay:           {average_delay_h:.2f} h")
    print(f"Average reduction:       {average_reduction_pct:.2f}%")
    print(f"Weighted total reduction:{total_reduction_pct:.2f}%")
    print(f"Baseline total:          {baseline_total:.6f} gCO2eq")
    print(f"Carbon-aware total:      {carbon_aware_total:.6f} gCO2eq")
    print(f"Saved total:             {baseline_total - carbon_aware_total:.6f} gCO2eq")
    print(f"Results CSV:             {results_csv}")
    print(f"Telemetry JSONL:         {telemetry_path}")
    print("=" * 72)


async def run_benchmark() -> None:
    args = parse_args()

    provider = SlidingCsvForecastProvider(args.csv_path)
    actual_mapping = load_actual_intensity_mapping(args.csv_path)
    offsets = iter_submission_offsets(
        provider.interval_count,
        start_offset=args.start_offset,
        limit=args.limit,
        step=args.step,
    )
    telemetry_path = reset_output_file(args.telemetry_path)
    telemetry_sink = JsonlTelemetrySink(telemetry_path)
    carbon_aware_job = build_carbon_aware_job(
        provider,
        telemetry_sink,
        max_delay_hours=args.max_delay_hours,
    )

    results: list[dict[str, object]] = []
    telemetry_count = 0

    with ExitStack() as stack:
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

        for scenario_index, start_offset in enumerate(offsets, start=1):
            provider.set_start_offset(start_offset)
            submission_interval = provider.current_interval
            submission_time = submission_interval.starts_at
            baseline_actual_intensity = actual_mapping.get(
                submission_time.strftime("%Y-%m-%dT%H:%MZ"),
                submission_interval.carbon_intensity_gco2eq_per_kwh,
            )

            baseline_energy_kwh = await run_baseline_job(
                args.call_count,
                args.energy_per_call_kwh,
            )
            baseline_summary = summarize_emissions(
                baseline_energy_kwh,
                baseline_actual_intensity,
                baseline_actual_intensity,
            )

            await carbon_aware_job(args.call_count, args.energy_per_call_kwh)
            telemetry_record, telemetry_count = read_last_new_jsonl_record(
                telemetry_path,
                telemetry_count,
            )

            delay_seconds = float(telemetry_record["schedule_plan"]["execution_delay_seconds"])
            scheduled_time = submission_time + timedelta(seconds=delay_seconds)
            actual_intensity_at_run = lookup_actual_intensity(
                actual_mapping,
                scheduled_time,
                fallback=baseline_actual_intensity,
            )

            carbon_aware_energy_kwh = float(telemetry_record["energy_kwh"])
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

            results.append(
                {
                    "scenario": scenario_index,
                    "start_offset": start_offset,
                    "submission_time": submission_time.isoformat(),
                    "scheduled_time": scheduled_time.isoformat(),
                    "call_count": args.call_count,
                    "energy_kwh": round(carbon_aware_energy_kwh, 10),
                    "baseline_forecast_intensity": float(
                        telemetry_record["schedule_plan"]["baseline_intensity_gco2eq_per_kwh"]
                    ),
                    "optimal_forecast_intensity": float(
                        telemetry_record["schedule_plan"]["optimal_intensity_gco2eq_per_kwh"]
                    ),
                    "baseline_actual_intensity": baseline_actual_intensity,
                    "actual_intensity_at_run": actual_intensity_at_run,
                    "scheduled_delay_h": round(delay_seconds / 3600.0, 4),
                    "baseline_actual_gco2eq": round(baseline_summary.actual_gco2eq, 8),
                    "carbon_aware_actual_gco2eq": round(carbon_aware_summary.actual_gco2eq, 8),
                    "saved_gco2eq": round(carbon_aware_summary.saved_gco2eq, 8),
                    "saved_pct": round(saved_pct, 6),
                }
            )

    write_results(args.results_csv, results)
    print_summary(
        results,
        results_csv=args.results_csv,
        telemetry_path=telemetry_path,
        call_count=args.call_count,
        max_delay_hours=args.max_delay_hours,
        fast_forwarded=not args.real_sleep,
        jitter_enabled=args.keep_jitter,
    )


if __name__ == "__main__":
    asyncio.run(run_benchmark())
