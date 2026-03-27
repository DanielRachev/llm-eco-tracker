import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from llm_eco_tracker.benchmarking import (
    build_trace_schedule_plan,
    contiguous_date_window,
    format_iso_date,
    group_complete_trace_days,
    last_completed_utc_day,
    load_trace_intervals,
    parse_iso_date,
    summarize_trace_days,
)
from llm_eco_tracker.emissions import summarize_emissions

DEFAULT_CSV = BASE_DIR / "tests" / "fixtures" / "benchmark_trace.csv"
DEFAULT_SCENARIO_RESULTS_CSV = BASE_DIR / "scenario_results.csv"
DEFAULT_DAILY_SUMMARY_CSV = BASE_DIR / "daily_summary.csv"
DEFAULT_SUMMARY_JSON = BASE_DIR / "benchmark_summary.json"

DEFAULT_CALL_COUNT = 50
DEFAULT_ENERGY_PER_CALL_KWH = 0.00001
DEFAULT_MAX_DELAY_HOURS = 4.0
DEFAULT_SUBMISSION_STEP = 1

logging.getLogger("llm_eco_tracker").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EcoTracker multi-day trace benchmark")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--scenario-results-csv", type=Path, default=DEFAULT_SCENARIO_RESULTS_CSV)
    parser.add_argument("--daily-summary-csv", type=Path, default=DEFAULT_DAILY_SUMMARY_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--call-count", type=int, default=DEFAULT_CALL_COUNT)
    parser.add_argument("--energy-per-call-kwh", type=float, default=DEFAULT_ENERGY_PER_CALL_KWH)
    parser.add_argument("--max-delay-hours", type=float, default=DEFAULT_MAX_DELAY_HOURS)
    parser.add_argument("--submission-step", type=int, default=DEFAULT_SUBMISSION_STEP)
    parser.add_argument("--start-day", type=str, default=None, help="Inclusive ISO day, for example 2026-03-18")
    parser.add_argument("--end-day", type=str, default=None, help="Inclusive ISO day, for example 2026-03-20")
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument(
        "--last-n-days",
        type=int,
        default=None,
        help="Convenience filter: evaluate the last N eligible complete days in the trace.",
    )
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows were available to write to '{path}'.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def compare_values(left: float, right: float, *, tolerance: float = 1e-12) -> str:
    delta = left - right
    if delta > tolerance:
        return "improved"
    if delta < -tolerance:
        return "worse"
    return "tied"


def summarize_capture_ratio(actual_saved: float, oracle_saved: float) -> float | None:
    if oracle_saved <= 0:
        return None
    return actual_saved / oracle_saved


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return (sorted_values[lower] * (1.0 - fraction)) + (sorted_values[upper] * fraction)


def resolve_day_filters(args: argparse.Namespace) -> tuple[date | None, date | None]:
    start_day = parse_iso_date(args.start_day) if args.start_day else None
    end_day = parse_iso_date(args.end_day) if args.end_day else None
    if start_day and end_day and start_day > end_day:
        raise ValueError("start_day must be on or before end_day.")

    if args.last_n_days is not None:
        if args.last_n_days <= 0:
            raise ValueError("last_n_days must be positive when provided.")
        derived_end_day = end_day or last_completed_utc_day()
        derived_start_day, derived_end_day = contiguous_date_window(
            derived_end_day,
            days=args.last_n_days,
        )
        start_day = start_day or derived_start_day
        end_day = end_day or derived_end_day

    return start_day, end_day


def build_summary_payload(
    scenario_rows: list[dict[str, Any]],
    daily_rows: list[dict[str, Any]],
    *,
    scenario_results_csv: Path,
    daily_summary_csv: Path,
) -> dict[str, Any]:
    baseline_total = sum(float(row["baseline_actual_gco2eq"]) for row in scenario_rows)
    ecotracker_total = sum(float(row["ecotracker_actual_gco2eq"]) for row in scenario_rows)
    oracle_total = sum(float(row["oracle_actual_gco2eq"]) for row in scenario_rows)

    ecotracker_total_reduction_pct = (
        ((baseline_total - ecotracker_total) / baseline_total) * 100.0 if baseline_total > 0 else 0.0
    )
    oracle_total_reduction_pct = (
        ((baseline_total - oracle_total) / baseline_total) * 100.0 if baseline_total > 0 else 0.0
    )
    oracle_capture_ratio = summarize_capture_ratio(
        baseline_total - ecotracker_total,
        baseline_total - oracle_total,
    )

    ecotracker_daily_reductions = [float(row["ecotracker_saved_pct"]) for row in daily_rows]
    oracle_daily_reductions = [float(row["oracle_saved_pct"]) for row in daily_rows]
    ecotracker_delays = [float(row["mean_ecotracker_delay_h"]) for row in daily_rows]
    scenario_delays = [float(row["ecotracker_delay_h"]) for row in scenario_rows]

    improved_scenarios = sum(1 for row in scenario_rows if row["ecotracker_outcome"] == "improved")
    tied_scenarios = sum(1 for row in scenario_rows if row["ecotracker_outcome"] == "tied")
    worse_scenarios = sum(1 for row in scenario_rows if row["ecotracker_outcome"] == "worse")
    improved_days = sum(1 for row in daily_rows if row["ecotracker_day_outcome"] == "improved")
    tied_days = sum(1 for row in daily_rows if row["ecotracker_day_outcome"] == "tied")
    worse_days = sum(1 for row in daily_rows if row["ecotracker_day_outcome"] == "worse")

    return {
        "day_count": len(daily_rows),
        "scenario_count": len(scenario_rows),
        "baseline_total_gco2eq": baseline_total,
        "ecotracker_total_gco2eq": ecotracker_total,
        "oracle_total_gco2eq": oracle_total,
        "ecotracker_total_reduction_pct": ecotracker_total_reduction_pct,
        "oracle_total_reduction_pct": oracle_total_reduction_pct,
        "aggregate_oracle_capture_ratio": oracle_capture_ratio,
        "mean_daily_ecotracker_reduction_pct": statistics.mean(ecotracker_daily_reductions),
        "median_daily_ecotracker_reduction_pct": statistics.median(ecotracker_daily_reductions),
        "mean_daily_oracle_reduction_pct": statistics.mean(oracle_daily_reductions),
        "mean_daily_ecotracker_delay_h": statistics.mean(ecotracker_delays),
        "median_scenario_ecotracker_delay_h": statistics.median(scenario_delays),
        "p95_scenario_ecotracker_delay_h": percentile(scenario_delays, 0.95),
        "improved_scenarios": improved_scenarios,
        "tied_scenarios": tied_scenarios,
        "worse_scenarios": worse_scenarios,
        "improved_days": improved_days,
        "tied_days": tied_days,
        "worse_days": worse_days,
        "scenario_results_csv": str(scenario_results_csv),
        "daily_summary_csv": str(daily_summary_csv),
    }


def print_summary(summary: dict[str, Any], *, max_delay_hours: float, job_energy_kwh: float) -> None:
    print("=" * 72)
    print("MULTI-DAY TRACE BENCHMARK")
    print("=" * 72)
    print(f"Independent days:                {summary['day_count']}")
    print(f"Submission scenarios:            {summary['scenario_count']}")
    print(f"Job energy:                      {job_energy_kwh:.8f} kWh")
    print(f"Max delay budget:                {max_delay_hours:.2f} h")
    print(f"Baseline total:                  {summary['baseline_total_gco2eq']:.6f} gCO2eq")
    print(f"EcoTracker total:                {summary['ecotracker_total_gco2eq']:.6f} gCO2eq")
    print(f"Oracle total:                    {summary['oracle_total_gco2eq']:.6f} gCO2eq")
    print(f"EcoTracker total reduction:      {summary['ecotracker_total_reduction_pct']:.2f}%")
    print(f"Oracle total reduction:          {summary['oracle_total_reduction_pct']:.2f}%")
    aggregate_capture_ratio = summary["aggregate_oracle_capture_ratio"]
    if aggregate_capture_ratio is None:
        print("Aggregate Oracle capture ratio:  n/a")
    else:
        print(f"Aggregate Oracle capture ratio:  {aggregate_capture_ratio * 100.0:.2f}%")
    print(f"Mean daily reduction:            {summary['mean_daily_ecotracker_reduction_pct']:.2f}%")
    print(f"Median daily reduction:          {summary['median_daily_ecotracker_reduction_pct']:.2f}%")
    print(f"Mean daily delay:                {summary['mean_daily_ecotracker_delay_h']:.2f} h")
    print(f"Median scenario delay:           {summary['median_scenario_ecotracker_delay_h']:.2f} h")
    print(f"Scenario outcomes:               {summary['improved_scenarios']} improved, {summary['tied_scenarios']} tied, {summary['worse_scenarios']} worse")
    print(f"Day outcomes:                    {summary['improved_days']} improved, {summary['tied_days']} tied, {summary['worse_days']} worse")
    print(f"Scenario results CSV:            {summary['scenario_results_csv']}")
    print(f"Daily summary CSV:               {summary['daily_summary_csv']}")
    print("=" * 72)


def run_benchmark() -> None:
    args = parse_args()
    if args.call_count <= 0:
        raise ValueError("call_count must be positive.")
    if args.energy_per_call_kwh <= 0:
        raise ValueError("energy_per_call_kwh must be positive.")
    if args.max_delay_hours < 0:
        raise ValueError("max_delay_hours must be non-negative.")
    if args.submission_step <= 0:
        raise ValueError("submission_step must be positive.")

    start_day, end_day = resolve_day_filters(args)
    trace_intervals = load_trace_intervals(args.csv_path)
    trace_days = group_complete_trace_days(
        trace_intervals,
        max_delay_hours=args.max_delay_hours,
    )
    trace_days = summarize_trace_days(
        trace_days,
        start_day=start_day,
        end_day=end_day,
        limit_days=args.limit_days,
    )
    if not trace_days:
        raise RuntimeError("No eligible complete trace days were found for the selected filters.")

    job_energy_kwh = args.call_count * args.energy_per_call_kwh

    scenario_rows: list[dict[str, Any]] = []
    daily_accumulators: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "baseline_total": 0.0,
            "ecotracker_total": 0.0,
            "oracle_total": 0.0,
            "ecotracker_delay_hours": [],
            "oracle_delay_hours": [],
            "scenario_count": 0,
            "ecotracker_improved": 0,
            "ecotracker_tied": 0,
            "ecotracker_worse": 0,
        }
    )

    scenario_index = 0
    for day_index, trace_day in enumerate(trace_days, start=1):
        for submission_slot, submission_offset in enumerate(
            trace_day.submission_offsets[:: args.submission_step],
            start=1,
        ):
            scenario_index += 1
            submission_interval = trace_intervals[submission_offset]
            baseline_result = summarize_emissions(
                job_energy_kwh,
                submission_interval.actual_intensity_gco2eq_per_kwh,
                submission_interval.actual_intensity_gco2eq_per_kwh,
            )

            ecotracker_plan, ecotracker_offset = build_trace_schedule_plan(
                trace_intervals,
                submission_offset,
                args.max_delay_hours,
                intensity_kind="forecast",
            )
            ecotracker_interval = trace_intervals[ecotracker_offset]
            ecotracker_result = summarize_emissions(
                job_energy_kwh,
                submission_interval.actual_intensity_gco2eq_per_kwh,
                ecotracker_interval.actual_intensity_gco2eq_per_kwh,
            )

            oracle_plan, oracle_offset = build_trace_schedule_plan(
                trace_intervals,
                submission_offset,
                args.max_delay_hours,
                intensity_kind="actual",
            )
            oracle_interval = trace_intervals[oracle_offset]
            oracle_result = summarize_emissions(
                job_energy_kwh,
                submission_interval.actual_intensity_gco2eq_per_kwh,
                oracle_interval.actual_intensity_gco2eq_per_kwh,
            )

            ecotracker_outcome = compare_values(
                baseline_result.actual_gco2eq,
                ecotracker_result.actual_gco2eq,
            )
            oracle_outcome = compare_values(
                baseline_result.actual_gco2eq,
                oracle_result.actual_gco2eq,
            )
            oracle_capture_ratio = summarize_capture_ratio(
                ecotracker_result.saved_gco2eq,
                oracle_result.saved_gco2eq,
            )

            day_key = format_iso_date(trace_day.day)
            accumulator = daily_accumulators[day_key]
            accumulator["baseline_total"] += baseline_result.actual_gco2eq
            accumulator["ecotracker_total"] += ecotracker_result.actual_gco2eq
            accumulator["oracle_total"] += oracle_result.actual_gco2eq
            accumulator["ecotracker_delay_hours"].append(
                ecotracker_plan.execution_delay_seconds / 3600.0
            )
            accumulator["oracle_delay_hours"].append(oracle_plan.execution_delay_seconds / 3600.0)
            accumulator["scenario_count"] += 1
            accumulator[f"ecotracker_{ecotracker_outcome}"] += 1

            scenario_rows.append(
                {
                    "scenario": scenario_index,
                    "day_index": day_index,
                    "day": day_key,
                    "submission_slot": submission_slot,
                    "absolute_submission_offset": submission_offset,
                    "submission_time": submission_interval.starts_at.isoformat(),
                    "baseline_execution_time": submission_interval.starts_at.isoformat(),
                    "ecotracker_execution_time": ecotracker_interval.starts_at.isoformat(),
                    "oracle_execution_time": oracle_interval.starts_at.isoformat(),
                    "call_count": args.call_count,
                    "energy_kwh": round(job_energy_kwh, 10),
                    "baseline_forecast_intensity": submission_interval.forecast_intensity_gco2eq_per_kwh,
                    "baseline_actual_intensity": submission_interval.actual_intensity_gco2eq_per_kwh,
                    "ecotracker_selected_forecast_intensity": ecotracker_plan.optimal_intensity_gco2eq_per_kwh,
                    "ecotracker_actual_intensity_at_run": ecotracker_interval.actual_intensity_gco2eq_per_kwh,
                    "oracle_selected_actual_intensity": oracle_interval.actual_intensity_gco2eq_per_kwh,
                    "ecotracker_delay_h": round(ecotracker_plan.execution_delay_seconds / 3600.0, 4),
                    "oracle_delay_h": round(oracle_plan.execution_delay_seconds / 3600.0, 4),
                    "baseline_actual_gco2eq": round(baseline_result.actual_gco2eq, 8),
                    "ecotracker_actual_gco2eq": round(ecotracker_result.actual_gco2eq, 8),
                    "oracle_actual_gco2eq": round(oracle_result.actual_gco2eq, 8),
                    "ecotracker_saved_gco2eq": round(ecotracker_result.saved_gco2eq, 8),
                    "oracle_saved_gco2eq": round(oracle_result.saved_gco2eq, 8),
                    "ecotracker_saved_pct": round(
                        (ecotracker_result.saved_gco2eq / baseline_result.actual_gco2eq) * 100.0
                        if baseline_result.actual_gco2eq > 0
                        else 0.0,
                        6,
                    ),
                    "oracle_saved_pct": round(
                        (oracle_result.saved_gco2eq / baseline_result.actual_gco2eq) * 100.0
                        if baseline_result.actual_gco2eq > 0
                        else 0.0,
                        6,
                    ),
                    "oracle_capture_ratio": (
                        round(oracle_capture_ratio, 6) if oracle_capture_ratio is not None else ""
                    ),
                    "ecotracker_outcome": ecotracker_outcome,
                    "oracle_outcome": oracle_outcome,
                }
            )

    daily_rows: list[dict[str, Any]] = []
    for day_key in sorted(daily_accumulators):
        accumulator = daily_accumulators[day_key]
        baseline_total = accumulator["baseline_total"]
        ecotracker_total = accumulator["ecotracker_total"]
        oracle_total = accumulator["oracle_total"]
        ecotracker_saved_gco2eq = baseline_total - ecotracker_total
        oracle_saved_gco2eq = baseline_total - oracle_total
        ecotracker_saved_pct = (ecotracker_saved_gco2eq / baseline_total) * 100.0 if baseline_total > 0 else 0.0
        oracle_saved_pct = (oracle_saved_gco2eq / baseline_total) * 100.0 if baseline_total > 0 else 0.0
        daily_capture_ratio = summarize_capture_ratio(ecotracker_saved_gco2eq, oracle_saved_gco2eq)

        daily_rows.append(
            {
                "day": day_key,
                "scenario_count": accumulator["scenario_count"],
                "baseline_total_gco2eq": round(baseline_total, 8),
                "ecotracker_total_gco2eq": round(ecotracker_total, 8),
                "oracle_total_gco2eq": round(oracle_total, 8),
                "ecotracker_saved_gco2eq": round(ecotracker_saved_gco2eq, 8),
                "oracle_saved_gco2eq": round(oracle_saved_gco2eq, 8),
                "ecotracker_saved_pct": round(ecotracker_saved_pct, 6),
                "oracle_saved_pct": round(oracle_saved_pct, 6),
                "oracle_capture_ratio": (
                    round(daily_capture_ratio, 6) if daily_capture_ratio is not None else ""
                ),
                "mean_ecotracker_delay_h": round(statistics.mean(accumulator["ecotracker_delay_hours"]), 6),
                "median_ecotracker_delay_h": round(statistics.median(accumulator["ecotracker_delay_hours"]), 6),
                "mean_oracle_delay_h": round(statistics.mean(accumulator["oracle_delay_hours"]), 6),
                "ecotracker_improved_scenarios": accumulator["ecotracker_improved"],
                "ecotracker_tied_scenarios": accumulator["ecotracker_tied"],
                "ecotracker_worse_scenarios": accumulator["ecotracker_worse"],
                "ecotracker_day_outcome": compare_values(baseline_total, ecotracker_total),
                "oracle_day_outcome": compare_values(baseline_total, oracle_total),
            }
        )

    write_rows(args.scenario_results_csv, scenario_rows)
    write_rows(args.daily_summary_csv, daily_rows)

    summary = build_summary_payload(
        scenario_rows,
        daily_rows,
        scenario_results_csv=args.scenario_results_csv,
        daily_summary_csv=args.daily_summary_csv,
    )
    summary.update(
        {
            "job_energy_kwh": job_energy_kwh,
            "call_count": args.call_count,
            "energy_per_call_kwh": args.energy_per_call_kwh,
            "max_delay_hours": args.max_delay_hours,
            "source_csv": str(args.csv_path),
            "start_day": format_iso_date(start_day) if start_day else None,
            "end_day": format_iso_date(end_day) if end_day else None,
        }
    )
    write_json(args.summary_json, summary)
    print_summary(summary, max_delay_hours=args.max_delay_hours, job_energy_kwh=job_energy_kwh)


if __name__ == "__main__":
    run_benchmark()
