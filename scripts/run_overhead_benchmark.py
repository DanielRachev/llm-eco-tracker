import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from llm_eco_tracker.api import carbon_aware
from llm_eco_tracker.telemetry import NoOpTelemetrySink

DEFAULT_RUNS_CSV = BASE_DIR / "overhead_benchmark_runs.csv"
DEFAULT_SUMMARY_JSON = BASE_DIR / "overhead_benchmark_summary.json"
DEFAULT_BATCH_ITERATIONS = 1000
DEFAULT_BATCH_REPETITIONS = 30
DEFAULT_WARMUP_BATCHES = 3

logging.getLogger("llm_eco_tracker").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EcoTracker decorator overhead benchmark")
    parser.add_argument("--runs-csv", type=Path, default=DEFAULT_RUNS_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--batch-iterations", type=int, default=DEFAULT_BATCH_ITERATIONS)
    parser.add_argument("--batch-repetitions", type=int, default=DEFAULT_BATCH_REPETITIONS)
    parser.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES)
    return parser.parse_args()


def dummy_workload() -> int:
    total = 0
    for value in range(1, 12):
        total += value * value
    return total


@carbon_aware(max_delay_hours=0, telemetry_sink=NoOpTelemetrySink())
def decorated_dummy_workload() -> int:
    return dummy_workload()


def benchmark_batch(callable_under_test, *, iterations: int) -> tuple[int, int]:
    checksum = 0
    start_ns = time.perf_counter_ns()
    for _ in range(iterations):
        checksum += callable_under_test()
    elapsed_ns = time.perf_counter_ns() - start_ns
    return elapsed_ns, checksum


def write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, payload: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return (sorted_values[lower] * (1.0 - fraction)) + (sorted_values[upper] * fraction)


def run_benchmark() -> None:
    args = parse_args()
    if args.batch_iterations <= 0:
        raise ValueError("batch_iterations must be positive.")
    if args.batch_repetitions <= 0:
        raise ValueError("batch_repetitions must be positive.")
    if args.warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative.")

    for _ in range(args.warmup_batches):
        benchmark_batch(dummy_workload, iterations=args.batch_iterations)
        benchmark_batch(decorated_dummy_workload, iterations=args.batch_iterations)

    rows: list[dict[str, float | int]] = []
    per_call_overheads_ms: list[float] = []
    baseline_per_call_ms: list[float] = []
    decorated_per_call_ms: list[float] = []

    for repetition in range(1, args.batch_repetitions + 1):
        baseline_elapsed_ns, baseline_checksum = benchmark_batch(
            dummy_workload,
            iterations=args.batch_iterations,
        )
        decorated_elapsed_ns, decorated_checksum = benchmark_batch(
            decorated_dummy_workload,
            iterations=args.batch_iterations,
        )
        if baseline_checksum != decorated_checksum:
            raise RuntimeError("The decorated workload changed the dummy function's output.")

        baseline_ms = baseline_elapsed_ns / 1_000_000.0
        decorated_ms = decorated_elapsed_ns / 1_000_000.0
        overhead_per_call_ms = (decorated_ms - baseline_ms) / args.batch_iterations

        baseline_per_call_ms.append(baseline_ms / args.batch_iterations)
        decorated_per_call_ms.append(decorated_ms / args.batch_iterations)
        per_call_overheads_ms.append(overhead_per_call_ms)

        rows.append(
            {
                "repetition": repetition,
                "batch_iterations": args.batch_iterations,
                "baseline_batch_ms": round(baseline_ms, 6),
                "decorated_batch_ms": round(decorated_ms, 6),
                "baseline_per_call_ms": round(baseline_ms / args.batch_iterations, 9),
                "decorated_per_call_ms": round(decorated_ms / args.batch_iterations, 9),
                "added_overhead_per_call_ms": round(overhead_per_call_ms, 9),
            }
        )

    summary = {
        "batch_iterations": args.batch_iterations,
        "batch_repetitions": args.batch_repetitions,
        "warmup_batches": args.warmup_batches,
        "median_baseline_per_call_ms": statistics.median(baseline_per_call_ms),
        "median_decorated_per_call_ms": statistics.median(decorated_per_call_ms),
        "median_added_overhead_per_call_ms": statistics.median(per_call_overheads_ms),
        "p95_added_overhead_per_call_ms": percentile(per_call_overheads_ms, 0.95),
        "mean_added_overhead_per_call_ms": statistics.mean(per_call_overheads_ms),
        "max_added_overhead_per_call_ms": max(per_call_overheads_ms),
        "runs_csv": str(args.runs_csv),
    }

    write_rows(args.runs_csv, rows)
    write_summary(args.summary_json, summary)

    print("=" * 72)
    print("DECORATOR OVERHEAD BENCHMARK")
    print("=" * 72)
    print(f"Batch repetitions:               {args.batch_repetitions}")
    print(f"Iterations per batch:            {args.batch_iterations}")
    print(f"Median baseline per-call time:   {summary['median_baseline_per_call_ms']:.6f} ms")
    print(f"Median decorated per-call time:  {summary['median_decorated_per_call_ms']:.6f} ms")
    print(f"Median added overhead:           {summary['median_added_overhead_per_call_ms']:.6f} ms")
    print(f"P95 added overhead:              {summary['p95_added_overhead_per_call_ms']:.6f} ms")
    print(f"Runs CSV:                        {args.runs_csv}")
    print(f"Summary JSON:                    {args.summary_json}")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
