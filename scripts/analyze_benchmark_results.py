import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

DEFAULT_DAILY_SUMMARY_CSV = BASE_DIR / "daily_summary.csv"
DEFAULT_SCENARIO_RESULTS_CSV = BASE_DIR / "scenario_results.csv"
DEFAULT_ANALYSIS_JSON = BASE_DIR / "benchmark_analysis.json"
DEFAULT_ANALYSIS_MD = BASE_DIR / "benchmark_analysis.md"
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_SEED = 1729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical analysis for EcoTracker benchmark results")
    parser.add_argument("--daily-summary-csv", type=Path, default=DEFAULT_DAILY_SUMMARY_CSV)
    parser.add_argument("--scenario-results-csv", type=Path, default=DEFAULT_SCENARIO_RESULTS_CSV)
    parser.add_argument("--analysis-json", type=Path, default=DEFAULT_ANALYSIS_JSON)
    parser.add_argument("--analysis-markdown", type=Path, default=DEFAULT_ANALYSIS_MD)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def bootstrap_mean_ci(
    values: list[float],
    *,
    resamples: int,
    seed: int,
    lower_q: float = 0.025,
    upper_q: float = 0.975,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        sampled = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(statistics.mean(sampled))
    return percentile(means, lower_q), percentile(means, upper_q)


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def run_wilcoxon(baseline: list[float], ecotracker: list[float]) -> dict[str, Any]:
    if wilcoxon is None:
        return {
            "available": False,
            "test": "wilcoxon_signed_rank",
            "reason": "scipy_not_installed",
        }

    statistic, p_value = wilcoxon(
        baseline,
        ecotracker,
        alternative="greater",
        zero_method="pratt",
        method="auto",
    )
    return {
        "available": True,
        "test": "wilcoxon_signed_rank",
        "alternative": "baseline > ecotracker",
        "statistic": float(statistic),
        "p_value": float(p_value),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# Benchmark Analysis",
        "",
        f"- Independent days: {analysis['day_count']}",
        f"- Submission scenarios: {analysis['scenario_count']}",
        f"- Aggregate EcoTracker reduction: {analysis['aggregate_ecotracker_reduction_pct']:.2f}%",
        f"- Aggregate Oracle reduction: {analysis['aggregate_oracle_reduction_pct']:.2f}%",
        (
            "- Aggregate Oracle capture ratio: "
            + (
                "n/a"
                if analysis["aggregate_oracle_capture_ratio"] is None
                else f"{analysis['aggregate_oracle_capture_ratio'] * 100.0:.2f}%"
            )
        ),
        f"- Mean daily EcoTracker reduction: {analysis['mean_daily_ecotracker_reduction_pct']:.2f}%",
        f"- Median daily EcoTracker reduction: {analysis['median_daily_ecotracker_reduction_pct']:.2f}%",
        (
            "- 95% bootstrap CI for mean daily EcoTracker reduction: "
            f"[{analysis['bootstrap_mean_daily_reduction_ci'][0]:.2f}%, "
            f"{analysis['bootstrap_mean_daily_reduction_ci'][1]:.2f}%]"
        ),
        (
            f"- Scenario outcomes: {analysis['scenario_outcomes']['improved']} improved, "
            f"{analysis['scenario_outcomes']['tied']} tied, "
            f"{analysis['scenario_outcomes']['worse']} worse"
        ),
        (
            f"- Day outcomes: {analysis['day_outcomes']['improved']} improved, "
            f"{analysis['day_outcomes']['tied']} tied, "
            f"{analysis['day_outcomes']['worse']} worse"
        ),
        f"- Mean EcoTracker delay: {analysis['mean_scenario_delay_h']:.2f} h",
        "",
        "## Wilcoxon Signed-Rank Test",
        "",
    ]
    wilcoxon_payload = analysis["wilcoxon_signed_rank"]
    if wilcoxon_payload["available"]:
        lines.extend(
            [
                f"- Statistic: {wilcoxon_payload['statistic']:.4f}",
                f"- p-value: {wilcoxon_payload['p_value']:.6f}",
            ]
        )
    else:
        lines.append(f"- Not available: {wilcoxon_payload['reason']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_analysis() -> None:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive.")

    daily_rows = load_csv_rows(args.daily_summary_csv)
    scenario_rows = load_csv_rows(args.scenario_results_csv)
    if not daily_rows or not scenario_rows:
        raise RuntimeError("The benchmark CSV inputs were empty.")

    daily_baseline = [float(row["baseline_total_gco2eq"]) for row in daily_rows]
    daily_ecotracker = [float(row["ecotracker_total_gco2eq"]) for row in daily_rows]
    daily_oracle = [float(row["oracle_total_gco2eq"]) for row in daily_rows]
    daily_ecotracker_pct = [float(row["ecotracker_saved_pct"]) for row in daily_rows]
    daily_oracle_pct = [float(row["oracle_saved_pct"]) for row in daily_rows]
    daily_capture_ratios = [
        float(row["oracle_capture_ratio"])
        for row in daily_rows
        if row["oracle_capture_ratio"] not in {"", None}
    ]

    scenario_ecotracker_pct = [float(row["ecotracker_saved_pct"]) for row in scenario_rows]
    scenario_delays = [float(row["ecotracker_delay_h"]) for row in scenario_rows]
    scenario_capture_ratios = [
        float(row["oracle_capture_ratio"])
        for row in scenario_rows
        if row["oracle_capture_ratio"] not in {"", None}
    ]

    baseline_total = sum(daily_baseline)
    ecotracker_total = sum(daily_ecotracker)
    oracle_total = sum(daily_oracle)
    ecotracker_abs_saved = baseline_total - ecotracker_total
    oracle_abs_saved = baseline_total - oracle_total

    bootstrap_ci = bootstrap_mean_ci(
        daily_ecotracker_pct,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    wilcoxon_payload = run_wilcoxon(daily_baseline, daily_ecotracker)

    analysis = {
        "day_count": len(daily_rows),
        "scenario_count": len(scenario_rows),
        "baseline_total_gco2eq": baseline_total,
        "ecotracker_total_gco2eq": ecotracker_total,
        "oracle_total_gco2eq": oracle_total,
        "ecotracker_abs_saved_gco2eq": ecotracker_abs_saved,
        "oracle_abs_saved_gco2eq": oracle_abs_saved,
        "aggregate_ecotracker_reduction_pct": (ecotracker_abs_saved / baseline_total) * 100.0
        if baseline_total > 0
        else 0.0,
        "aggregate_oracle_reduction_pct": (oracle_abs_saved / baseline_total) * 100.0
        if baseline_total > 0
        else 0.0,
        "aggregate_oracle_capture_ratio": (
            ecotracker_abs_saved / oracle_abs_saved if oracle_abs_saved > 0 else None
        ),
        "mean_daily_ecotracker_reduction_pct": safe_mean(daily_ecotracker_pct),
        "median_daily_ecotracker_reduction_pct": safe_median(daily_ecotracker_pct),
        "min_daily_ecotracker_reduction_pct": min(daily_ecotracker_pct),
        "max_daily_ecotracker_reduction_pct": max(daily_ecotracker_pct),
        "mean_daily_oracle_reduction_pct": safe_mean(daily_oracle_pct),
        "mean_daily_oracle_capture_ratio": safe_mean(daily_capture_ratios),
        "bootstrap_mean_daily_reduction_ci": bootstrap_ci,
        "mean_scenario_reduction_pct": safe_mean(scenario_ecotracker_pct),
        "median_scenario_reduction_pct": safe_median(scenario_ecotracker_pct),
        "mean_scenario_delay_h": safe_mean(scenario_delays),
        "median_scenario_delay_h": safe_median(scenario_delays),
        "p95_scenario_delay_h": percentile(scenario_delays, 0.95),
        "mean_scenario_oracle_capture_ratio": safe_mean(scenario_capture_ratios),
        "scenario_outcomes": {
            "improved": sum(1 for row in scenario_rows if row["ecotracker_outcome"] == "improved"),
            "tied": sum(1 for row in scenario_rows if row["ecotracker_outcome"] == "tied"),
            "worse": sum(1 for row in scenario_rows if row["ecotracker_outcome"] == "worse"),
        },
        "day_outcomes": {
            "improved": sum(1 for row in daily_rows if row["ecotracker_day_outcome"] == "improved"),
            "tied": sum(1 for row in daily_rows if row["ecotracker_day_outcome"] == "tied"),
            "worse": sum(1 for row in daily_rows if row["ecotracker_day_outcome"] == "worse"),
        },
        "wilcoxon_signed_rank": wilcoxon_payload,
        "daily_summary_csv": str(args.daily_summary_csv),
        "scenario_results_csv": str(args.scenario_results_csv),
    }

    write_json(args.analysis_json, analysis)
    write_markdown(args.analysis_markdown, analysis)

    print("=" * 72)
    print("BENCHMARK ANALYSIS")
    print("=" * 72)
    print(f"Independent days:                {analysis['day_count']}")
    print(f"Submission scenarios:            {analysis['scenario_count']}")
    print(f"Aggregate EcoTracker reduction:  {analysis['aggregate_ecotracker_reduction_pct']:.2f}%")
    print(f"Aggregate Oracle reduction:      {analysis['aggregate_oracle_reduction_pct']:.2f}%")
    print(
        "95% bootstrap CI (mean daily):  "
        f"[{analysis['bootstrap_mean_daily_reduction_ci'][0]:.2f}%, "
        f"{analysis['bootstrap_mean_daily_reduction_ci'][1]:.2f}%]"
    )
    if wilcoxon_payload["available"]:
        print(f"Wilcoxon p-value:                {wilcoxon_payload['p_value']:.6f}")
    else:
        print("Wilcoxon p-value:                n/a (scipy not installed)")
    print(f"Analysis JSON:                   {args.analysis_json}")
    print(f"Analysis Markdown:               {args.analysis_markdown}")
    print("=" * 72)


if __name__ == "__main__":
    run_analysis()
