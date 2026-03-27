import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required to generate paper figures. "
        "Install the dependencies from requirements.txt first."
    ) from exc


DEFAULT_RESULTS_CSV = BASE_DIR / "trace_benchmark_results.csv"
DEFAULT_FORECAST_CSV = BASE_DIR / "tests" / "fixtures" / "mock_forecast.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "paper_figures"
DEFAULT_FORMATS = ("png", "pdf")
DEFAULT_DPI = 300
DEFAULT_CURVE_HOURS = 24

COLOR_GRID = "#0072B2"
COLOR_BASELINE = "#D55E00"
COLOR_AWARE = "#009E73"
COLOR_FILL = "#56B4E9"
COLOR_NEUTRAL = "#4D4D4D"
COLOR_GRIDLINES = "#D9D9D9"


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    scenario: int
    start_offset: int
    submission_time: datetime
    scheduled_time: datetime
    call_count: int
    energy_kwh: float
    baseline_forecast_intensity: float
    optimal_forecast_intensity: float
    baseline_actual_intensity: float
    actual_intensity_at_run: float
    scheduled_delay_h: float
    baseline_actual_gco2eq: float
    carbon_aware_actual_gco2eq: float
    saved_gco2eq: float
    saved_pct: float


@dataclass(frozen=True, slots=True)
class ForecastPoint:
    starts_at: datetime
    ends_at: datetime
    forecast_intensity: float
    actual_intensity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready benchmark figures")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--forecast-csv", type=Path, default=DEFAULT_FORECAST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--curve-hours", type=int, default=DEFAULT_CURVE_HOURS)
    parser.add_argument(
        "--curve-point-mode",
        choices=("shifted-only", "all"),
        default="shifted-only",
        help="Which benchmark scenarios to overlay on the intensity curve.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
        help="Image formats to write, for example: png pdf",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.color": COLOR_GRIDLINES,
            "grid.alpha": 0.8,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def ensure_inputs_exist(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing required input file(s): {joined}. "
            "Run the benchmark first if the results CSV has not been generated yet."
        )


def parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value)


def load_benchmark_rows(path: Path) -> list[BenchmarkRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            BenchmarkRow(
                scenario=int(row["scenario"]),
                start_offset=int(row["start_offset"]),
                submission_time=parse_timestamp(row["submission_time"]),
                scheduled_time=parse_timestamp(row["scheduled_time"]),
                call_count=int(row["call_count"]),
                energy_kwh=float(row["energy_kwh"]),
                baseline_forecast_intensity=float(row["baseline_forecast_intensity"]),
                optimal_forecast_intensity=float(row["optimal_forecast_intensity"]),
                baseline_actual_intensity=float(row["baseline_actual_intensity"]),
                actual_intensity_at_run=float(row["actual_intensity_at_run"]),
                scheduled_delay_h=float(row["scheduled_delay_h"]),
                baseline_actual_gco2eq=float(row["baseline_actual_gco2eq"]),
                carbon_aware_actual_gco2eq=float(row["carbon_aware_actual_gco2eq"]),
                saved_gco2eq=float(row["saved_gco2eq"]),
                saved_pct=float(row["saved_pct"]),
            )
            for row in reader
        ]
    if not rows:
        raise SystemExit(f"No benchmark rows were found in '{path}'.")
    return rows


def load_forecast_points(path: Path) -> list[ForecastPoint]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        points = [
            ForecastPoint(
                starts_at=parse_timestamp(row["from"]),
                ends_at=parse_timestamp(row["to"]),
                forecast_intensity=float(row["intensity_forecast"]),
                actual_intensity=float(row["intensity_actual"]),
            )
            for row in reader
        ]
    if not points:
        raise SystemExit(f"No forecast rows were found in '{path}'.")
    return points


def save_figure(fig, output_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for fmt in formats:
        output_path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {"dpi": dpi} if fmt.lower() != "pdf" else {}
        fig.savefig(output_path, **save_kwargs)
        written_paths.append(output_path)
    plt.close(fig)
    return written_paths


def build_curve_window(
    rows: list[BenchmarkRow],
    *,
    requested_hours: int,
) -> tuple[datetime, datetime]:
    window_start = min(row.submission_time for row in rows)
    submission_window_end = window_start + timedelta(hours=requested_hours)
    final_execution_time = max(
        max(row.submission_time, row.scheduled_time) for row in rows
    ) + timedelta(minutes=30)
    return window_start, max(submission_window_end, final_execution_time)


def filter_forecast_window(
    forecast_points: list[ForecastPoint],
    *,
    window_start: datetime,
    window_end: datetime,
) -> list[ForecastPoint]:
    window_points = [
        point
        for point in forecast_points
        if point.starts_at >= window_start and point.starts_at <= window_end
    ]
    if not window_points:
        raise SystemExit("No forecast points overlap with the requested plotting window.")
    return window_points


def apply_time_axis(ax, *, window_start: datetime, window_end: datetime) -> None:
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(window_start, window_end)


def plot_intensity_shift_curve(
    rows: list[BenchmarkRow],
    forecast_points: list[ForecastPoint],
    *,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    curve_hours: int,
    point_mode: str,
) -> list[Path]:
    plotted_rows = [row for row in rows if row.scheduled_delay_h > 0] if point_mode == "shifted-only" else rows
    if not plotted_rows:
        plotted_rows = rows

    window_start, window_end = build_curve_window(rows, requested_hours=curve_hours)
    window_points = filter_forecast_window(
        forecast_points,
        window_start=window_start,
        window_end=window_end,
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    ax.plot(
        [point.starts_at for point in window_points],
        [point.actual_intensity for point in window_points],
        color=COLOR_GRID,
        linewidth=2.4,
        label="Grid carbon intensity (actual)",
        zorder=2,
    )
    ax.scatter(
        [row.submission_time for row in plotted_rows],
        [row.baseline_actual_intensity for row in plotted_rows],
        color=COLOR_BASELINE,
        marker="o",
        s=52,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.5,
        label="Baseline execution",
        zorder=4,
    )
    ax.scatter(
        [row.scheduled_time for row in plotted_rows],
        [row.actual_intensity_at_run for row in plotted_rows],
        color=COLOR_AWARE,
        marker="^",
        s=68,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
        label="Carbon-aware execution",
        zorder=5,
    )

    apply_time_axis(ax, window_start=window_start, window_end=window_end)
    ax.set_title("Figure 1. Grid Carbon Intensity and Carbon-Aware Temporal Shifting")
    ax.set_xlabel("Execution time")
    ax.set_ylabel("Carbon intensity (gCO2eq/kWh)")
    ax.legend(loc="upper left", ncol=1)

    summary_text = (
        f"Shifted scenarios: {sum(1 for row in rows if row.scheduled_delay_h > 0)} / {len(rows)}\n"
        f"Average delay: {sum(row.scheduled_delay_h for row in rows) / len(rows):.2f} h"
    )
    ax.text(
        0.99,
        0.03,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=COLOR_NEUTRAL,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.4"},
    )

    return save_figure(fig, output_dir, "figure_1_curve", formats, dpi)


def plot_total_emissions_bar(
    rows: list[BenchmarkRow],
    *,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    baseline_total = sum(row.baseline_actual_gco2eq for row in rows)
    carbon_aware_total = sum(row.carbon_aware_actual_gco2eq for row in rows)
    reduction_pct = (
        ((baseline_total - carbon_aware_total) / baseline_total) * 100.0 if baseline_total > 0 else 0.0
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    labels = ["Baseline", "Carbon-aware"]
    values = [baseline_total, carbon_aware_total]
    colors = [COLOR_BASELINE, COLOR_AWARE]
    bars = ax.bar(labels, values, color=colors, width=0.58, zorder=3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(values) * 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=COLOR_NEUTRAL,
        )

    ax.set_title("Figure 2. Total Emissions Across the Benchmark Sweep")
    ax.set_ylabel("Total emitted carbon (gCO2eq)")
    ax.set_axisbelow(True)
    ax.text(
        0.98,
        0.95,
        f"Reduction: {reduction_pct:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color=COLOR_NEUTRAL,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.35"},
    )

    return save_figure(fig, output_dir, "figure_2_total_emissions", formats, dpi)


def plot_emissions_profile(
    rows: list[BenchmarkRow],
    *,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    sorted_rows = sorted(rows, key=lambda row: row.submission_time)
    x_values = [row.submission_time for row in sorted_rows]
    baseline_values = [row.baseline_actual_gco2eq for row in sorted_rows]
    aware_values = [row.carbon_aware_actual_gco2eq for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    ax.plot(
        x_values,
        baseline_values,
        color=COLOR_BASELINE,
        linewidth=2.1,
        marker="o",
        markersize=4,
        label="Baseline",
        zorder=4,
    )
    ax.plot(
        x_values,
        aware_values,
        color=COLOR_AWARE,
        linewidth=2.1,
        marker="^",
        markersize=4,
        label="Carbon-aware",
        zorder=5,
    )
    ax.fill_between(
        x_values,
        aware_values,
        baseline_values,
        where=[baseline >= aware for baseline, aware in zip(baseline_values, aware_values)],
        color=COLOR_FILL,
        alpha=0.18,
        interpolate=True,
        label="Realized savings",
        zorder=2,
    )

    apply_time_axis(
        ax,
        window_start=min(x_values),
        window_end=max(x_values) + timedelta(minutes=30),
    )
    ax.set_title("Figure 3. Scenario-Level Emissions Across Submission Times")
    ax.set_xlabel("Submission time")
    ax.set_ylabel("Emitted carbon per job (gCO2eq)")
    ax.legend(loc="upper left", ncol=3)

    average_reduction_pct = sum(row.saved_pct for row in rows) / len(rows)
    ax.text(
        0.99,
        0.03,
        f"Average reduction: {average_reduction_pct:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=COLOR_NEUTRAL,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.35"},
    )

    return save_figure(fig, output_dir, "figure_3_emissions_profile", formats, dpi)


def build_manifest(
    rows: list[BenchmarkRow],
    written_paths: list[Path],
    *,
    output_dir: Path,
) -> Path:
    baseline_total = sum(row.baseline_actual_gco2eq for row in rows)
    carbon_aware_total = sum(row.carbon_aware_actual_gco2eq for row in rows)
    manifest = {
        "scenario_count": len(rows),
        "shifted_scenarios": sum(1 for row in rows if row.scheduled_delay_h > 0),
        "average_delay_h": sum(row.scheduled_delay_h for row in rows) / len(rows),
        "average_reduction_pct": sum(row.saved_pct for row in rows) / len(rows),
        "weighted_total_reduction_pct": (
            ((baseline_total - carbon_aware_total) / baseline_total) * 100.0 if baseline_total > 0 else 0.0
        ),
        "baseline_total_gco2eq": baseline_total,
        "carbon_aware_total_gco2eq": carbon_aware_total,
        "written_files": [str(path) for path in written_paths],
    }
    manifest_path = output_dir / "figure_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    ensure_inputs_exist(args.results_csv, args.forecast_csv)

    rows = load_benchmark_rows(args.results_csv)
    forecast_points = load_forecast_points(args.forecast_csv)

    written_paths: list[Path] = []
    written_paths.extend(
        plot_intensity_shift_curve(
            rows,
            forecast_points,
            output_dir=args.output_dir,
            formats=args.formats,
            dpi=args.dpi,
            curve_hours=args.curve_hours,
            point_mode=args.curve_point_mode,
        )
    )
    written_paths.extend(
        plot_total_emissions_bar(
            rows,
            output_dir=args.output_dir,
            formats=args.formats,
            dpi=args.dpi,
        )
    )
    written_paths.extend(
        plot_emissions_profile(
            rows,
            output_dir=args.output_dir,
            formats=args.formats,
            dpi=args.dpi,
        )
    )
    manifest_path = build_manifest(rows, written_paths, output_dir=args.output_dir)

    print("=" * 72)
    print("PAPER FIGURES GENERATED")
    print("=" * 72)
    for path in written_paths:
        print(path)
    print(manifest_path)
    print("=" * 72)


if __name__ == "__main__":
    main()
