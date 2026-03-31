import argparse
import csv
import json
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
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


DEFAULT_SCENARIO_RESULTS_CSV = BASE_DIR / "scenario_results.csv"
DEFAULT_DAILY_SUMMARY_CSV = BASE_DIR / "daily_summary.csv"
DEFAULT_FORECAST_CSV = BASE_DIR / "tests" / "fixtures" / "benchmark_trace.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "paper_figures"
DEFAULT_FORMATS = ("png", "pdf")
DEFAULT_DPI = 300

COLOR_GRID = "#4D4D4D"
COLOR_BASELINE = "#D55E00"
COLOR_ECOTRACKER = "#009E73"
COLOR_ORACLE = "#0072B2"
COLOR_FILL = "#56B4E9"
COLOR_GRIDLINES = "#D9D9D9"
CAPTION_BBOX = {"facecolor": "white", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.35"}


@dataclass(frozen=True, slots=True)
class ScenarioRow:
    day: date
    submission_slot: int
    submission_time: datetime
    baseline_execution_time: datetime
    ecotracker_execution_time: datetime
    oracle_execution_time: datetime
    baseline_actual_intensity: float
    ecotracker_actual_intensity_at_run: float
    oracle_actual_intensity_at_run: float
    baseline_actual_gco2eq: float
    ecotracker_actual_gco2eq: float
    oracle_actual_gco2eq: float
    ecotracker_delay_h: float
    oracle_delay_h: float
    ecotracker_saved_pct: float
    oracle_saved_pct: float
    ecotracker_outcome: str


@dataclass(frozen=True, slots=True)
class DailyRow:
    day: date
    baseline_total_gco2eq: float
    ecotracker_total_gco2eq: float
    oracle_total_gco2eq: float
    ecotracker_saved_pct: float
    oracle_saved_pct: float
    oracle_capture_ratio: float | None
    mean_ecotracker_delay_h: float


@dataclass(frozen=True, slots=True)
class ForecastPoint:
    starts_at: datetime
    ends_at: datetime
    forecast_intensity: float
    actual_intensity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready benchmark figures")
    parser.add_argument("--scenario-results-csv", type=Path, default=DEFAULT_SCENARIO_RESULTS_CSV)
    parser.add_argument("--daily-summary-csv", type=Path, default=DEFAULT_DAILY_SUMMARY_CSV)
    parser.add_argument("--forecast-csv", type=Path, default=DEFAULT_FORECAST_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--curve-day",
        type=str,
        default=None,
        help="Representative UTC day to plot in Figure 1, for example 2026-03-01.",
    )
    parser.add_argument(
        "--curve-point-mode",
        choices=("shifted-only", "all"),
        default="shifted-only",
        help="Whether to overlay only shifted submissions or all submissions in Figure 1.",
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
            "Run the benchmark and analysis scripts first."
        )


def parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value)


def load_scenario_rows(path: Path) -> list[ScenarioRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            ScenarioRow(
                day=date.fromisoformat(row["day"]),
                submission_slot=int(row["submission_slot"]),
                submission_time=parse_timestamp(row["submission_time"]),
                baseline_execution_time=parse_timestamp(row["baseline_execution_time"]),
                ecotracker_execution_time=parse_timestamp(row["ecotracker_execution_time"]),
                oracle_execution_time=parse_timestamp(row["oracle_execution_time"]),
                baseline_actual_intensity=float(row["baseline_actual_intensity"]),
                ecotracker_actual_intensity_at_run=float(row["ecotracker_actual_intensity_at_run"]),
                oracle_actual_intensity_at_run=float(row["oracle_selected_actual_intensity"]),
                baseline_actual_gco2eq=float(row["baseline_actual_gco2eq"]),
                ecotracker_actual_gco2eq=float(row["ecotracker_actual_gco2eq"]),
                oracle_actual_gco2eq=float(row["oracle_actual_gco2eq"]),
                ecotracker_delay_h=float(row["ecotracker_delay_h"]),
                oracle_delay_h=float(row["oracle_delay_h"]),
                ecotracker_saved_pct=float(row["ecotracker_saved_pct"]),
                oracle_saved_pct=float(row["oracle_saved_pct"]),
                ecotracker_outcome=row["ecotracker_outcome"],
            )
            for row in reader
        ]
    if not rows:
        raise SystemExit(f"No scenario rows were found in '{path}'.")
    return rows


def create_figure(*, figsize: tuple[float, float], bottom_margin: float = 0.2):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=bottom_margin)
    return fig, ax


def add_figure_caption(
    fig,
    text: str,
    *,
    x: float = 0.985,
    y: float = 0.035,
    ha: str = "right",
) -> None:
    fig.text(
        x,
        y,
        text,
        transform=fig.transFigure,
        ha=ha,
        va="bottom",
        fontsize=10,
        bbox=CAPTION_BBOX,
    )


def load_daily_rows(path: Path) -> list[DailyRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            DailyRow(
                day=date.fromisoformat(row["day"]),
                baseline_total_gco2eq=float(row["baseline_total_gco2eq"]),
                ecotracker_total_gco2eq=float(row["ecotracker_total_gco2eq"]),
                oracle_total_gco2eq=float(row["oracle_total_gco2eq"]),
                ecotracker_saved_pct=float(row["ecotracker_saved_pct"]),
                oracle_saved_pct=float(row["oracle_saved_pct"]),
                oracle_capture_ratio=(
                    float(row["oracle_capture_ratio"])
                    if row["oracle_capture_ratio"] not in {"", None}
                    else None
                ),
                mean_ecotracker_delay_h=float(row["mean_ecotracker_delay_h"]),
            )
            for row in reader
        ]
    if not rows:
        raise SystemExit(f"No daily summary rows were found in '{path}'.")
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


def choose_curve_day(daily_rows: list[DailyRow], requested_day: str | None) -> date:
    if requested_day is not None:
        target_day = date.fromisoformat(requested_day)
        if any(row.day == target_day for row in daily_rows):
            return target_day
        raise SystemExit(f"The requested curve day '{requested_day}' is not present in daily_summary.csv.")

    sorted_rows = sorted(daily_rows, key=lambda row: row.ecotracker_saved_pct)
    return sorted_rows[len(sorted_rows) // 2].day


def apply_time_axis(ax, *, window_start: datetime, window_end: datetime) -> None:
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(window_start, window_end)


def select_curve_rows(
    scenario_rows: list[ScenarioRow],
    curve_day: date,
    *,
    point_mode: str,
) -> list[ScenarioRow]:
    rows = [row for row in scenario_rows if row.day == curve_day]
    if point_mode == "shifted-only":
        shifted_rows = [row for row in rows if row.ecotracker_delay_h > 0]
        if shifted_rows:
            return shifted_rows
    return rows


def select_curve_window(
    curve_day: date,
    curve_rows: list[ScenarioRow],
) -> tuple[datetime, datetime]:
    window_start = datetime.combine(curve_day, time(0, 0), tzinfo=timezone.utc)
    execution_end = max(
        max(row.submission_time, row.ecotracker_execution_time) for row in curve_rows
    ) + timedelta(minutes=30)
    window_end = max(
        datetime.combine(curve_day + timedelta(days=1), time(0, 0), tzinfo=timezone.utc),
        execution_end,
    )
    return window_start, window_end


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


def plot_representative_day_curve(
    scenario_rows: list[ScenarioRow],
    daily_rows: list[DailyRow],
    forecast_points: list[ForecastPoint],
    *,
    curve_day: date,
    point_mode: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    curve_rows = select_curve_rows(scenario_rows, curve_day, point_mode=point_mode)
    if not curve_rows:
        raise SystemExit(f"No scenario rows were available for representative day {curve_day.isoformat()}.")

    window_start, window_end = select_curve_window(curve_day, curve_rows)
    window_points = filter_forecast_window(
        forecast_points,
        window_start=window_start,
        window_end=window_end,
    )
    day_summary = next(row for row in daily_rows if row.day == curve_day)

    fig, ax = create_figure(figsize=(10.8, 6.4), bottom_margin=0.2)
    ax.plot(
        [point.starts_at for point in window_points],
        [point.actual_intensity for point in window_points],
        color=COLOR_GRID,
        linewidth=2.3,
        label="Grid carbon intensity (actual)",
        zorder=2,
    )
    ax.scatter(
        [row.baseline_execution_time for row in curve_rows],
        [row.baseline_actual_intensity for row in curve_rows],
        color=COLOR_BASELINE,
        marker="o",
        s=48,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.5,
        label="Baseline execution",
        zorder=4,
    )
    ax.scatter(
        [row.ecotracker_execution_time for row in curve_rows],
        [row.ecotracker_actual_intensity_at_run for row in curve_rows],
        color=COLOR_ECOTRACKER,
        marker="^",
        s=62,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
        label="EcoTracker execution",
        zorder=5,
    )

    apply_time_axis(ax, window_start=window_start, window_end=window_end)
    ax.set_xlabel("Execution time")
    ax.set_ylabel("Carbon intensity (gCO2eq/kWh)")
    ax.legend(loc="upper left")

    summary_text = (
        f"Representative day: {curve_day.isoformat()}\n"
        f"Daily reduction: {day_summary.ecotracker_saved_pct:.2f}%\n"
        f"Mean delay: {day_summary.mean_ecotracker_delay_h:.2f} h"
    )
    add_figure_caption(fig, summary_text)

    return save_figure(fig, output_dir, "figure_1_curve", formats, dpi)


def plot_total_emissions_bar(
    daily_rows: list[DailyRow],
    *,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    baseline_total = sum(row.baseline_total_gco2eq for row in daily_rows)
    ecotracker_total = sum(row.ecotracker_total_gco2eq for row in daily_rows)
    oracle_total = sum(row.oracle_total_gco2eq for row in daily_rows)
    ecotracker_reduction_pct = (
        ((baseline_total - ecotracker_total) / baseline_total) * 100.0 if baseline_total > 0 else 0.0
    )
    oracle_reduction_pct = (
        ((baseline_total - oracle_total) / baseline_total) * 100.0 if baseline_total > 0 else 0.0
    )

    fig, ax = create_figure(figsize=(8.2, 6.4), bottom_margin=0.2)
    labels = ["Baseline", "EcoTracker", "Oracle"]
    values = [baseline_total, ecotracker_total, oracle_total]
    colors = [COLOR_BASELINE, COLOR_ECOTRACKER, COLOR_ORACLE]
    bars = ax.bar(labels, values, color=colors, width=0.58, zorder=3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(values) * 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("Total emitted carbon (gCO2eq)")
    ax.set_axisbelow(True)
    add_figure_caption(
        fig,
        (
            f"EcoTracker reduction: {ecotracker_reduction_pct:.2f}%\n"
            f"Oracle reduction: {oracle_reduction_pct:.2f}%"
        ),
    )

    return save_figure(fig, output_dir, "figure_2_total_emissions", formats, dpi)


def plot_daily_reduction_distribution(
    daily_rows: list[DailyRow],
    *,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    ecotracker_reductions = [row.ecotracker_saved_pct for row in daily_rows]
    oracle_reductions = [row.oracle_saved_pct for row in daily_rows]

    fig, ax = create_figure(figsize=(8.6, 6.4), bottom_margin=0.2)
    boxplot = ax.boxplot(
        [ecotracker_reductions, oracle_reductions],
        tick_labels=["EcoTracker", "Oracle"],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, color in zip(boxplot["boxes"], [COLOR_ECOTRACKER, COLOR_ORACLE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.4)
    for whisker, color in zip(boxplot["whiskers"], [COLOR_ECOTRACKER, COLOR_ECOTRACKER, COLOR_ORACLE, COLOR_ORACLE]):
        whisker.set_color(color)
        whisker.set_linewidth(1.2)
    for cap, color in zip(boxplot["caps"], [COLOR_ECOTRACKER, COLOR_ECOTRACKER, COLOR_ORACLE, COLOR_ORACLE]):
        cap.set_color(color)
        cap.set_linewidth(1.2)

    rng = random.Random(1729)
    for position, values, color in (
        (1, ecotracker_reductions, COLOR_ECOTRACKER),
        (2, oracle_reductions, COLOR_ORACLE),
    ):
        jitter = [position + rng.uniform(-0.08, 0.08) for _ in values]
        ax.scatter(
            jitter,
            values,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            s=36,
            alpha=0.85,
            zorder=4,
        )

    ax.set_ylabel("Daily reduction relative to baseline (%)")
    ax.set_axisbelow(True)
    add_figure_caption(
        fig,
        (
            f"EcoTracker mean: {statistics.mean(ecotracker_reductions):.2f}%\n"
            f"EcoTracker median: {statistics.median(ecotracker_reductions):.2f}%\n"
            f"Days: {len(daily_rows)}"
        ),
    )

    return save_figure(fig, output_dir, "figure_3_daily_reductions", formats, dpi)


def build_manifest(
    daily_rows: list[DailyRow],
    written_paths: list[Path],
    *,
    output_dir: Path,
    curve_day: date,
) -> Path:
    baseline_total = sum(row.baseline_total_gco2eq for row in daily_rows)
    ecotracker_total = sum(row.ecotracker_total_gco2eq for row in daily_rows)
    oracle_total = sum(row.oracle_total_gco2eq for row in daily_rows)
    manifest = {
        "day_count": len(daily_rows),
        "curve_day": curve_day.isoformat(),
        "aggregate_ecotracker_reduction_pct": (
            ((baseline_total - ecotracker_total) / baseline_total) * 100.0 if baseline_total > 0 else 0.0
        ),
        "aggregate_oracle_reduction_pct": (
            ((baseline_total - oracle_total) / baseline_total) * 100.0 if baseline_total > 0 else 0.0
        ),
        "mean_daily_ecotracker_reduction_pct": statistics.mean(
            [row.ecotracker_saved_pct for row in daily_rows]
        ),
        "median_daily_ecotracker_reduction_pct": statistics.median(
            [row.ecotracker_saved_pct for row in daily_rows]
        ),
        "written_files": [str(path) for path in written_paths],
    }
    manifest_path = output_dir / "figure_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest_path


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    ensure_inputs_exist(
        args.scenario_results_csv,
        args.daily_summary_csv,
        args.forecast_csv,
    )

    scenario_rows = load_scenario_rows(args.scenario_results_csv)
    daily_rows = load_daily_rows(args.daily_summary_csv)
    forecast_points = load_forecast_points(args.forecast_csv)
    curve_day = choose_curve_day(daily_rows, args.curve_day)

    written_paths: list[Path] = []
    written_paths.extend(
        plot_representative_day_curve(
            scenario_rows,
            daily_rows,
            forecast_points,
            curve_day=curve_day,
            point_mode=args.curve_point_mode,
            output_dir=args.output_dir,
            formats=args.formats,
            dpi=args.dpi,
        )
    )
    written_paths.extend(
        plot_total_emissions_bar(
            daily_rows,
            output_dir=args.output_dir,
            formats=args.formats,
            dpi=args.dpi,
        )
    )
    written_paths.extend(
        plot_daily_reduction_distribution(
            daily_rows,
            output_dir=args.output_dir,
            formats=args.formats,
            dpi=args.dpi,
        )
    )
    manifest_path = build_manifest(
        daily_rows,
        written_paths,
        output_dir=args.output_dir,
        curve_day=curve_day,
    )

    print("=" * 72)
    print("PAPER FIGURES GENERATED")
    print("=" * 72)
    for path in written_paths:
        print(path)
    print(manifest_path)
    print("=" * 72)


if __name__ == "__main__":
    main()
