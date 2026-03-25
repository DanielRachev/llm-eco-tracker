from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

logger = logging.getLogger(__name__)

DEFAULT_TELEMETRY_PATH = Path("eco_telemetry.jsonl")
LOGGER_RECORD_MARKER = "Telemetry record: "
CAR_GCO2EQ_PER_KM = 170.0


class ReportInputError(Exception):
    """Raised when a telemetry report input cannot be processed."""


@dataclass(frozen=True, slots=True)
class TelemetryReport:
    total_jobs: int = 0
    total_baseline_gco2eq: float = 0.0
    total_actual_gco2eq: float = 0.0
    total_saved_gco2eq: float = 0.0

    @property
    def equivalent_km_driven(self) -> float:
        if CAR_GCO2EQ_PER_KM <= 0:
            return 0.0
        return self.total_actual_gco2eq / CAR_GCO2EQ_PER_KM


class ReportLoader(Protocol):
    name: str

    def matches(self, sample_lines: Sequence[str]) -> bool:
        """Return whether this loader can parse the sampled input."""

    def iter_payloads(self, path: Path) -> Iterator[dict[str, Any]]:
        """Yield parsed telemetry payloads for one input file."""


class JsonlReportLoader:
    name = "jsonl"

    def matches(self, sample_lines: Sequence[str]) -> bool:
        for line in sample_lines:
            stripped = line.strip()
            if stripped:
                return stripped.startswith("{")
        return True

    def iter_payloads(self, path: Path) -> Iterator[dict[str, Any]]:
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    payload = _parse_payload(line, path=path, line_number=line_number)
                    if payload is not None:
                        yield payload
        except OSError as exc:
            raise ReportInputError(f"Could not read telemetry input '{path}': {exc}") from exc


class LoggerReportLoader:
    name = "logger"

    def matches(self, sample_lines: Sequence[str]) -> bool:
        return any(LOGGER_RECORD_MARKER in line for line in sample_lines)

    def iter_payloads(self, path: Path) -> Iterator[dict[str, Any]]:
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    marker_index = line.find(LOGGER_RECORD_MARKER)
                    if marker_index < 0:
                        continue

                    payload_text = line[marker_index + len(LOGGER_RECORD_MARKER) :].strip()
                    if not payload_text:
                        logger.warning("Skipping empty logger payload in '%s' line %s", path, line_number)
                        continue

                    payload = _parse_payload(payload_text, path=path, line_number=line_number)
                    if payload is not None:
                        yield payload
        except OSError as exc:
            raise ReportInputError(f"Could not read telemetry input '{path}': {exc}") from exc


REPORT_LOADERS: tuple[ReportLoader, ...] = (
    LoggerReportLoader(),
    JsonlReportLoader(),
)


def iter_telemetry_payloads(
    path: str | Path,
    *,
    report_format: str = "auto",
) -> Iterator[dict[str, Any]]:
    resolved_path = Path(path)
    loader = _resolve_loader(resolved_path, report_format=report_format)
    yield from loader.iter_payloads(resolved_path)


def build_report(
    paths: Sequence[str | Path] | None = None,
    *,
    report_format: str = "auto",
) -> TelemetryReport:
    report_paths = [Path(path) for path in paths] if paths else [DEFAULT_TELEMETRY_PATH]

    total_jobs = 0
    total_baseline_gco2eq = 0.0
    total_actual_gco2eq = 0.0
    total_saved_gco2eq = 0.0

    for report_path in report_paths:
        for payload in iter_telemetry_payloads(report_path, report_format=report_format):
            total_jobs += 1

            baseline_gco2eq = _coerce_float(
                payload.get("baseline_gco2eq"),
                field_name="baseline_gco2eq",
                path=report_path,
            )
            actual_gco2eq = _coerce_float(
                payload.get("actual_gco2eq"),
                field_name="actual_gco2eq",
                path=report_path,
            )
            saved_gco2eq = _coerce_float(
                payload.get("saved_gco2eq"),
                field_name="saved_gco2eq",
                path=report_path,
            )

            if baseline_gco2eq is not None:
                total_baseline_gco2eq += baseline_gco2eq
            if actual_gco2eq is not None:
                total_actual_gco2eq += actual_gco2eq

            if baseline_gco2eq is not None and actual_gco2eq is not None:
                total_saved_gco2eq += baseline_gco2eq - actual_gco2eq
            elif saved_gco2eq is not None:
                total_saved_gco2eq += saved_gco2eq
            else:
                logger.warning(
                    "Telemetry payload in '%s' did not contain enough emissions data for savings.",
                    report_path,
                )

    return TelemetryReport(
        total_jobs=total_jobs,
        total_baseline_gco2eq=total_baseline_gco2eq,
        total_actual_gco2eq=total_actual_gco2eq,
        total_saved_gco2eq=total_saved_gco2eq,
    )


def render_report(report: TelemetryReport) -> str:
    lines = [
        "EcoTracker Lifetime Report",
        "==========================",
    ]

    if report.total_jobs == 0:
        lines.append("No telemetry records found.")

    lines.extend(
        [
            f"{'Total LLM Jobs Run':<28} {report.total_jobs}",
            f"{'Total gCO2eq Emitted':<28} {report.total_actual_gco2eq:,.4f}",
            f"{'Total gCO2eq Saved by EcoTracker':<28} {report.total_saved_gco2eq:,.4f}",
            f"{'Equivalent to':<28} {report.equivalent_km_driven:,.3f} km driven in a car",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize EcoTracker telemetry from JSONL or logger-backed inputs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Telemetry inputs to summarize. Defaults to ./eco_telemetry.jsonl",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "jsonl", "logger"),
        default="auto",
        help="Telemetry input format. Defaults to auto-detection.",
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(args.paths, report_format=args.format)
    except ReportInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(render_report(report))
    return 0


def _resolve_loader(path: Path, *, report_format: str) -> ReportLoader:
    if not path.exists():
        raise ReportInputError(f"Telemetry input not found: {path}")
    if not path.is_file():
        raise ReportInputError(f"Telemetry input is not a file: {path}")

    if report_format != "auto":
        for loader in REPORT_LOADERS:
            if loader.name == report_format:
                return loader
        raise ReportInputError(f"Unsupported report format: {report_format}")

    sample_lines = _read_sample_lines(path)
    for loader in REPORT_LOADERS:
        if loader.matches(sample_lines):
            return loader

    return JsonlReportLoader()


def _read_sample_lines(path: Path, *, limit: int = 20) -> list[str]:
    sample_lines: list[str] = []

    try:
        with path.open(encoding="utf-8") as handle:
            for _, line in zip(range(limit), handle):
                sample_lines.append(line)
    except OSError as exc:
        raise ReportInputError(f"Could not read telemetry input '{path}': {exc}") from exc

    return sample_lines


def _parse_payload(
    payload_text: str,
    *,
    path: Path,
    line_number: int,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        logger.warning("Skipping malformed telemetry payload in '%s' line %s", path, line_number)
        return None

    if not isinstance(payload, dict):
        logger.warning("Skipping non-object telemetry payload in '%s' line %s", path, line_number)
        return None

    return payload


def _coerce_float(value: Any, *, field_name: str, path: Path) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Skipping non-numeric field '%s' in '%s': %r", field_name, path, value)
        return None


if __name__ == "__main__":
    raise SystemExit(main())
