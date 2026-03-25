from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from llm_eco_tracker.report import (
    LOGGER_RECORD_MARKER,
    build_report,
    iter_telemetry_payloads,
    main,
    render_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TelemetryReportTests(unittest.TestCase):
    def test_jsonl_loader_parses_valid_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            telemetry_path = self._write_jsonl(
                temp_path / "eco_telemetry.jsonl",
                [
                    self._payload(baseline=14.0, actual=10.0),
                    self._payload(baseline=8.0, actual=5.5),
                ],
            )

            payloads = list(iter_telemetry_payloads(telemetry_path, report_format="jsonl"))

        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["actual_gco2eq"], 10.0)
        self.assertEqual(payloads[1]["saved_gco2eq"], 2.5)

    def test_logger_loader_extracts_payloads_and_ignores_other_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log_path = temp_path / "app.log"
            payload = self._payload(baseline=9.0, actual=6.0)
            log_path.write_text(
                "\n".join(
                    [
                        "INFO Booting app",
                        f"2026-03-25 10:00:00 INFO {LOGGER_RECORD_MARKER}{json.dumps(payload)}",
                        "INFO Shutting down",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            payloads = list(iter_telemetry_payloads(log_path, report_format="logger"))

        self.assertEqual(payloads, [payload])

    def test_auto_detection_chooses_logger_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            payload = self._payload(baseline=20.0, actual=12.5)
            log_path = temp_path / "telemetry.log"
            log_path.write_text(
                f"DEBUG {LOGGER_RECORD_MARKER}{json.dumps(payload)}\n",
                encoding="utf-8",
            )

            payloads = list(iter_telemetry_payloads(log_path, report_format="auto"))

        self.assertEqual(payloads, [payload])

    def test_build_report_aggregates_multiple_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first_path = self._write_jsonl(
                temp_path / "one.jsonl",
                [self._payload(baseline=10.0, actual=7.0)],
            )
            second_path = self._write_jsonl(
                temp_path / "two.jsonl",
                [
                    self._payload(baseline=5.5, actual=3.0),
                    self._payload(baseline=8.0, actual=6.0),
                ],
            )

            report = build_report([first_path, second_path], report_format="jsonl")

        self.assertEqual(report.total_jobs, 3)
        self.assertAlmostEqual(report.total_baseline_gco2eq, 23.5)
        self.assertAlmostEqual(report.total_actual_gco2eq, 16.0)
        self.assertAlmostEqual(report.total_saved_gco2eq, 7.5)

    def test_malformed_lines_are_skipped_without_aborting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            telemetry_path = temp_path / "eco_telemetry.jsonl"
            telemetry_path.write_text(
                "\n".join(
                    [
                        json.dumps(self._payload(baseline=11.0, actual=9.0)),
                        "not-json",
                        json.dumps(self._payload(baseline=4.0, actual=2.0)),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertLogs("llm_eco_tracker.report", level="WARNING") as captured_logs:
                report = build_report([telemetry_path], report_format="jsonl")

        self.assertEqual(report.total_jobs, 2)
        self.assertAlmostEqual(report.total_saved_gco2eq, 4.0)
        self.assertTrue(
            any("Skipping malformed telemetry payload" in message for message in captured_logs.output)
        )

    def test_render_report_includes_empty_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            telemetry_path = temp_path / "empty.jsonl"
            telemetry_path.write_text("", encoding="utf-8")

            report = build_report([telemetry_path], report_format="jsonl")

        output = render_report(report)

        self.assertIn("No telemetry records found.", output)
        self.assertIn("Total LLM Jobs Run", output)
        self.assertIn("Equivalent to", output)

    def test_main_returns_error_for_missing_file(self):
        missing_path = REPO_ROOT / "missing-telemetry.jsonl"
        stderr = StringIO()

        with redirect_stderr(stderr):
            exit_code = main([str(missing_path)])

        self.assertEqual(exit_code, 1)
        self.assertIn("Telemetry input not found", stderr.getvalue())

    def test_cli_smoke_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            telemetry_path = self._write_jsonl(
                temp_path / "eco_telemetry.jsonl",
                [self._payload(baseline=9.5, actual=7.25)],
            )

            result = subprocess.run(
                [sys.executable, "-m", "llm_eco_tracker.report", str(telemetry_path), "--format", "jsonl"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=20,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("EcoTracker Lifetime Report", result.stdout)
        self.assertIn("Total LLM Jobs Run", result.stdout)

    def test_cli_smoke_logger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            payload = self._payload(baseline=18.0, actual=11.0)
            log_path = temp_path / "app.log"
            log_path.write_text(
                "\n".join(
                    [
                        "INFO Background task complete",
                        f"2026-03-25 11:00:00 INFO {LOGGER_RECORD_MARKER}{json.dumps(payload)}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "-m", "llm_eco_tracker.report", str(log_path), "--format", "logger"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=20,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Total gCO2eq Emitted", result.stdout)
        self.assertIn("km driven in a car", result.stdout)

    @staticmethod
    def _payload(*, baseline: float, actual: float, saved: float | None = None) -> dict[str, float | str]:
        if saved is None:
            saved = baseline - actual

        return {
            "timestamp": "2026-03-25T12:00:00+00:00",
            "baseline_gco2eq": baseline,
            "actual_gco2eq": actual,
            "saved_gco2eq": saved,
        }

    @staticmethod
    def _write_jsonl(path: Path, payloads: list[dict[str, float | str]]) -> Path:
        path.write_text(
            "\n".join(json.dumps(payload) for payload in payloads) + "\n",
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()
