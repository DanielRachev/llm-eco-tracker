from __future__ import annotations

import unittest
from pathlib import Path

from llm_eco_tracker.benchmarking import (
    build_trace_schedule_plan,
    group_complete_trace_days,
    load_trace_intervals,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "mock_forecast.csv"


class BenchmarkingHelpersTests(unittest.TestCase):
    def test_group_complete_trace_days_filters_to_eligible_complete_days(self):
        trace_intervals = load_trace_intervals(FIXTURE_PATH)

        trace_days = group_complete_trace_days(trace_intervals, max_delay_hours=4.0)

        self.assertGreaterEqual(len(trace_days), 2)
        self.assertEqual(trace_days[0].day.isoformat(), "2026-03-18")
        self.assertEqual(trace_days[1].day.isoformat(), "2026-03-19")
        self.assertEqual(trace_days[0].submission_count, 48)

    def test_build_trace_schedule_plan_respects_delay_budget_for_forecast_and_oracle(self):
        trace_intervals = load_trace_intervals(FIXTURE_PATH)
        trace_days = group_complete_trace_days(trace_intervals, max_delay_hours=4.0)
        first_offset = trace_days[0].submission_offsets[0]

        forecast_plan, forecast_offset = build_trace_schedule_plan(
            trace_intervals,
            first_offset,
            max_delay_hours=4.0,
            intensity_kind="forecast",
        )
        oracle_plan, oracle_offset = build_trace_schedule_plan(
            trace_intervals,
            first_offset,
            max_delay_hours=4.0,
            intensity_kind="actual",
        )

        self.assertGreaterEqual(forecast_offset, first_offset)
        self.assertGreaterEqual(oracle_offset, first_offset)
        self.assertLessEqual(forecast_plan.execution_delay_seconds, 4.0 * 3600.0)
        self.assertLessEqual(oracle_plan.execution_delay_seconds, 4.0 * 3600.0)
        self.assertIsNotNone(forecast_plan.selected_interval)
        self.assertIsNotNone(oracle_plan.selected_interval)


if __name__ == "__main__":
    unittest.main()
