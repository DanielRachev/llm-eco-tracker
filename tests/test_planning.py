from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from llm_eco_tracker.models import ForecastInterval
from llm_eco_tracker.planning import apply_jitter_to_plan, build_schedule_plan


class PlanningJitterTests(unittest.TestCase):
    def test_jitter_starts_at_selected_interval_and_moves_forward_only(self):
        now = datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc)
        plan = build_schedule_plan(
            (
                ForecastInterval(now, now + timedelta(minutes=30), 400.0),
                ForecastInterval(
                    now + timedelta(hours=2),
                    now + timedelta(hours=2, minutes=30),
                    100.0,
                ),
            ),
            max_delay_hours=3,
            reference_time=now,
        )

        jittered_plan = apply_jitter_to_plan(
            plan,
            max_jitter_seconds=600.0,
            random_uniform=lambda lower, upper: upper,
        )

        self.assertAlmostEqual(plan.raw_delay_seconds, 7200.0)
        self.assertAlmostEqual(jittered_plan.execution_delay_seconds, 7800.0)
        self.assertGreaterEqual(jittered_plan.execution_delay_seconds, plan.raw_delay_seconds)
        self.assertLessEqual(
            jittered_plan.execution_delay_seconds,
            jittered_plan.latest_execution_delay_seconds,
        )

    def test_jitter_is_capped_by_selected_interval_end(self):
        now = datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc)
        plan = build_schedule_plan(
            (
                ForecastInterval(now, now + timedelta(minutes=30), 350.0),
                ForecastInterval(
                    now + timedelta(hours=2),
                    now + timedelta(hours=2, minutes=5),
                    100.0,
                ),
            ),
            max_delay_hours=4,
            reference_time=now,
        )

        jittered_plan = apply_jitter_to_plan(
            plan,
            max_jitter_seconds=900.0,
            random_uniform=lambda lower, upper: upper,
        )

        self.assertAlmostEqual(plan.latest_execution_delay_seconds, 7500.0)
        self.assertAlmostEqual(jittered_plan.execution_delay_seconds, 7500.0)

    def test_jitter_is_capped_by_max_delay_budget(self):
        now = datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc)
        plan = build_schedule_plan(
            (
                ForecastInterval(now, now + timedelta(minutes=30), 400.0),
                ForecastInterval(
                    now + timedelta(hours=1, minutes=50),
                    now + timedelta(hours=2, minutes=20),
                    100.0,
                ),
            ),
            max_delay_hours=2,
            reference_time=now,
        )

        jittered_plan = apply_jitter_to_plan(
            plan,
            max_jitter_seconds=1800.0,
            random_uniform=lambda lower, upper: upper,
        )

        self.assertAlmostEqual(plan.raw_delay_seconds, 6600.0)
        self.assertAlmostEqual(plan.latest_execution_delay_seconds, 7200.0)
        self.assertAlmostEqual(jittered_plan.execution_delay_seconds, 7200.0)

    def test_immediate_execution_is_not_delayed_by_jitter(self):
        now = datetime(2026, 3, 27, 12, 0, tzinfo=timezone.utc)
        plan = build_schedule_plan(
            (
                ForecastInterval(now, now + timedelta(minutes=30), 100.0),
            ),
            max_delay_hours=2,
            reference_time=now,
        )

        jittered_plan = apply_jitter_to_plan(
            plan,
            max_jitter_seconds=600.0,
            random_uniform=lambda lower, upper: upper,
        )

        self.assertEqual(plan.raw_delay_seconds, 0.0)
        self.assertEqual(jittered_plan.execution_delay_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
