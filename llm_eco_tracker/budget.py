from __future__ import annotations

from .models import CarbonBudgetPolicy, SchedulePlan


def build_carbon_budget_policy(
    schedule_plan: SchedulePlan,
    *,
    max_session_gco2eq: float | None = None,
) -> CarbonBudgetPolicy:
    if max_session_gco2eq is None:
        return CarbonBudgetPolicy(
            enabled=False,
            max_session_gco2eq=None,
            actual_intensity_gco2eq_per_kwh=float(schedule_plan.optimal_intensity_gco2eq_per_kwh),
        )

    normalized_max_session_gco2eq = float(max_session_gco2eq)
    if normalized_max_session_gco2eq <= 0:
        raise ValueError("max_session_gco2eq must be greater than 0 when provided.")

    return CarbonBudgetPolicy(
        enabled=True,
        max_session_gco2eq=normalized_max_session_gco2eq,
        actual_intensity_gco2eq_per_kwh=float(schedule_plan.optimal_intensity_gco2eq_per_kwh),
    )
