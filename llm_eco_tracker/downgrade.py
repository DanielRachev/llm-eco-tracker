from __future__ import annotations

from collections.abc import Mapping

from .models import ModelDowngradePolicy, SchedulePlan

DEFAULT_OPENAI_MODEL_FALLBACKS: dict[str, str] = {
    "gpt-4o": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1-mini",
    "gpt-4-turbo": "gpt-4o-mini",
    "gpt-4": "gpt-4o-mini",
}


def build_model_downgrade_policy(
    schedule_plan: SchedulePlan,
    *,
    auto_downgrade: bool = False,
    dirty_threshold: float = 300.0,
    model_fallbacks: Mapping[str, str] | None = None,
) -> ModelDowngradePolicy:
    fallback_map = dict(DEFAULT_OPENAI_MODEL_FALLBACKS)
    if model_fallbacks:
        fallback_map.update(model_fallbacks)

    return ModelDowngradePolicy(
        enabled=bool(auto_downgrade),
        dirty_threshold_gco2eq_per_kwh=float(dirty_threshold),
        execution_intensity_gco2eq_per_kwh=float(schedule_plan.optimal_intensity_gco2eq_per_kwh),
        fallback_map=fallback_map,
    )
