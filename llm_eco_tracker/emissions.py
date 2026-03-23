from __future__ import annotations

from .models import EmissionSummary


def summarize_emissions(
    energy_kwh: float,
    baseline_intensity_gco2eq_per_kwh: float,
    actual_intensity_gco2eq_per_kwh: float,
) -> EmissionSummary:
    baseline_gco2eq = energy_kwh * baseline_intensity_gco2eq_per_kwh
    actual_gco2eq = energy_kwh * actual_intensity_gco2eq_per_kwh
    saved_gco2eq = baseline_gco2eq - actual_gco2eq

    return EmissionSummary(
        energy_kwh=energy_kwh,
        baseline_gco2eq=baseline_gco2eq,
        actual_gco2eq=actual_gco2eq,
        saved_gco2eq=saved_gco2eq,
    )
