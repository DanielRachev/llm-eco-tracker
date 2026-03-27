from __future__ import annotations

from numbers import Real


def normalize_energy_value(energy) -> float | None:
    if isinstance(energy, Real):
        return float(energy)

    mean_value = getattr(energy, "mean", None)
    if isinstance(mean_value, Real):
        return float(mean_value)

    minimum = getattr(energy, "min", None)
    maximum = getattr(energy, "max", None)
    if isinstance(minimum, Real) and isinstance(maximum, Real):
        return float((minimum + maximum) / 2.0)
    if isinstance(minimum, Real):
        return float(minimum)
    if isinstance(maximum, Real):
        return float(maximum)

    return None
