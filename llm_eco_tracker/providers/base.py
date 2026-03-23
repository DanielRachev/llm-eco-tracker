from __future__ import annotations

from typing import Protocol

from ..models import ForecastSnapshot


class ForecastProvider(Protocol):
    provider_name: str

    def load_forecast(self, max_delay_hours: float) -> ForecastSnapshot:
        """Load forecast intervals for the requested delay horizon."""
