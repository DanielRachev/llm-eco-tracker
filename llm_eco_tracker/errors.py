from __future__ import annotations


class CarbonBudgetExceededError(RuntimeError):
    def __init__(
        self,
        *,
        max_session_gco2eq: float,
        actual_gco2eq: float,
        energy_kwh: float,
    ):
        self.max_session_gco2eq = float(max_session_gco2eq)
        self.actual_gco2eq = float(actual_gco2eq)
        self.energy_kwh = float(energy_kwh)
        super().__init__(
            "Carbon session budget exceeded: "
            f"{self.actual_gco2eq:.4f} gCO2eq used exceeds "
            f"{self.max_session_gco2eq:.4f} gCO2eq."
        )
