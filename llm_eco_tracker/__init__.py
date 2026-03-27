from .api import carbon_aware
from .errors import CarbonBudgetExceededError

__version__ = "0.1.0"

__all__ = ["CarbonBudgetExceededError", "__version__", "carbon_aware"]
