from __future__ import annotations

import contextvars
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..errors import CarbonBudgetExceededError
from ..models import CarbonBudgetPolicy, ModelDowngradePolicy, ModelUsageSummary
from .base import TelemetryAdapter

logger = logging.getLogger(__name__)


def _disabled_model_downgrade_policy() -> ModelDowngradePolicy:
    return ModelDowngradePolicy(
        enabled=False,
        dirty_threshold_gco2eq_per_kwh=0.0,
        execution_intensity_gco2eq_per_kwh=0.0,
        fallback_map={},
    )


def _disabled_carbon_budget_policy() -> CarbonBudgetPolicy:
    return CarbonBudgetPolicy(
        enabled=False,
        max_session_gco2eq=None,
        actual_intensity_gco2eq_per_kwh=0.0,
    )


@dataclass(slots=True)
class _SessionState:
    energy_kwh: float = 0.0
    actual_gco2eq_so_far: float = 0.0
    llm_providers: set[str] = field(default_factory=set)
    carbon_budget_policy: CarbonBudgetPolicy = field(default_factory=_disabled_carbon_budget_policy)
    carbon_budget_exceeded: bool = False
    model_downgrade_policy: ModelDowngradePolicy = field(
        default_factory=_disabled_model_downgrade_policy
    )
    model_usage_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    warned_unmapped_models: set[str] = field(default_factory=set)


class _RuntimeSessionHooks:
    def __init__(self, runtime: "EcoLogitsRuntime"):
        self._runtime = runtime

    def get_model_downgrade_policy(self) -> ModelDowngradePolicy:
        return self._runtime._get_model_downgrade_policy()

    def record_energy_kwh(self, energy_kwh: float) -> None:
        self._runtime._record_energy(energy_kwh)

    def record_llm_provider(self, provider_name: str) -> None:
        self._runtime._record_llm_provider(provider_name)

    def record_model_usage(self, requested_model: str | None, effective_model: str | None) -> None:
        self._runtime._record_model_usage(requested_model, effective_model)

    def should_warn_unmapped_model(self, requested_model: str) -> bool:
        return self._runtime._should_warn_unmapped_model(requested_model)


class EcoTelemetrySession:
    def __init__(
        self,
        runtime: "EcoLogitsRuntime",
        *,
        carbon_budget_policy: CarbonBudgetPolicy | None = None,
        model_downgrade_policy: ModelDowngradePolicy | None = None,
    ):
        self._runtime = runtime
        self._carbon_budget_policy = carbon_budget_policy or _disabled_carbon_budget_policy()
        self._model_downgrade_policy = model_downgrade_policy or _disabled_model_downgrade_policy()
        self._state: _SessionState | None = None
        self._token = None
        self._patch_enabled = False

    def __enter__(self):
        self._state = _SessionState(
            carbon_budget_policy=self._carbon_budget_policy,
            model_downgrade_policy=self._model_downgrade_policy,
        )
        self._token = self._runtime._session_state.set(self._state)
        ecologits_ready = self._runtime._ensure_ecologits_initialized()
        self._patch_enabled = self._runtime._install_adapters()
        if self._carbon_budget_policy.enabled and not self._carbon_budget_policy.is_enforced:
            logger.warning(
                "Carbon circuit breaker is configured but cannot be enforced because execution carbon intensity is unavailable."
            )
        elif self._carbon_budget_policy.enabled and (not ecologits_ready or not self._patch_enabled):
            logger.warning(
                "Carbon circuit breaker is configured, but energy tracking is unavailable for this session."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._patch_enabled:
            self._runtime._remove_adapters()
        self._runtime._session_state.reset(self._token)

    @property
    def energy_kwh(self) -> float:
        if self._state is None:
            return 0.0
        return float(self._state.energy_kwh)

    @property
    def actual_gco2eq_so_far(self) -> float:
        if self._state is None:
            return 0.0
        return float(self._state.actual_gco2eq_so_far)

    @property
    def carbon_budget_exceeded(self) -> bool:
        if self._state is None:
            return False
        return bool(self._state.carbon_budget_exceeded)

    @property
    def llm_provider(self) -> str | None:
        if self._state is None or not self._state.llm_providers:
            return None
        if len(self._state.llm_providers) != 1:
            return None
        return next(iter(self._state.llm_providers))

    @property
    def model_usage(self) -> tuple[ModelUsageSummary, ...]:
        if self._state is None:
            return ()
        return tuple(
            ModelUsageSummary(
                requested_model=requested_model,
                effective_model=effective_model,
                call_count=call_count,
                downgraded=requested_model != effective_model,
            )
            for (requested_model, effective_model), call_count in self._state.model_usage_counts.items()
        )

    @property
    def session_metadata(self) -> dict[str, float | bool | str]:
        if self._state is None or not self._state.carbon_budget_policy.enabled:
            return {}

        metadata: dict[str, float | bool | str] = {
            "max_session_gco2eq": float(self._state.carbon_budget_policy.max_session_gco2eq or 0.0),
            "actual_gco2eq_so_far": float(self._state.actual_gco2eq_so_far),
            "carbon_budget_exceeded": bool(self._state.carbon_budget_exceeded),
        }
        if self._state.carbon_budget_exceeded:
            metadata["termination_reason"] = "carbon_budget_exceeded"
        return metadata


class EcoLogitsRuntime:
    """
    Coordinates process-wide adapter patching with session-scoped telemetry state.
    """

    def __init__(self, adapters: Sequence[TelemetryAdapter]):
        self._adapters = tuple(adapters)
        self._session_state = contextvars.ContextVar("_session_state", default=None)
        self._ecologits_init_lock = threading.Lock()
        self._ecologits_initialized = False
        self._warned_missing_ecologits = False
        self._warned_ecologits_init_failure = False
        self._patch_lock = threading.RLock()
        self._active_sessions = 0
        self._installed_adapters: tuple[TelemetryAdapter, ...] = ()
        self._session_hooks = _RuntimeSessionHooks(self)

    def session(
        self,
        *,
        carbon_budget_policy: CarbonBudgetPolicy | None = None,
        model_downgrade_policy: ModelDowngradePolicy | None = None,
    ) -> EcoTelemetrySession:
        return EcoTelemetrySession(
            self,
            carbon_budget_policy=carbon_budget_policy,
            model_downgrade_policy=model_downgrade_policy,
        )

    def _ensure_ecologits_initialized(self) -> bool:
        if self._ecologits_initialized:
            return True

        with self._ecologits_init_lock:
            if self._ecologits_initialized:
                return True

            try:
                from ecologits import EcoLogits
            except ImportError:
                if not self._warned_missing_ecologits:
                    logger.warning("EcoLogits not installed; energy tracking is disabled.")
                    self._warned_missing_ecologits = True
                return False

            try:
                EcoLogits.init(providers=self._provider_names())
            except Exception as exc:
                if not self._warned_ecologits_init_failure:
                    logger.warning("Failed to initialize EcoLogits: %s", exc)
                    self._warned_ecologits_init_failure = True
                return False

            self._ecologits_initialized = True
            logger.info(
                "EcoLogits initialized for telemetry providers: %s",
                ", ".join(self._provider_names()),
            )
            return True

    def _provider_names(self) -> list[str]:
        provider_names = []

        for adapter in self._adapters:
            if hasattr(adapter, "is_available") and not adapter.is_available():
                continue
            if adapter.provider_name not in provider_names:
                provider_names.append(adapter.provider_name)

        return provider_names

    def _install_adapters(self) -> bool:
        with self._patch_lock:
            if self._active_sessions > 0:
                self._active_sessions += 1
                return True

            if self._active_sessions == 0:
                installed_adapters = []

                for adapter in self._adapters:
                    try:
                        if adapter.install(self._session_hooks):
                            installed_adapters.append(adapter)
                    except Exception as exc:
                        logger.warning(
                            "Failed to install telemetry adapter '%s': %s",
                            adapter.provider_name,
                            exc,
                        )

                self._installed_adapters = tuple(installed_adapters)

            if not self._installed_adapters:
                return False

            self._active_sessions += 1
            return True

    def _remove_adapters(self) -> None:
        with self._patch_lock:
            self._active_sessions -= 1
            if self._active_sessions > 0:
                return

            for adapter in reversed(self._installed_adapters):
                try:
                    adapter.uninstall()
                except Exception as exc:
                    logger.warning(
                        "Failed to uninstall telemetry adapter '%s': %s",
                        adapter.provider_name,
                        exc,
                    )

            self._installed_adapters = ()

    def _record_energy(self, energy_kwh: float) -> None:
        state = self._session_state.get()
        if state is None:
            return

        try:
            normalized_energy_kwh = float(energy_kwh)
        except (TypeError, ValueError):
            logger.debug("Skipping non-numeric energy value: %r", energy_kwh)
            return

        state.energy_kwh += normalized_energy_kwh

        carbon_budget_policy = state.carbon_budget_policy
        if not carbon_budget_policy.is_enforced:
            return

        state.actual_gco2eq_so_far += (
            normalized_energy_kwh * carbon_budget_policy.actual_intensity_gco2eq_per_kwh
        )
        if state.actual_gco2eq_so_far <= float(carbon_budget_policy.max_session_gco2eq or 0.0):
            return

        state.carbon_budget_exceeded = True
        raise CarbonBudgetExceededError(
            max_session_gco2eq=float(carbon_budget_policy.max_session_gco2eq or 0.0),
            actual_gco2eq=state.actual_gco2eq_so_far,
            energy_kwh=state.energy_kwh,
        )

    def _record_llm_provider(self, provider_name: str) -> None:
        state = self._session_state.get()
        if state is None:
            return
        if not provider_name:
            return
        state.llm_providers.add(str(provider_name))

    def _get_model_downgrade_policy(self) -> ModelDowngradePolicy:
        state = self._session_state.get()
        if state is None:
            return _disabled_model_downgrade_policy()
        return state.model_downgrade_policy

    def _record_model_usage(
        self,
        requested_model: str | None,
        effective_model: str | None,
    ) -> None:
        state = self._session_state.get()
        if state is None:
            return

        resolved_requested_model = requested_model or effective_model
        resolved_effective_model = effective_model or requested_model
        if not resolved_requested_model or not resolved_effective_model:
            return

        key = (str(resolved_requested_model), str(resolved_effective_model))
        state.model_usage_counts[key] = state.model_usage_counts.get(key, 0) + 1

    def _should_warn_unmapped_model(self, requested_model: str) -> bool:
        state = self._session_state.get()
        if state is None:
            return False
        if requested_model in state.warned_unmapped_models:
            return False

        state.warned_unmapped_models.add(requested_model)
        return True
