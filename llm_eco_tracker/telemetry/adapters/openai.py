from __future__ import annotations

import functools
import inspect
import logging
from numbers import Real

from ..base import TelemetrySessionHooks

logger = logging.getLogger(__name__)


class OpenAIChatCompletionsAdapter:
    provider_name = "openai"

    def __init__(self):
        self._original_sync_create = None
        self._original_async_create = None
        self._original_sync_descriptor = None
        self._original_async_descriptor = None
        self._sync_completions_cls = None
        self._async_completions_cls = None
        self._warned_missing_openai = False

    def install(self, session_hooks: TelemetrySessionHooks) -> bool:
        completions_cls, async_completions_cls = self._import_openai_classes()
        if completions_cls is None:
            return False

        self._sync_completions_cls = completions_cls
        self._async_completions_cls = async_completions_cls
        self._original_sync_descriptor = completions_cls.__dict__.get("create")
        self._original_sync_create = completions_cls.create
        completions_cls.create = self._build_sync_tracker(session_hooks)

        if async_completions_cls is not None:
            self._original_async_descriptor = async_completions_cls.__dict__.get("create")
            self._original_async_create = async_completions_cls.create
            async_completions_cls.create = self._build_async_tracker(session_hooks)

        return True

    def uninstall(self) -> None:
        if self._sync_completions_cls is not None and self._original_sync_descriptor is not None:
            self._sync_completions_cls.create = self._original_sync_descriptor
        if self._async_completions_cls is not None and self._original_async_descriptor is not None:
            self._async_completions_cls.create = self._original_async_descriptor

        self._original_sync_create = None
        self._original_async_create = None
        self._original_sync_descriptor = None
        self._original_async_descriptor = None
        self._sync_completions_cls = None
        self._async_completions_cls = None

    def _import_openai_classes(self):
        try:
            from openai.resources.chat.completions import Completions
        except ImportError:
            if not self._warned_missing_openai:
                logger.warning("OpenAI SDK not installed; session telemetry is disabled.")
                self._warned_missing_openai = True
            return None, None

        try:
            from openai.resources.chat.completions import AsyncCompletions
        except ImportError:
            AsyncCompletions = None

        return Completions, AsyncCompletions

    def _build_sync_tracker(self, session_hooks: TelemetrySessionHooks):
        original_sync_create = self._original_sync_create
        create_signature = inspect.signature(original_sync_create)

        @functools.wraps(original_sync_create)
        def tracking_create(*args, **kwargs):
            requested_model, effective_model, bound_args, bound_kwargs = self._bind_request(
                create_signature,
                args,
                kwargs,
                session_hooks,
            )
            response = original_sync_create(*bound_args, **bound_kwargs)
            session_hooks.record_model_usage(requested_model, effective_model)
            self._record_response_energy(response, session_hooks)
            return response

        return tracking_create

    def _build_async_tracker(self, session_hooks: TelemetrySessionHooks):
        original_async_create = self._original_async_create
        create_signature = inspect.signature(original_async_create)

        @functools.wraps(original_async_create)
        async def tracking_create(*args, **kwargs):
            requested_model, effective_model, bound_args, bound_kwargs = self._bind_request(
                create_signature,
                args,
                kwargs,
                session_hooks,
            )
            response = await original_async_create(*bound_args, **bound_kwargs)
            session_hooks.record_model_usage(requested_model, effective_model)
            self._record_response_energy(response, session_hooks)
            return response

        return tracking_create

    def _bind_request(self, create_signature, args, kwargs, session_hooks: TelemetrySessionHooks):
        bound_arguments = create_signature.bind_partial(*args, **kwargs)
        requested_model = bound_arguments.arguments.get("model")
        effective_model = self._resolve_model(requested_model, session_hooks)
        if "model" in bound_arguments.arguments:
            bound_arguments.arguments["model"] = effective_model
        return requested_model, effective_model, bound_arguments.args, bound_arguments.kwargs

    def _resolve_model(
        self,
        requested_model,
        session_hooks: TelemetrySessionHooks,
    ):
        if not isinstance(requested_model, str):
            return requested_model

        policy = session_hooks.get_model_downgrade_policy()
        if not policy.is_dirty:
            return requested_model

        fallback_model = policy.fallback_map.get(requested_model)
        if fallback_model is not None:
            logger.info(
                "Dirty-grid downgrade applied for OpenAI model '%s' -> '%s' (%.1f >= %.1f gCO2eq/kWh).",
                requested_model,
                fallback_model,
                policy.execution_intensity_gco2eq_per_kwh,
                policy.dirty_threshold_gco2eq_per_kwh,
            )
            return fallback_model

        if session_hooks.should_warn_unmapped_model(requested_model):
            logger.warning(
                "Dirty-grid downgrade is enabled, but no fallback is configured for OpenAI model '%s'.",
                requested_model,
            )
        return requested_model

    @staticmethod
    def _record_response_energy(response, session_hooks: TelemetrySessionHooks) -> None:
        impacts = getattr(response, "impacts", None)
        energy = getattr(getattr(impacts, "energy", None), "value", None)
        normalized_energy = OpenAIChatCompletionsAdapter._normalize_energy_value(energy)
        if normalized_energy is None:
            return

        session_hooks.record_energy_kwh(normalized_energy)

    @staticmethod
    def _normalize_energy_value(energy) -> float | None:
        if isinstance(energy, Real):
            return float(energy)

        mean_value = getattr(energy, "mean", None)
        if isinstance(mean_value, Real):
            # EcoLogits reports a min/max range for some models; use its midpoint
            # so session totals remain comparable to other scalar energy sources.
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
