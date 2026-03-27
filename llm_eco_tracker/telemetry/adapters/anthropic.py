from __future__ import annotations

import functools
import inspect
import logging

from ..base import TelemetrySessionHooks
from ._energy import normalize_energy_value

logger = logging.getLogger(__name__)


class AnthropicMessagesAdapter:
    provider_name = "anthropic"

    def __init__(self):
        self._original_sync_create = None
        self._original_async_create = None
        self._original_sync_descriptor = None
        self._original_async_descriptor = None
        self._sync_messages_cls = None
        self._async_messages_cls = None
        self._warned_missing_anthropic = False

    def install(self, session_hooks: TelemetrySessionHooks) -> bool:
        messages_cls, async_messages_cls = self._import_anthropic_classes()
        if messages_cls is None:
            return False

        self._sync_messages_cls = messages_cls
        self._async_messages_cls = async_messages_cls
        self._original_sync_descriptor = messages_cls.__dict__.get("create")
        self._original_sync_create = messages_cls.create
        messages_cls.create = self._build_sync_tracker(session_hooks)

        if async_messages_cls is not None:
            self._original_async_descriptor = async_messages_cls.__dict__.get("create")
            self._original_async_create = async_messages_cls.create
            async_messages_cls.create = self._build_async_tracker(session_hooks)

        return True

    def uninstall(self) -> None:
        if self._sync_messages_cls is not None and self._original_sync_descriptor is not None:
            self._sync_messages_cls.create = self._original_sync_descriptor
        if self._async_messages_cls is not None and self._original_async_descriptor is not None:
            self._async_messages_cls.create = self._original_async_descriptor

        self._original_sync_create = None
        self._original_async_create = None
        self._original_sync_descriptor = None
        self._original_async_descriptor = None
        self._sync_messages_cls = None
        self._async_messages_cls = None

    def is_available(self) -> bool:
        messages_cls, _ = self._import_anthropic_classes()
        return messages_cls is not None

    def _import_anthropic_classes(self):
        try:
            from anthropic.resources.messages import AsyncMessages, Messages
        except ImportError:
            if not self._warned_missing_anthropic:
                logger.warning("Anthropic SDK not installed; session telemetry is disabled.")
                self._warned_missing_anthropic = True
            return None, None

        return Messages, AsyncMessages

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
            session_hooks.record_llm_provider(self.provider_name)
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
            session_hooks.record_llm_provider(self.provider_name)
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

    def _resolve_model(self, requested_model, session_hooks: TelemetrySessionHooks):
        if not isinstance(requested_model, str):
            return requested_model

        policy = session_hooks.get_model_downgrade_policy()
        if not policy.is_dirty:
            return requested_model

        fallback_model = policy.fallback_map.get(requested_model)
        if fallback_model is not None:
            logger.info(
                "Dirty-grid downgrade applied for Anthropic model '%s' -> '%s' (%.1f >= %.1f gCO2eq/kWh).",
                requested_model,
                fallback_model,
                policy.execution_intensity_gco2eq_per_kwh,
                policy.dirty_threshold_gco2eq_per_kwh,
            )
            return fallback_model

        if session_hooks.should_warn_unmapped_model(requested_model):
            logger.warning(
                "Dirty-grid downgrade is enabled, but no fallback is configured for Anthropic model '%s'.",
                requested_model,
            )
        return requested_model

    @staticmethod
    def _record_response_energy(response, session_hooks: TelemetrySessionHooks) -> None:
        impacts = getattr(response, "impacts", None)
        energy = getattr(getattr(impacts, "energy", None), "value", None)
        normalized_energy = normalize_energy_value(energy)
        if normalized_energy is None:
            return

        session_hooks.record_energy_kwh(normalized_energy)
