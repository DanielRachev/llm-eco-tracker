from __future__ import annotations

import functools
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class OpenAIChatCompletionsAdapter:
    provider_name = "openai"

    def __init__(self):
        self._original_sync_create = None
        self._original_async_create = None
        self._sync_completions_cls = None
        self._async_completions_cls = None
        self._warned_missing_openai = False

    def install(self, record_energy_kwh: Callable[[float], None]) -> bool:
        completions_cls, async_completions_cls = self._import_openai_classes()
        if completions_cls is None:
            return False

        self._sync_completions_cls = completions_cls
        self._async_completions_cls = async_completions_cls
        self._original_sync_create = completions_cls.create
        completions_cls.create = self._build_sync_tracker(record_energy_kwh)

        if async_completions_cls is not None:
            self._original_async_create = async_completions_cls.create
            async_completions_cls.create = self._build_async_tracker(record_energy_kwh)

        return True

    def uninstall(self) -> None:
        if self._sync_completions_cls is not None and self._original_sync_create is not None:
            self._sync_completions_cls.create = self._original_sync_create
        if self._async_completions_cls is not None and self._original_async_create is not None:
            self._async_completions_cls.create = self._original_async_create

        self._original_sync_create = None
        self._original_async_create = None
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

    def _build_sync_tracker(self, record_energy_kwh: Callable[[float], None]):
        original_sync_create = self._original_sync_create

        @functools.wraps(original_sync_create)
        def tracking_create(*args, **kwargs):
            response = original_sync_create(*args, **kwargs)
            self._record_response_energy(response, record_energy_kwh)
            return response

        return tracking_create

    def _build_async_tracker(self, record_energy_kwh: Callable[[float], None]):
        original_async_create = self._original_async_create

        @functools.wraps(original_async_create)
        async def tracking_create(*args, **kwargs):
            response = await original_async_create(*args, **kwargs)
            self._record_response_energy(response, record_energy_kwh)
            return response

        return tracking_create

    @staticmethod
    def _record_response_energy(response, record_energy_kwh: Callable[[float], None]) -> None:
        impacts = getattr(response, "impacts", None)
        energy = getattr(getattr(impacts, "energy", None), "value", None)
        if energy is None:
            return

        record_energy_kwh(energy)
