from __future__ import annotations

import contextvars
import logging
import threading
from collections.abc import Sequence

from .base import TelemetryAdapter

logger = logging.getLogger(__name__)


class EcoTelemetrySession:
    def __init__(self, runtime: "EcoLogitsRuntime"):
        self._runtime = runtime
        self._token = None
        self._patch_enabled = False

    def __enter__(self):
        self._token = self._runtime._session_energy_kwh.set(0.0)
        self._runtime._ensure_ecologits_initialized()
        self._patch_enabled = self._runtime._install_adapters()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._patch_enabled:
            self._runtime._remove_adapters()
        self._runtime._session_energy_kwh.reset(self._token)

    @property
    def energy_kwh(self) -> float:
        return float(self._runtime._session_energy_kwh.get())


class EcoLogitsRuntime:
    """
    Coordinates process-wide adapter patching with session-scoped energy totals.
    """

    def __init__(self, adapters: Sequence[TelemetryAdapter]):
        self._adapters = tuple(adapters)
        self._session_energy_kwh = contextvars.ContextVar("session_energy_kwh", default=0.0)
        self._ecologits_init_lock = threading.Lock()
        self._ecologits_initialized = False
        self._warned_missing_ecologits = False
        self._warned_ecologits_init_failure = False
        self._patch_lock = threading.RLock()
        self._active_sessions = 0
        self._installed_adapters: tuple[TelemetryAdapter, ...] = ()

    def session(self) -> EcoTelemetrySession:
        return EcoTelemetrySession(self)

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
                        if adapter.install(self._record_energy):
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
        try:
            self._session_energy_kwh.set(self._session_energy_kwh.get() + float(energy_kwh))
        except (TypeError, ValueError):
            logger.debug("Skipping non-numeric energy value: %r", energy_kwh)
