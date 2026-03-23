import asyncio
import contextvars
import csv
import functools
import inspect
import json
import logging
import random
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scheduler import _get_live_schedule_plan


logger = logging.getLogger(__name__)

_session_energy_kwh = contextvars.ContextVar("session_energy_kwh", default=0.0)
_telemetry_path = Path("eco_telemetry.jsonl")


class EcoTelemetrySession:
    """
    Tracks EcoLogits energy impacts across all OpenAI chat completion calls
    made inside a decorated function.

    The OpenAI method patch is process-wide, so a lock and reference count are
    used to avoid restoring the original methods while another decorated
    session is still running.
    """

    _patch_lock = threading.RLock()
    _active_sessions = 0
    _original_sync_create = None
    _original_async_create = None
    _sync_completions_cls = None
    _async_completions_cls = None
    _warned_missing_openai = False

    def __enter__(self):
        self._token = _session_energy_kwh.set(0.0)
        self._patch_enabled = self._install_patches()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._patch_enabled:
            self._remove_patches()
        _session_energy_kwh.reset(self._token)

    @classmethod
    def _install_patches(cls):
        with cls._patch_lock:
            if cls._active_sessions == 0:
                if not cls._patch_openai_methods():
                    return False
            cls._active_sessions += 1
            return True

    @classmethod
    def _remove_patches(cls):
        with cls._patch_lock:
            cls._active_sessions -= 1
            if cls._active_sessions > 0:
                return

            if cls._sync_completions_cls and cls._original_sync_create:
                cls._sync_completions_cls.create = cls._original_sync_create
            if cls._async_completions_cls and cls._original_async_create:
                cls._async_completions_cls.create = cls._original_async_create

            cls._original_sync_create = None
            cls._original_async_create = None
            cls._sync_completions_cls = None
            cls._async_completions_cls = None

    @classmethod
    def _patch_openai_methods(cls):
        completions_cls, async_completions_cls = cls._import_openai_classes()
        if completions_cls is None:
            return False

        cls._sync_completions_cls = completions_cls
        cls._async_completions_cls = async_completions_cls
        cls._original_sync_create = completions_cls.create
        completions_cls.create = cls._build_sync_tracker()

        if async_completions_cls is not None:
            cls._original_async_create = async_completions_cls.create
            async_completions_cls.create = cls._build_async_tracker()

        return True

    @classmethod
    def _import_openai_classes(cls):
        try:
            from openai.resources.chat.completions import Completions
        except ImportError:
            if not cls._warned_missing_openai:
                logger.warning("OpenAI SDK not installed; session telemetry is disabled.")
                cls._warned_missing_openai = True
            return None, None

        try:
            from openai.resources.chat.completions import AsyncCompletions
        except ImportError:
            AsyncCompletions = None

        return Completions, AsyncCompletions

    @classmethod
    def _build_sync_tracker(cls):
        @functools.wraps(cls._original_sync_create)
        def tracking_create(*args, **kwargs):
            response = cls._original_sync_create(*args, **kwargs)
            cls._record_energy(response)
            return response

        return tracking_create

    @classmethod
    def _build_async_tracker(cls):
        @functools.wraps(cls._original_async_create)
        async def tracking_create(*args, **kwargs):
            response = await cls._original_async_create(*args, **kwargs)
            cls._record_energy(response)
            return response

        return tracking_create

    @classmethod
    def _record_energy(cls, response):
        impacts = getattr(response, "impacts", None)
        energy = getattr(getattr(impacts, "energy", None), "value", None)
        if energy is None:
            return

        try:
            _session_energy_kwh.set(_session_energy_kwh.get() + float(energy))
        except (TypeError, ValueError):
            logger.debug("Skipping non-numeric energy value: %r", energy)


def apply_jitter(delay_seconds):
    """
    Randomly spreads delayed executions to avoid many jobs resuming at the
    exact same timestamp.
    """
    if delay_seconds <= 0:
        return 0.0

    jitter_window = delay_seconds * 0.10
    jitter = random.uniform(-jitter_window, jitter_window)
    return max(0.0, delay_seconds + jitter)


def save_telemetry(baseline_emissions, actual_emissions):
    """
    Persists telemetry as newline-delimited JSON so each invocation appends a
    single record without rewriting the full file.
    """
    savings = baseline_emissions - actual_emissions
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_gco2eq": baseline_emissions,
        "actual_gco2eq": actual_emissions,
        "saved_gco2eq": savings,
    }

    try:
        with _telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning("Failed to write telemetry file '%s': %s", _telemetry_path, exc)


def _parse_utc_timestamp(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)


def _load_mock_forecast(mock_csv):
    mock_path = Path(mock_csv)

    try:
        with mock_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                rows.append(
                    {
                        "from": _parse_utc_timestamp(row["from"]),
                        "to": _parse_utc_timestamp(row["to"]),
                        "forecast": float(row["intensity_forecast"]),
                    }
                )
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Failed to read mock forecast data from '%s': %s", mock_csv, exc)
        return []

    return rows


def _get_mock_schedule_plan(max_delay_hours, mock_csv):
    rows = _load_mock_forecast(mock_csv)
    if not rows:
        return 0.0, 0.0, 0.0

    current_block = rows[0]
    baseline_intensity = current_block["forecast"]
    search_deadline = current_block["from"] + timedelta(hours=max_delay_hours)

    candidate_rows = [
        row for row in rows if row["from"] <= search_deadline
    ] or [current_block]

    best_block = min(candidate_rows, key=lambda row: row["forecast"])
    delay_seconds = max(0.0, (best_block["from"] - current_block["from"]).total_seconds())
    optimal_intensity = best_block["forecast"]

    logger.info("Mock current forecast: %.1f gCO2eq/kWh", baseline_intensity)
    logger.info(
        "Mock best forecast found: %.1f gCO2eq/kWh at %s",
        optimal_intensity,
        best_block["from"].isoformat(),
    )

    return baseline_intensity, delay_seconds, optimal_intensity


def _get_schedule_plan(max_delay_hours, location, mock_csv):
    if max_delay_hours <= 0:
        return 0.0, 0.0, 0.0

    if mock_csv:
        logger.info("Mock mode enabled: reading forecast data from '%s'.", mock_csv)
        return _get_mock_schedule_plan(max_delay_hours, mock_csv)

    if location.upper() not in {"UK", "GB"}:
        logger.warning(
            "Live scheduling currently uses the UK Carbon Intensity API. "
            "Location '%s' is not applied yet.",
            location,
        )

    logger.info("Live mode enabled: fetching grid forecast.")
    return _get_live_schedule_plan(max_delay_hours)


def _get_session_energy():
    return float(_session_energy_kwh.get())


def _log_telemetry_summary(total_kwh, baseline_intensity, optimal_intensity):
    if total_kwh <= 0:
        logger.info("Session complete. No EcoLogits energy impacts were captured.")
        return

    baseline_gco2eq = total_kwh * baseline_intensity
    actual_gco2eq = total_kwh * optimal_intensity
    saved_gco2eq = baseline_gco2eq - actual_gco2eq

    logger.info("Session complete. Total energy: %.6f kWh", total_kwh)
    logger.info("Carbon delta: %.4f gCO2eq", saved_gco2eq)
    save_telemetry(baseline_gco2eq, actual_gco2eq)


def carbon_aware(max_delay_hours=2, location="NL", mock_csv=None):
    """
    Delay non-urgent work until a greener grid window, then record session-wide
    energy telemetry for OpenAI chat completions executed inside the function.

    Args:
        max_delay_hours (int): Maximum time to wait for a greener grid window.
        location (str): Region hint for future scheduler backends.
        mock_csv (str | None): Optional CSV path for deterministic mock forecasts.
    """

    def _log_intercept(func):
        logger.info("Intercepting call to '%s'", func.__name__)
        logger.info("Target location: %s, max delay: %sh", location, max_delay_hours)
        if mock_csv:
            logger.info("Using mock forecast data from: %s", mock_csv)

    def _build_delay_plan():
        baseline_intensity, raw_delay, optimal_intensity = _get_schedule_plan(
            max_delay_hours, location, mock_csv
        )
        final_delay = apply_jitter(raw_delay)

        logger.info(
            "Scheduling plan: baseline %.1f gCO2eq/kWh, optimal %.1f gCO2eq/kWh, raw delay %.2fs, jittered delay %.2fs",
            baseline_intensity,
            optimal_intensity,
            raw_delay,
            final_delay,
        )

        return baseline_intensity, optimal_intensity, final_delay

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                _log_intercept(func)
                baseline_intensity, optimal_intensity, final_delay = _build_delay_plan()

                if final_delay > 0:
                    logger.info(
                        "Carbon-aware scheduler: Awaiting %.2f seconds to reach a greener grid window.",
                        final_delay,
                    )
                    await asyncio.sleep(final_delay)
                else:
                    logger.info("Carbon-aware scheduler: Executing immediately.")

                logger.info("Greener window reached. Proceeding with execution.")

                with EcoTelemetrySession():
                    try:
                        result = await func(*args, **kwargs)
                    finally:
                        total_kwh = _get_session_energy()

                _log_telemetry_summary(total_kwh, baseline_intensity, optimal_intensity)
                return result

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _log_intercept(func)
            baseline_intensity, optimal_intensity, final_delay = _build_delay_plan()

            done = threading.Event()
            outcome = {}

            def run_later():
                try:
                    logger.info("Greener window reached. Proceeding with execution.")
                    with EcoTelemetrySession():
                        try:
                            outcome["result"] = func(*args, **kwargs)
                        finally:
                            total_kwh = _get_session_energy()
                    _log_telemetry_summary(total_kwh, baseline_intensity, optimal_intensity)
                except BaseException as exc:
                    outcome["exception"] = exc
                finally:
                    done.set()

            if final_delay > 0:
                logger.info(
                    "Carbon-aware scheduler: Scheduling execution in %.2f seconds.",
                    final_delay,
                )
                timer = threading.Timer(final_delay, run_later)
                timer.start()
            else:
                logger.info("Carbon-aware scheduler: Executing immediately.")
                timer = None
                run_later()

            try:
                done.wait()
            except KeyboardInterrupt:
                if timer is not None:
                    timer.cancel()
                raise

            if "exception" in outcome:
                raise outcome["exception"]

            return outcome.get("result")

        return sync_wrapper

    return decorator
