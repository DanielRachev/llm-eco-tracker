import logging
from datetime import datetime, timedelta, timezone

import requests

from .models import ForecastInterval
from .planning import build_schedule_plan, immediate_schedule_plan

logger = logging.getLogger(__name__)


def _fetch_live_forecast(max_delay_hours):
    """
    Fetches the carbon intensity forecast for the next `max_delay_hours`.
    """
    if max_delay_hours <= 0:
        return None, []

    # The UK API requires UTC time.
    now = datetime.now(timezone.utc)
    end_time = now + timedelta(hours=max_delay_hours)

    # Format to ISO8601 as required by the API (e.g., 2026-03-23T22:00Z).
    start_str = now.strftime("%Y-%m-%dT%H:%MZ")
    end_str = end_time.strftime("%Y-%m-%dT%H:%MZ")

    url = f"https://api.carbonintensity.org.uk/intensity/{start_str}/{end_str}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json().get("data", [])

        if not data:
            logger.warning("No forecast data returned. Defaulting to immediate execution.")
            return now, []

        return now, data

    except Exception as exc:
        # FAIL-SAFE: If the API goes down, NEVER break the developer's app!
        logger.error("Failed to fetch grid data: %s. Executing immediately.", exc)
        return now, []


def _parse_live_forecast(data):
    forecast_intervals = []

    for entry in data:
        try:
            forecast_intervals.append(
                ForecastInterval(
                    starts_at=datetime.strptime(entry["from"], "%Y-%m-%dT%H:%MZ").replace(
                        tzinfo=timezone.utc
                    ),
                    ends_at=datetime.strptime(entry["to"], "%Y-%m-%dT%H:%MZ").replace(
                        tzinfo=timezone.utc
                    ),
                    carbon_intensity_gco2eq_per_kwh=float(entry["intensity"]["forecast"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed live forecast row: %s", exc)

    return forecast_intervals


def _get_live_schedule_plan(max_delay_hours):
    now, data = _fetch_live_forecast(max_delay_hours)
    forecast_intervals = _parse_live_forecast(data)

    if not forecast_intervals:
        return immediate_schedule_plan()

    schedule_plan = build_schedule_plan(
        forecast_intervals,
        max_delay_hours,
        reference_time=now,
    )

    logger.info(
        "Current forecast: %.1f gCO2eq/kWh",
        schedule_plan.baseline_intensity_gco2eq_per_kwh,
    )
    logger.info(
        "Best forecast found: %.1f gCO2eq/kWh at %s",
        schedule_plan.optimal_intensity_gco2eq_per_kwh,
        schedule_plan.selected_interval.starts_at.isoformat(),
    )

    return schedule_plan


def _get_live_delay_seconds(max_delay_hours):
    schedule_plan = _get_live_schedule_plan(max_delay_hours)
    return int(schedule_plan.execution_delay_seconds)
