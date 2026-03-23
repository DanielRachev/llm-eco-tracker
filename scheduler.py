import logging
import requests
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _fetch_live_forecast(max_delay_hours):
    """
    Fetches the carbon intensity forecast for the next `max_delay_hours`.
    """
    if max_delay_hours <= 0:
        return None, []

    # The UK API requires UTC time
    now = datetime.now(timezone.utc)
    end_time = now + timedelta(hours=max_delay_hours)

    # Format to ISO8601 as required by the API (e.g., 2026-03-23T22:00Z)
    start_str = now.strftime("%Y-%m-%dT%H:%MZ")
    end_str = end_time.strftime("%Y-%m-%dT%H:%MZ")

    url = f"https://api.carbonintensity.org.uk/intensity/{start_str}/{end_str}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json().get("data", [])

        if not data:
            logger.warning(
                "No forecast data returned. Defaulting to immediate execution."
            )
            return now, []

        return now, data

    except Exception as e:
        # FAIL-SAFE: If the API goes down, NEVER break the developer's app!
        logger.error(f"Failed to fetch grid data: {e}. Executing immediately.")
        return now, []


def _get_live_schedule_plan(max_delay_hours):
    """
    Returns a tuple of:
    - current grid intensity in gCO2eq/kWh
    - delay in seconds until the greenest block in the forecast window
    - optimal grid intensity in gCO2eq/kWh
    """
    now, data = _fetch_live_forecast(max_delay_hours)

    if not data:
        return 0.0, 0.0, 0.0

    # Find the 30-minute block with the lowest forecasted carbon intensity
    best_block = min(data, key=lambda x: x["intensity"]["forecast"])
    best_time_str = best_block["from"]
    baseline_intensity = float(data[0]["intensity"]["forecast"])
    optimal_intensity = float(best_block["intensity"]["forecast"])

    logger.info(f"Current forecast: {baseline_intensity} gCO2eq/kWh")
    logger.info(
        f"Best forecast found: {optimal_intensity} gCO2eq/kWh at {best_time_str}"
    )

    # Calculate the difference in seconds
    best_time = datetime.strptime(best_time_str, "%Y-%m-%dT%H:%MZ")
    best_time = best_time.replace(tzinfo=timezone.utc)
    delay_seconds = max(0.0, (best_time - now).total_seconds())

    return baseline_intensity, delay_seconds, optimal_intensity


def _get_live_delay_seconds(max_delay_hours):
    _, delay_seconds, _ = _get_live_schedule_plan(max_delay_hours)
    return int(delay_seconds)
