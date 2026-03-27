from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_eco_tracker.providers import ElectricityMapsProvider


class ElectricityMapsProviderTests(unittest.TestCase):
    def test_load_forecast_with_zone_builds_expected_request_and_intervals(self):
        provider = ElectricityMapsProvider(
            zone="DE",
            auth_token="test-token",
            temporal_granularity="hourly",
        )

        captured_request: dict[str, object] = {}

        def fake_get(url, *, params, headers, timeout):
            captured_request["url"] = url
            captured_request["params"] = params
            captured_request["headers"] = headers
            captured_request["timeout"] = timeout
            return _FakeResponse(
                [
                    {"datetime": "2026-03-27T10:00:00Z", "carbonIntensity": 210},
                    {"datetime": "2026-03-27T11:00:00Z", "carbonIntensity": 180},
                ]
            )

        with patch("llm_eco_tracker.providers.electricity_maps.requests.get", side_effect=fake_get):
            snapshot = provider.load_forecast(max_delay_hours=4)

        self.assertEqual(captured_request["url"], "https://api.electricitymap.org/v3/carbon-intensity/forecast")
        self.assertEqual(captured_request["headers"], {"auth-token": "test-token"})
        self.assertEqual(captured_request["timeout"], 5.0)
        self.assertEqual(captured_request["params"]["zone"], "DE")
        self.assertEqual(captured_request["params"]["horizonHours"], 24)
        self.assertEqual(captured_request["params"]["temporalGranularity"], "hourly")
        self.assertEqual(len(snapshot.intervals), 2)
        self.assertEqual(snapshot.reference_time, snapshot.intervals[0].starts_at)
        self.assertEqual(snapshot.intervals[0].carbon_intensity_gco2eq_per_kwh, 210.0)
        self.assertEqual(snapshot.intervals[0].ends_at, snapshot.intervals[1].starts_at)
        self.assertEqual(
            (snapshot.intervals[1].ends_at - snapshot.intervals[1].starts_at).total_seconds(),
            3600.0,
        )

    def test_load_forecast_supports_lat_lon_and_nested_data_payload(self):
        provider = ElectricityMapsProvider(
            lat=52.52,
            lon=13.40,
            auth_token="test-token",
            temporal_granularity="15_minutes",
            disable_estimations=True,
        )

        captured_request: dict[str, object] = {}

        def fake_get(url, *, params, headers, timeout):
            del url, headers, timeout
            captured_request["params"] = params
            return _FakeResponse(
                {
                    "data": [
                        {"datetime": "2026-03-27T10:00:00+00:00", "carbonIntensity": {"value": 240}},
                        {"datetime": "2026-03-27T10:15:00+00:00", "carbonIntensity": {"value": 225}},
                    ]
                }
            )

        with patch("llm_eco_tracker.providers.electricity_maps.requests.get", side_effect=fake_get):
            snapshot = provider.load_forecast(max_delay_hours=30)

        self.assertEqual(captured_request["params"]["lat"], 52.52)
        self.assertEqual(captured_request["params"]["lon"], 13.4)
        self.assertEqual(captured_request["params"]["disableEstimations"], "true")
        self.assertEqual(captured_request["params"]["horizonHours"], 48)
        self.assertEqual(len(snapshot.intervals), 2)
        self.assertEqual(snapshot.intervals[1].carbon_intensity_gco2eq_per_kwh, 225.0)
        self.assertEqual(
            (snapshot.intervals[1].ends_at - snapshot.intervals[1].starts_at).total_seconds(),
            900.0,
        )

    def test_load_forecast_returns_empty_snapshot_when_auth_token_is_missing(self):
        provider = ElectricityMapsProvider(zone="DE", auth_token=None)

        with self.assertLogs("llm_eco_tracker.providers.electricity_maps", level="ERROR") as captured:
            with patch("llm_eco_tracker.providers.electricity_maps.requests.get") as mocked_get:
                snapshot = provider.load_forecast(max_delay_hours=4)

        mocked_get.assert_not_called()
        self.assertEqual(snapshot.intervals, ())
        self.assertEqual(len(captured.output), 1)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


if __name__ == "__main__":
    unittest.main()
