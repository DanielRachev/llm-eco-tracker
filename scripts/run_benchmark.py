import asyncio
import csv
import json
import logging
import random
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from llm_eco_tracker.api import carbon_aware, _default_telemetry_runtime
from llm_eco_tracker.providers.csv_forecast import CsvForecastProvider
from llm_eco_tracker.emissions import summarize_emissions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MOCK_CSV = BASE_DIR / "tests" / "fixtures" / "mock_forecast.csv"
RESULTS_CSV = BASE_DIR / "experiment_results.csv"

ITERATIONS = 30
MAX_DELAY_HOURS = 4

# Reduce noise
logging.getLogger("llm_eco_tracker").setLevel(logging.ERROR)

# EcoLogits energy estimate for one GPT-4o-mini call (~550 tokens).
# Source: EcoLogits model card for GPT-4o-mini.
ENERGY_PER_CALL_KWH = 0.00001

async def dummy_batch_summarization(batch_size: int):
    """
    Simulates summarizing `batch_size` short text files using an LLM API.

    Instead of calling the real OpenAI API (which would require a valid key)
    or mocking it (which would bypass EcoLogits' tracking wrapper), we directly
    call the EcoLogits runtime's _record_energy() method — this is the exact
    same callback that OpenAIChatCompletionsAdapter uses internally when it
    intercepts a real API response.
    """
    for _ in range(batch_size):
        # Record energy for one simulated LLM call via EcoLogits runtime
        _default_telemetry_runtime._record_energy(ENERGY_PER_CALL_KWH)

@carbon_aware(max_delay_hours=MAX_DELAY_HOURS, forecast_provider=CsvForecastProvider(MOCK_CSV))
async def summarize_carbon_aware(batch_size: int):
    """Executes the batch summarization using the Carbon-Aware decorator."""
    await dummy_batch_summarization(batch_size)

async def summarize_baseline(batch_size: int):
    """Executes the batch summarization without any decorator (Immediate)."""
    # Simply running it without the decorator. However, we still want to use
    # EcoLogits to measure the energy consumed to ensure comparisons are fair!
    with _default_telemetry_runtime.session() as session:
        await dummy_batch_summarization(batch_size)
        return session.energy_kwh

async def mock_sleep(seconds: float):
    """Bypasses actual waiting during the benchmark to allow it to run quickly."""
    pass

def load_actual_intensity_mapping():
    """Reads the CSV to map start times to ACTUAL carbon intensity (Ground Truth)."""
    mapping = {}
    with MOCK_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["from"]] = float(row["intensity_actual"])
    return mapping

async def run_benchmark():
    parser = argparse.ArgumentParser(description="EcoTracker Benchmark")
    parser.add_argument("--start-offset", type=int, default=38, help="Row offset in mock_forecast.csv")
    args = parser.parse_args()

    print(f"Loading data from: {MOCK_CSV}")
    provider = CsvForecastProvider(MOCK_CSV)
    
    # We load the forecast from the mock CSV to determine our starting parameters
    snapshot_full = provider.load_forecast(max_delay_hours=0)
    actual_mapping = load_actual_intensity_mapping()
    
    start_offset = args.start_offset
    print(f"Starting {ITERATIONS} iterations (Offset: {start_offset})")
    
    # Get the "baseline" (immediate) time & intensity
    first_interval = snapshot_full.intervals[start_offset]
    first_start_str = first_interval.starts_at.strftime("%Y-%m-%dT%H:%MZ")
    baseline_actual_intensity = actual_mapping.get(first_start_str)
    
    results = []

    # Mock the load_forecast method to return the sliced snapshot
    original_load = CsvForecastProvider.load_forecast
    def mock_load_forecast(self, max_delay_hours: float):
        snap = original_load(self, max_delay_hours)
        from llm_eco_tracker.models import ForecastSnapshot
        return ForecastSnapshot(
            intervals=snap.intervals[start_offset:],
            reference_time=snap.intervals[start_offset].starts_at
        )

    patch_sleep = patch("llm_eco_tracker.execution.asyncio.sleep", side_effect=mock_sleep)
    patch_forecast = patch.object(CsvForecastProvider, "load_forecast", autospec=True, side_effect=mock_load_forecast)

    with patch_sleep, patch_forecast:
        for i in range(1, ITERATIONS + 1):
            print(f"Iteration {i}/{ITERATIONS}...", end="\r")
            
            # 1. Randomize batch size (varied workload)
            it_rng = random.Random(i * 17)
            batch_size = it_rng.randint(20, 100)
            
            # --- BASELINE (Without Decorator) ---
            energy_kwh = await summarize_baseline(batch_size)
            
            baseline_result = summarize_emissions(
                energy_kwh,
                baseline_actual_intensity,
                baseline_actual_intensity
            )
            
            results.append({
                "iteration": i,
                "batch_size": batch_size,
                "mode": "baseline",
                "energy_kwh": round(energy_kwh, 7),
                "scheduled_delay_h": 0.0,
                "actual_intensity_at_run": baseline_actual_intensity,
                "actual_gco2eq": round(baseline_result.actual_gco2eq, 6),
                "saved_gco2eq": 0.0,
            })
            
            # --- CARBON-AWARE (With Decorator) ---
            await summarize_carbon_aware(batch_size)
            
            last_record = None
            telemetry_path = Path("eco_telemetry.jsonl")
            if not telemetry_path.exists():
                telemetry_path.touch()
                
            with telemetry_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    last_record = json.loads(lines[-1])
                    
            if not last_record:
                continue

            aware_energy_kwh = last_record["energy_kwh"]
            delay_seconds = float(last_record["schedule_plan"]["execution_delay_seconds"])
            
            # Get actual intensity for the delayed time
            delayed_time = first_interval.starts_at + timedelta(seconds=delay_seconds)
            time_str = delayed_time.replace(minute=30 if delayed_time.minute >= 30 else 0, second=0).strftime("%Y-%m-%dT%H:%MZ")
            aware_actual_intensity = actual_mapping.get(time_str, baseline_actual_intensity)
            
            aware_result = summarize_emissions(
                aware_energy_kwh,
                baseline_actual_intensity,
                aware_actual_intensity
            )

            results.append({
                "iteration": i,
                "batch_size": batch_size,
                "mode": "carbon-aware",
                "energy_kwh": round(aware_energy_kwh, 7),
                "scheduled_delay_h": round(delay_seconds / 3600, 2),
                "actual_intensity_at_run": aware_actual_intensity,
                "actual_gco2eq": round(aware_result.actual_gco2eq, 6),
                "saved_gco2eq": round(aware_result.saved_gco2eq, 6),
            })

    # --- Save results ---
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # --- Executive Summary ---
    b_rows = [r for r in results if r["mode"] == "baseline"]
    a_rows = [r for r in results if r["mode"] == "carbon-aware"]

    total_b_co2 = sum(r["actual_gco2eq"] for r in b_rows)
    total_a_co2 = sum(r["actual_gco2eq"] for r in a_rows)
    
    pct = ((total_b_co2 - total_a_co2) / total_b_co2) * 100 if total_b_co2 > 0 else 0

    print()
    print("=" * 60)
    print("FINAL GRADE BENCHMARK SUMMARY (ECO-LOGITS NATIVE)")
    print("=" * 60)
    print(f"  Total Reduction:         {pct:.1f}%")
    print(f"  Baseline total:          {total_b_co2:.4f} gCO2eq")
    print(f"  Carbon-Aware total:      {total_a_co2:.4f} gCO2eq")
    print(f"  Saved gCO2eq:            {total_b_co2 - total_a_co2:.4f}")
    print(f"  Results recorded in:     {RESULTS_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
