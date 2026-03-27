# llm-eco-tracker

`llm-eco-tracker` is a Python library for carbon-aware LLM execution. It wraps your application code with a `@carbon_aware` decorator, delays non-urgent work into greener grid windows, captures session energy through EcoLogits, and writes telemetry you can analyze later.

It is designed for normal application code, scheduled batch jobs, and agentic workflows built on top of OpenAI- and Anthropic-backed stacks such as LangChain.

## Why It Exists

Most LLM applications run immediately, even when the grid is unusually carbon-intensive. `llm-eco-tracker` gives you a lightweight software layer that can:

- delay flexible work until a cleaner grid window
- record baseline vs actual carbon emissions
- downgrade to smaller models on dirty grids
- stop a run when a per-session carbon budget is exceeded
- plug into existing OpenAI and Anthropic SDK usage without infrastructure changes

## Features

- Carbon-aware scheduling with configurable delay budgets
- Forecast providers for UK Carbon Intensity, Electricity Maps, and deterministic CSV traces
- Telemetry adapters for OpenAI chat completions and Anthropic messages
- Per-session model usage summaries
- Dirty-grid eco-fallbacks / model downgrades
- Carbon circuit breaker / budget enforcement
- JSONL, logger, composite, and no-op telemetry sinks
- Benchmark, analysis, and figure-generation scripts for evaluation

## Installation

### From Source

```bash
pip install -r requirements.txt
pip install -e .
```

### From PyPI

```bash
pip install llm-eco-tracker
```

### Optional Extras

```bash
pip install "llm-eco-tracker[langchain]"
pip install "llm-eco-tracker[benchmarks]"
pip install "llm-eco-tracker[dev]"
```

## Quickstart

```python
from openai import OpenAI

from llm_eco_tracker import carbon_aware

client = OpenAI()

@carbon_aware(max_delay_hours=2)
def summarize(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content
```

By default, telemetry is written to `./eco_telemetry.jsonl`.

## Provider Support

### Forecast Providers

- `UKCarbonIntensityProvider`
- `ElectricityMapsProvider`
- `CsvForecastProvider`

### Telemetry Adapters

- OpenAI `client.chat.completions.create(...)`
- Anthropic `client.messages.create(...)`

## LangChain / Agentic Workflows

`llm-eco-tracker` works with LangChain-style workflows when the workflow itself is orchestrated by LangChain and the actual model calls inside that workflow go through supported SDKs such as OpenAI or Anthropic. That is the safest integration pattern today, and it works well for agent loops, planners, critics, and multi-step chains.

The repository includes a runnable mocked demo at [langchain_agentic_demo.py](scripts/langchain_agentic_demo.py). It uses LangChain runnables to orchestrate a multi-step workflow while the underlying OpenAI SDK calls are intercepted by `@carbon_aware`.

## Forecast Provider Examples

### Deterministic CSV Trace

```python
from llm_eco_tracker import carbon_aware
from llm_eco_tracker.providers import CsvForecastProvider

@carbon_aware(
    max_delay_hours=2,
    forecast_provider=CsvForecastProvider("tests/fixtures/mock_forecast.csv"),
)
def run_batch_job():
    ...
```

### Electricity Maps

```python
import os

from llm_eco_tracker import carbon_aware
from llm_eco_tracker.providers import ElectricityMapsProvider

electricity_maps = ElectricityMapsProvider(
    zone="DE",
    auth_token=os.environ["ELECTRICITY_MAPS_API_TOKEN"],
)

@carbon_aware(
    max_delay_hours=2,
    forecast_provider=electricity_maps,
)
def run_with_electricity_maps():
    ...
```

## Eco-Fallbacks and Carbon Budgets

### Dirty-Grid Model Downgrade

```python
from llm_eco_tracker import carbon_aware

@carbon_aware(
    max_delay_hours=2,
    auto_downgrade=True,
    dirty_threshold=300.0,
    model_fallbacks={"gpt-4.1": "gpt-4.1-mini"},
)
def run_with_fallback():
    ...
```

### Circuit Breaker

```python
from llm_eco_tracker import CarbonBudgetExceededError, carbon_aware

@carbon_aware(max_session_gco2eq=5.0)
def run_budgeted_workflow():
    ...

try:
    run_budgeted_workflow()
except CarbonBudgetExceededError as exc:
    print(exc.actual_gco2eq, exc.max_session_gco2eq)
```

## Telemetry

By default, each decorated run emits a normalized telemetry record containing:

- timestamp
- captured energy in `kWh`
- baseline and actual emissions in `gCO2eq`
- saved carbon
- selected schedule plan
- forecast provider
- LLM provider
- effective model
- per-session model usage summary

The built-in telemetry sinks are:

- `JsonlTelemetrySink`
- `LoggerTelemetrySink`
- `CompositeTelemetrySink`
- `NoOpTelemetrySink`

## CLI

The package ships a telemetry report CLI:

```bash
ecotracker-report
python -m llm_eco_tracker.report
```

You can also point it at custom telemetry inputs:

```bash
ecotracker-report path/to/eco_telemetry.jsonl
python -m llm_eco_tracker.report app.log --format logger
```

## Demo Scripts

These scripts are designed to be screenshot-friendly and work without paid API traffic.

- [langchain_agentic_demo.py](scripts/langchain_agentic_demo.py)
  Runs a mocked LangChain workflow that makes multiple OpenAI calls inside one decorated function.
- [eco_fallback_demo.py](scripts/eco_fallback_demo.py)
  Demonstrates dirty-grid model downgrading with a deterministic CSV forecast.
- [circuit_breaker_demo.py](scripts/circuit_breaker_demo.py)
  Demonstrates the carbon budget / circuit breaker aborting a session.

Run them from the repository root:

```bash
python scripts/langchain_agentic_demo.py
python scripts/eco_fallback_demo.py
python scripts/circuit_breaker_demo.py
```

## Evaluation Pipeline

The repository includes a full benchmark and analysis pipeline:

```bash
python scripts/run_benchmark.py
python scripts/analyze_benchmark_results.py
python scripts/run_openai_integration_benchmark.py
python scripts/run_overhead_benchmark.py
python scripts/generate_paper_figures.py
```

Generated outputs include:

- `scenario_results.csv`
- `daily_summary.csv`
- `benchmark_summary.json`
- `benchmark_analysis.json`
- `benchmark_analysis.md`
- `paper_figures/`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
