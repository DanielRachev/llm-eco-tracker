# llm-eco-tracker

`llm-eco-tracker` is a lightweight Python library designed to make LLM usage more sustainable. It provides a carbon-aware decorator that can delay non-urgent work until a greener grid window and record telemetry for supported LLM calls.

## Features

- **Carbon-Aware Scheduling**: Delay execution based on forecasted grid carbon intensity.
- **Forecast Providers**: Use the live UK Carbon Intensity API or inject a CSV-backed provider for deterministic runs.
- **Telemetry Adapters**: Current telemetry support is wired for OpenAI through EcoLogits, with an adapter structure ready for more providers.

## Installation
Within your virtual environment run:

```bash
pip install -r requirements.txt
```

## Usage

### Carbon-Aware Decorator

Wrap your LLM calls with the `@carbon_aware` decorator to enable sustainable scheduling.
The decorator supports both synchronous and asynchronous functions.

The public decorator signature is:

```python
carbon_aware(
    *,
    max_delay_hours=2,
    forecast_provider=None,
    telemetry_sink=None,
    auto_downgrade=False,
    dirty_threshold=300.0,
    model_fallbacks=None,
    max_session_gco2eq=None,
)
```

```python
from llm_eco_tracker import carbon_aware

@carbon_aware(max_delay_hours=2)
def call_llm(prompt):
    # Your LLM logic here
    pass
```

```python
from llm_eco_tracker import carbon_aware

@carbon_aware(max_delay_hours=2)
async def call_llm_async(prompt):
    # Your async LLM logic here
    pass
```

```python
from llm_eco_tracker import carbon_aware
from llm_eco_tracker.providers import CsvForecastProvider

@carbon_aware(
    max_delay_hours=2,
    forecast_provider=CsvForecastProvider("tests/fixtures/mock_forecast.csv"),
    auto_downgrade=True,
    max_session_gco2eq=50.0,
)
def call_llm_with_csv_forecast(prompt):
    # Your LLM logic here
    pass
```

`auto_downgrade=True` enables an execution-time fallback when the chosen grid window is
still above `dirty_threshold`. The current OpenAI defaults are:

- `gpt-4o -> gpt-4o-mini`
- `gpt-4.1 -> gpt-4.1-mini`
- `gpt-4-turbo -> gpt-4o-mini`
- `gpt-4 -> gpt-4o-mini`

You can override or extend that map per decorator call:

```python
@carbon_aware(
    max_delay_hours=2,
    auto_downgrade=True,
    dirty_threshold=300.0,
    model_fallbacks={"custom-large-model": "custom-small-model"},
    max_session_gco2eq=50.0,
)
def call_llm_with_fallbacks(prompt):
    pass
```

`max_session_gco2eq` enables a hard per-session carbon budget. When one decorated run
exceeds that emitted-carbon limit, the library raises `CarbonBudgetExceededError`.

```python
from llm_eco_tracker import CarbonBudgetExceededError, carbon_aware

@carbon_aware(max_session_gco2eq=50.0)
def run_agent():
    pass

try:
    run_agent()
except CarbonBudgetExceededError as exc:
    print(exc.actual_gco2eq, exc.max_session_gco2eq)
```

### Telemetry Outputs

By default, `@carbon_aware` writes normalized telemetry records to `eco_telemetry.jsonl`
in the current working directory.

The library currently supports these telemetry sink shapes:

- `JsonlTelemetrySink`: writes one serialized telemetry payload per line.
- `LoggerTelemetrySink`: emits `Telemetry record: {...}` through Python logging.
- `NoOpTelemetrySink`: discards telemetry records.
- `CompositeTelemetrySink`: fans out one telemetry record to multiple sinks.

Telemetry records include per-session `model_usage` summaries so you can see which
requested models were kept versus downgraded. When the carbon circuit breaker is enabled,
telemetry metadata also records the configured session budget, the emitted carbon reached
so far, and whether the run was terminated for exceeding that budget.

### Telemetry Report CLI

You can summarize lifetime telemetry directly from the package with:

```bash
python -m llm_eco_tracker.report
```

You can also point it at one or more custom inputs:

```bash
python -m llm_eco_tracker.report path/to/eco_telemetry.jsonl
python -m llm_eco_tracker.report telemetry-a.jsonl telemetry-b.jsonl
python -m llm_eco_tracker.report app.log --format logger
```

The report command auto-detects JSONL versus logger-backed inputs by default and prints:

- Total LLM jobs run
- Total gCO2eq emitted
- Total gCO2eq saved by EcoTracker
- A car-travel equivalence based on emitted gCO2eq

### Fixture Data

The repository includes a CSV fixture at `tests/fixtures/mock_forecast.csv`.
To refresh it from the UK Carbon Intensity API, run:

```bash
python scripts/download_mock_data.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
