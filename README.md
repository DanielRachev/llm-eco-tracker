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
)
def call_llm_with_csv_forecast(prompt):
    # Your LLM logic here
    pass
```

### Fixture Data

The repository includes a CSV fixture at `tests/fixtures/mock_forecast.csv`.
To refresh it from the UK Carbon Intensity API, run:

```bash
python scripts/download_mock_data.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
