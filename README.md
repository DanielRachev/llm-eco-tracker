# llm-eco-tracker

`llm-eco-tracker` is a lightweight Python library designed to make LLM usage more sustainable. It provides a carbon-aware decorator that can delay non-urgent work until a greener grid window and record telemetry for supported LLM calls.

## Features

- **Carbon-Aware Scheduling**: Delay execution based on forecasted grid carbon intensity.
- **Forecast Providers**: Use the live UK Carbon Intensity API or inject a CSV-backed provider for deterministic runs.
- **Telemetry Adapters**: Current telemetry support is wired for both OpenAI chat completions and Anthropic messages through EcoLogits.

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
still above `dirty_threshold`. The built-in default fallbacks currently cover OpenAI:

- `gpt-4o -> gpt-4o-mini`
- `gpt-4.1 -> gpt-4.1-mini`
- `gpt-4-turbo -> gpt-4o-mini`
- `gpt-4 -> gpt-4o-mini`

You can override or extend that map per decorator call. This is also how you provide
Anthropic fallbacks:

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

The library currently captures telemetry from these SDK paths:

- OpenAI `client.chat.completions.create(...)`
- Anthropic `client.messages.create(...)`

The telemetry sinks themselves are:

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

### Trace Data

The repository keeps two CSV datasets:

- `tests/fixtures/mock_forecast.csv`
  A small deterministic fixture used by tests and the integration benchmark.
- `tests/fixtures/benchmark_trace.csv`
  A multi-day historical trace for the scheduler benchmark and paper figures.

To refresh the benchmark trace from the UK Carbon Intensity API, run:

```bash
python scripts/download_mock_data.py
```

The downloader defaults to a fixed 30-day historical window for reproducibility,
but you can override it:

```bash
python scripts/download_mock_data.py --start-day 2026-02-14 --end-day 2026-03-15
python scripts/download_mock_data.py --last-n-days 30 --output-path tests/fixtures/benchmark_trace.csv
```

### Benchmark Scripts

The repository now includes three evaluation scripts plus one analysis script:

- `python scripts/run_benchmark.py`
  Runs the multi-day trace-driven scheduler benchmark. For every eligible day
  in `tests/fixtures/benchmark_trace.csv`, it sweeps all 48 submission slots and
  evaluates three policies:
  - `Baseline`: immediate execution scored with actual grid intensity
  - `EcoTracker`: forecast-driven scheduling scored with actual grid intensity
  - `Oracle`: perfect-information scheduling scored with actual grid intensity

  It writes:
  - `scenario_results.csv`
  - `daily_summary.csv`
  - `benchmark_summary.json`

- `python scripts/analyze_benchmark_results.py`
  Performs the paper-facing statistics on the day-level benchmark output. It
  computes aggregate reductions, descriptive statistics, a bootstrap confidence
  interval for the mean daily reduction, and a paired Wilcoxon signed-rank test.

  It writes:
  - `benchmark_analysis.json`
  - `benchmark_analysis.md`

- `python scripts/run_openai_integration_benchmark.py`
  Runs an end-to-end OpenAI SDK benchmark using `AsyncOpenAI` with an
  `httpx.MockTransport`, so the request path is real but no paid API call is
  made. It verifies that telemetry is captured, model usage is written, energy
  is non-zero, and the SDK monkey-patch is restored after the session.

  It writes:
  - `openai_integration_telemetry.jsonl`
  - `openai_integration_summary.json`

- `python scripts/run_overhead_benchmark.py`
  Measures the decorator's developer-facing overhead with `max_delay_hours=0`
  by comparing repeated batches of undecorated and decorated function calls.

  It writes:
  - `overhead_benchmark_runs.csv`
  - `overhead_benchmark_summary.json`

- `python scripts/generate_paper_figures.py`
  Reads `scenario_results.csv`, `daily_summary.csv`, and the benchmark trace CSV
  and renders paper-ready figures into `paper_figures/` as both PNG and PDF:
  - `figure_1_curve`
  - `figure_2_total_emissions`
  - `figure_3_daily_reductions`

### Recommended Evaluation Workflow

Run the benchmark suite in this order:

```bash
python scripts/run_benchmark.py
python scripts/analyze_benchmark_results.py
python scripts/run_openai_integration_benchmark.py
python scripts/run_overhead_benchmark.py
python scripts/generate_paper_figures.py
```

This corresponds to the intended methodology:

- multi-day trace benchmark for scheduler effectiveness
- day-level statistical analysis
- end-to-end integration validation for the real OpenAI interception path
- lightweight overhead measurement for developer experience

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
