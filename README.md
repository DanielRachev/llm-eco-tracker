# llm-eco-tracker

`llm-eco-tracker` is a lightweight Python library designed to make LLM usage sustainable. It provides tools to track carbon footprint and a "Carbon-Aware Scheduler" to delay non-urgent processing until the local power grid runs on greener energy.

## Features

- **Carbon Tracking**: Estimate operational and embodied carbon footprint of LLM requests.
- **Carbon-Aware Scheduling**: Automatically delay execution of decorated functions based on real-time grid intensity forecasts.
- **Provider Support**: Initial focus on OpenAI, with plans to expand to other providers.

## Installation
Within your virtual environment run:

```bash
pip install -r requirements.txt
```

## Usage

### Carbon-Aware Decorator

Wrap your LLM calls with the `@carbon_aware` decorator to enable sustainable scheduling.

```python
from carbon_aware import carbon_aware

@carbon_aware(max_delay_hours=2, location="NL")
def call_llm(prompt):
    # Your LLM logic here
    pass
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
