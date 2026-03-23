import csv
from pathlib import Path

import requests


DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mock_forecast.csv"
)


def download_mock_data(output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch data from March 18 to March 20 2026.
    url = "https://api.carbonintensity.org.uk/intensity/2026-03-18T00:00Z/2026-03-20T23:59Z"

    print("Fetching historical data from UK Grid...")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()["data"]

    with output_path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["from", "to", "intensity_forecast", "intensity_actual", "index"]
        )

        for entry in data:
            writer.writerow(
                [
                    entry["from"],
                    entry["to"],
                    entry["intensity"]["forecast"],
                    entry["intensity"]["actual"],
                    entry["intensity"]["index"],
                ]
            )

    print(f"Success! {output_path} has been created.")
    return output_path


if __name__ == "__main__":
    download_mock_data()
