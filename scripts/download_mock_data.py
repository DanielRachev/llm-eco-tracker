import argparse
import csv
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = BASE_DIR / "tests" / "fixtures" / "benchmark_trace.csv"
DEFAULT_START_DAY = "2026-02-14"
DEFAULT_END_DAY = "2026-03-15"
DEFAULT_TIMEOUT_SECONDS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a historical UK grid trace for benchmark evaluation."
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--start-day",
        type=str,
        default=DEFAULT_START_DAY,
        help="Inclusive UTC start day in ISO format, for example 2026-02-15.",
    )
    parser.add_argument(
        "--end-day",
        type=str,
        default=DEFAULT_END_DAY,
        help="Inclusive UTC end day in ISO format, for example 2026-03-15.",
    )
    parser.add_argument(
        "--last-n-days",
        type=int,
        default=None,
        help="Override start/end and fetch the last N completed UTC days.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def parse_iso_day(value: str) -> date:
    return date.fromisoformat(value)


def resolve_day_window(args: argparse.Namespace) -> tuple[date, date]:
    if args.last_n_days is not None:
        if args.last_n_days <= 0:
            raise ValueError("last_n_days must be positive when provided.")
        end_day = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        start_day = end_day - timedelta(days=args.last_n_days - 1)
        return start_day, end_day

    start_day = parse_iso_day(args.start_day)
    end_day = parse_iso_day(args.end_day)
    if start_day > end_day:
        raise ValueError("start_day must be on or before end_day.")
    return start_day, end_day


def iter_days(start_day: date, end_day: date):
    cursor = start_day
    while cursor <= end_day:
        yield cursor
        cursor += timedelta(days=1)


def build_day_url(day: date) -> str:
    iso_day = day.isoformat()
    return f"https://api.carbonintensity.org.uk/intensity/{iso_day}T00:00Z/{iso_day}T23:59Z"


def download_trace(
    *,
    start_day: date,
    end_day: date,
    output_path: Path,
    timeout_seconds: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    session = requests.Session()
    try:
        for day in iter_days(start_day, end_day):
            url = build_day_url(day)
            print(f"Fetching {day.isoformat()} from UK Carbon Intensity API...")
            response = session.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()["data"]
            for entry in payload:
                rows.append(
                    {
                        "from": entry["from"],
                        "to": entry["to"],
                        "intensity_forecast": entry["intensity"]["forecast"],
                        "intensity_actual": entry["intensity"]["actual"],
                        "index": entry["intensity"]["index"],
                    }
                )
    finally:
        session.close()

    if not rows:
        raise RuntimeError("No historical rows were returned by the API.")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "from",
                "to",
                "intensity_forecast",
                "intensity_actual",
                "index",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Saved {len(rows)} half-hour intervals for {start_day.isoformat()} to "
        f"{end_day.isoformat()} at {output_path}"
    )
    return output_path


def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    start_day, end_day = resolve_day_window(args)
    download_trace(
        start_day=start_day,
        end_day=end_day,
        output_path=args.output_path,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
