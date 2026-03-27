from __future__ import annotations

import json
import logging
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from llm_eco_tracker.benchmarking import load_jsonl_records, reset_output_file


def configure_demo_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    logging.getLogger("llm_eco_tracker").setLevel(logging.INFO)


def build_mock_openai_http_clients(
    *,
    request_log: list[dict[str, Any]] | None = None,
    canned_text_prefix: str = "Mock response",
) -> tuple[httpx.Client, httpx.AsyncClient]:
    def build_payload(request_payload: dict[str, Any]) -> dict[str, Any]:
        model = request_payload.get("model", "gpt-4o-mini")
        messages = request_payload.get("messages", [])
        last_content = ""
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                last_content = str(last_message.get("content", ""))
        content = f"{canned_text_prefix}: {last_content[:80]}".strip()
        return {
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "created": 1710000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 24,
                "completion_tokens": 32,
                "total_tokens": 56,
            },
        }

    def sync_handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content.decode("utf-8"))
        if request_log is not None:
            request_log.append(request_payload)
        return httpx.Response(200, json=build_payload(request_payload))

    async def async_handler(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content.decode("utf-8"))
        if request_log is not None:
            request_log.append(request_payload)
        return httpx.Response(200, json=build_payload(request_payload))

    return (
        httpx.Client(transport=httpx.MockTransport(sync_handler)),
        httpx.AsyncClient(transport=httpx.MockTransport(async_handler)),
    )


async def demo_sleep(seconds: float) -> None:
    print(f"[demo] simulated scheduler delay: {seconds:.1f}s")


@contextmanager
def demo_runtime_patches():
    with ExitStack() as stack:
        stack.enter_context(patch("ecologits.log.logger.warning_once", return_value=None))
        stack.enter_context(patch("llm_eco_tracker.execution.asyncio.sleep", side_effect=demo_sleep))
        stack.enter_context(
            patch("llm_eco_tracker.api.apply_jitter_to_plan", side_effect=lambda schedule_plan: schedule_plan)
        )
        yield


def prepare_telemetry_path(filename: str) -> Path:
    return reset_output_file(BASE_DIR / filename)


def read_last_telemetry_record(path: str | Path) -> dict[str, Any]:
    records = load_jsonl_records(path)
    if not records:
        raise RuntimeError(f"No telemetry records were written to '{path}'.")
    return records[-1]


def summarize_model_usage(record: dict[str, Any]) -> int:
    return sum(int(entry.get("call_count", 0)) for entry in record.get("model_usage", []))
