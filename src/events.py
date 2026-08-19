"""Feedback event bus.

`emit(event_type, payload)` mirrors a Kafka producer's `send(topic, value)`
call closely enough that swapping this for a real `kafka-python` /
`confluent-kafka` producer later is a one-file change: keep the same
`emit()` signature, publish to a topic per `event_type` instead of
appending to a local JSONL log.
"""
from __future__ import annotations

import json
import threading
import time

from src import config

_lock = threading.Lock()


def emit(event_type: str, payload: dict) -> None:
    record = {"event_type": event_type, "ts": time.time(), **payload}
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(config.EVENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def read_all() -> list[dict]:
    if not config.EVENT_LOG_PATH.exists():
        return []
    with open(config.EVENT_LOG_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
