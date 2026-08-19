"""Online feature store: Redis-backed, with a transparent in-memory
(fakeredis) fallback so the demo runs without a live Redis instance.

Uses the real `redis-py` client and Redis wire protocol either way — the
only thing that changes is which server it talks to — so swapping the
fallback out for a real managed Redis (see docker-compose.yml) in an actual
deployment requires no code change, only unsetting `AD_ENGINE_USE_FAKE_REDIS`
or simply having a reachable Redis at connect time.
"""
from __future__ import annotations

import json
import os

import numpy as np
import redis

from src import config

_client = None
IS_FAKE = False  # True once get_redis_client() has fallen back to in-process fakeredis


def get_redis_client():
    global _client, IS_FAKE
    if _client is not None:
        return _client

    force_fake = os.environ.get("AD_ENGINE_USE_FAKE_REDIS", "").lower() in {"1", "true", "yes"}
    if not force_fake:
        try:
            candidate = redis.Redis(
                host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB,
                decode_responses=True, socket_connect_timeout=1,
            )
            candidate.ping()
            _client = candidate
            print(f"[feature_store] connected to real Redis at {config.REDIS_HOST}:{config.REDIS_PORT}")
            return _client
        except redis.exceptions.RedisError:
            pass

    import fakeredis
    print(
        "[feature_store] WARNING: no reachable Redis found — falling back to an in-process "
        "fakeredis instance for this demo. Run `docker compose up -d redis` for a real backend. "
        "Note: this in-process store is NOT shared with `scripts/push_feature_store.py` run as a "
        "separate process — `src.serve` self-seeds from the parquet snapshot on startup instead."
    )
    IS_FAKE = True
    _client = fakeredis.FakeStrictRedis(decode_responses=True)
    return _client


class FeatureStore:
    def __init__(self, client=None):
        self.r = client or get_redis_client()

    def write_customer(self, customer_id: int, features: list[float], current_holdings: list[int],
                        segment: str, age: float, income: float, consent: bool = True) -> None:
        self.r.hset(
            f"user:{customer_id}",
            mapping={
                "features": json.dumps(features),
                "holdings": json.dumps(current_holdings),
                "segment": segment,
                "age": age,
                "income": income,
                "consent": int(consent),
            },
        )

    def read_customer(self, customer_id: int) -> dict | None:
        raw = self.r.hgetall(f"user:{customer_id}")
        if not raw:
            return None
        return {
            "features": np.array(json.loads(raw["features"]), dtype=np.float32).reshape(1, -1),
            "holdings": set(json.loads(raw["holdings"])),
            "segment": raw["segment"],
            "age": float(raw["age"]),
            "income": float(raw["income"]),
            "consent": bool(int(raw["consent"])),
        }
