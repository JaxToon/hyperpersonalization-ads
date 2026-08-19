"""Guardrails applied on top of raw model scores before an ad is shown.

These run in this order at serving time:
  1. consent gate       — no personalized targeting without opt-in (UU PDP / OJK-style principle)
  2. active-holding mask — never advertise a product the customer already has
  3. eligibility filter  — simple rule-based age/income overrides (illustrative, not real underwriting)
  4. frequency cap       — stop re-showing the same product ad too often (ad fatigue)
"""
from __future__ import annotations

import time

import numpy as np

from src import config

PRODUCT_CODES = list(config.PRODUCT_CATALOG.keys())
MIN_AGE = np.array([config.PRODUCT_CATALOG[p]["min_age"] for p in PRODUCT_CODES])
MIN_INCOME = np.array([config.PRODUCT_CATALOG[p]["min_income"] for p in PRODUCT_CODES])


def consent_allowed(consent_flag: bool) -> bool:
    """No behavioral personalization without explicit customer consent.
    Callers must fall back to a generic (non-personalized) placement when
    this returns False, never silently skip the check."""
    return bool(consent_flag)


def mask_active_holdings(scores: np.ndarray, holdings_mask: np.ndarray) -> np.ndarray:
    """Zero out scores for products already held. `holdings_mask` is a
    boolean array/matrix, True where the product is currently owned."""
    out = scores.copy()
    out[holdings_mask] = 0.0
    return out


def eligibility_mask_batch(age: np.ndarray, income: np.ndarray) -> np.ndarray:
    """Vectorized eligibility check for a batch of customers.
    Returns a boolean matrix (n_customers, n_products), True = eligible."""
    age = np.asarray(age).reshape(-1, 1)
    income = np.asarray(income).reshape(-1, 1)
    return (age >= MIN_AGE.reshape(1, -1)) & (income >= MIN_INCOME.reshape(1, -1))


def is_eligible(product_code: str, age: float, income: float) -> bool:
    rule = config.PRODUCT_CATALOG[product_code]
    return age >= rule["min_age"] and income >= rule["min_income"]


class FrequencyCap:
    """Rolling-window impression cap backed by the feature store's Redis
    connection (a sorted set per customer+product, scored by impression
    timestamp — trivially portable to a real Redis cluster)."""

    def __init__(self, redis_client, max_exposures: int = config.FREQUENCY_CAP_MAX_EXPOSURES,
                 window_days: int = config.FREQUENCY_CAP_WINDOW_DAYS):
        self.r = redis_client
        self.max_exposures = max_exposures
        self.window_seconds = window_days * 86400

    def _key(self, customer_id: int, product_code: str) -> str:
        return f"freq_cap:{customer_id}:{product_code}"

    def is_capped(self, customer_id: int, product_code: str) -> bool:
        key = self._key(customer_id, product_code)
        now = time.time()
        self.r.zremrangebyscore(key, 0, now - self.window_seconds)
        return self.r.zcard(key) >= self.max_exposures

    def record_impression(self, customer_id: int, product_code: str) -> None:
        key = self._key(customer_id, product_code)
        now = time.time()
        self.r.zadd(key, {str(now): now})
        self.r.expire(key, self.window_seconds)
