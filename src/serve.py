"""Real-time ad serving microservice.

    POST /v1/ad/serve       -> pick + assemble one personalized ad
    POST /v1/event/feedback -> record click/conversion feedback
    GET  /health             -> liveness probe (no auth)

Run: `uvicorn src.serve:app --reload --port 8000`
"""
from __future__ import annotations

import json
import os
import random
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from fastapi import Depends, FastAPI, Header, HTTPException

from src import config, events, feature_store as feature_store_module, guardrails
from src.creative_engine import generate_creative
from src.data_prep import build_serving_snapshot
from src.explain import top_driver
from src.feature_store import FeatureStore
from src.schemas import AdRequest, AdResponse, FeedbackRequest, FeedbackResponse

app = FastAPI(title="Hyper-Personalized Mobile Banking Ads Engine", version="0.1.0")

booster = lgb.Booster(model_file=str(config.MODEL_PATH))
FEATURE_COLUMNS: list[str] = json.loads(config.FEATURE_COLUMNS_PATH.read_text())
POPULARITY = json.loads(config.POPULARITY_PATH.read_text())["ranked_sellable_indices"]
CODE_BY_INDEX = config.CODE_BY_INDEX

feature_store = FeatureStore()
frequency_cap = guardrails.FrequencyCap(feature_store.r)

if feature_store_module.IS_FAKE and config.FEATURES_MONTHLY_PATH.exists():
    # No real Redis reachable: self-seed the in-process store from the last
    # feature-engineering run so the API is immediately demoable without
    # requiring `scripts/push_feature_store.py` to run in the same process.
    _snapshot = build_serving_snapshot(pl.read_parquet(config.FEATURES_MONTHLY_PATH))
    for _rec in _snapshot:
        feature_store.write_customer(
            customer_id=_rec["ncodpers"], features=_rec["features"],
            current_holdings=_rec["current_holding_indices"], segment=_rec["segment"],
            age=_rec["age"], income=_rec["income"], consent=_rec["consent"],
        )
    print(f"[serve] self-seeded fakeredis with {len(_snapshot):,} customers from {config.FEATURES_MONTHLY_PATH.name}")
    if _snapshot:
        print(f"[serve] try customer_id={_snapshot[0]['ncodpers']}")

API_KEY = os.environ.get("AD_ENGINE_API_KEY", config.DEFAULT_API_KEY)
if API_KEY == config.DEFAULT_API_KEY:
    print("[serve] WARNING: using the default demo API key. Set AD_ENGINE_API_KEY for any non-local use.")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _generic_ad(customer_id: int, mode: str, latency_start: float) -> AdResponse:
    """Non-personalized fallback: most popular sellable product, generic copy."""
    idx = POPULARITY[0] if POPULARITY else None
    if idx is None:
        return AdResponse(
            customer_id=customer_id, product_code=None, product_name=None, headline=None,
            cta=None, deep_link=None, creative_source=None, score=None, segment=None,
            personalization_mode=mode, explanation=None,
            latency_ms=round((time.perf_counter() - latency_start) * 1000, 2),
        )
    code = CODE_BY_INDEX[idx]
    meta = config.PRODUCT_CATALOG[code]
    events.emit("impression", {"customer_id": customer_id, "product_code": code, "mode": mode})
    return AdResponse(
        customer_id=customer_id, product_code=code, product_name=meta["name"],
        headline=f"Penawaran untuk Anda: {meta['name']}.", cta="Pelajari Lebih Lanjut",
        deep_link=f"app://ads/open?product={code}", creative_source="generic_popularity",
        score=None, segment=None, personalization_mode=mode, explanation=None,
        latency_ms=round((time.perf_counter() - latency_start) * 1000, 2),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/ad/serve", response_model=AdResponse)
def serve_ad(req: AdRequest, _auth: None = Depends(require_api_key)):
    t_start = time.perf_counter()

    customer = feature_store.read_customer(req.customer_id)
    if customer is None:
        return _generic_ad(req.customer_id, "cold_start_popularity", t_start)

    if not guardrails.consent_allowed(customer["consent"]):
        return _generic_ad(req.customer_id, "consent_declined_generic", t_start)

    scores = booster.predict(customer["features"])[0].astype(np.float64)

    for code, meta in config.PRODUCT_CATALOG.items():
        idx = meta["index"]
        if (not meta["sellable"]) or (idx in customer["holdings"]) or (
            not guardrails.is_eligible(code, customer["age"], customer["income"])
        ):
            scores[idx] = -1.0  # ineligible / already owned / not an advertisable product

    ranked = [i for i in np.argsort(scores)[::-1] if scores[i] > -1.0][: config.TOP_K_CANDIDATES]

    mode = "exploit"
    if ranked and random.random() < config.EXPLORATION_EPSILON:
        ranked = [random.choice(ranked), *ranked]
        mode = "exploration"

    chosen_idx = None
    for idx in ranked:
        code = CODE_BY_INDEX[int(idx)]
        if not frequency_cap.is_capped(req.customer_id, code):
            chosen_idx = int(idx)
            break

    if chosen_idx is None:
        return _generic_ad(req.customer_id, "frequency_capped_fallback", t_start)

    product_code = CODE_BY_INDEX[chosen_idx]
    creative = generate_creative(product_code, customer["segment"], customer["age"], customer["income"])
    explanation = top_driver(booster, customer["features"], chosen_idx, FEATURE_COLUMNS)

    frequency_cap.record_impression(req.customer_id, product_code)
    events.emit("impression", {
        "customer_id": req.customer_id, "product_code": product_code,
        "mode": mode, "score": float(scores[chosen_idx]), "channel": req.channel,
    })

    return AdResponse(
        customer_id=req.customer_id, product_code=product_code, product_name=creative["product_name"],
        headline=creative["headline"], cta=creative["cta"], deep_link=creative["deep_link"],
        creative_source=creative["creative_source"], score=round(float(scores[chosen_idx]), 4),
        segment=customer["segment"], personalization_mode=mode, explanation=explanation,
        latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
    )


@app.post("/v1/event/feedback", response_model=FeedbackResponse)
def record_feedback(req: FeedbackRequest, _auth: None = Depends(require_api_key)):
    events.emit(req.event_type, {
        "customer_id": req.customer_id, "product_code": req.product_code, "channel": req.channel,
    })
    return FeedbackResponse(recorded=True)
