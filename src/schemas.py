from __future__ import annotations

from pydantic import BaseModel, Field


class AdRequest(BaseModel):
    customer_id: int
    channel: str = Field(default="app_home_banner")


class AdResponse(BaseModel):
    customer_id: int
    product_code: str | None
    product_name: str | None
    headline: str | None
    cta: str | None
    deep_link: str | None
    creative_source: str | None
    score: float | None
    segment: str | None
    personalization_mode: str
    explanation: dict | None
    latency_ms: float


class FeedbackRequest(BaseModel):
    customer_id: int
    product_code: str
    event_type: str = Field(pattern="^(click|conversion)$")
    channel: str = Field(default="app_home_banner")


class FeedbackResponse(BaseModel):
    recorded: bool
