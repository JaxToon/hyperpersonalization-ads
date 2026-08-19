"""Cheap per-request explainability using LightGBM's native `pred_contrib`
(SHAP-consistent feature contributions computed directly by the booster —
no extra `shap` dependency, no meaningful extra latency)."""
from __future__ import annotations

import numpy as np


def top_driver(booster, features_row: np.ndarray, class_idx: int, feature_names: list[str]) -> dict:
    contrib = booster.predict(features_row, pred_contrib=True)[0]
    n_features = len(feature_names)
    start = class_idx * (n_features + 1)
    class_contrib = contrib[start:start + n_features]
    top_i = int(np.argmax(np.abs(class_contrib)))
    return {"feature": feature_names[top_i], "contribution": round(float(class_contrib[top_i]), 4)}
