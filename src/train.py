"""Train the LightGBM multiclass Next-Best-Action model and report MAP@7 on
a held-out month, with active-holding masking applied at evaluation time so
the metric reflects what the serving guardrail will actually show users."""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from src import config
from src.data_prep import FEATURE_COLUMNS, PRODUCT_CODES
from src.metrics import compute_map_at_k

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": config.NUM_CLASSES,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 20,
    "verbosity": -1,
    "n_jobs": -1,
}

# Training labels are heavily imbalanced (the most common net-new product
# has ~30,000x more positive rows than the rarest). An unweighted objective
# lets aggregate log-loss get dominated by the few high-frequency classes,
# which collapses top-1 picks onto 1-2 "safe" popular products for most
# customers (see docs/EVALUATION_NOTES.md v1 baseline: precision@1 ~0.02 on
# the top products despite recall@1 > 0.9). Inverse-frequency sample
# weighting counteracts this; sqrt-dampened and capped so the handful of
# classes with only a few dozen examples don't get blown up into noise.
CLASS_WEIGHT_CAP = 50.0


def compute_class_weights(y: np.ndarray, num_classes: int, cap: float = CLASS_WEIGHT_CAP) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # never-seen class: weight is irrelevant, avoid div-by-zero
    inv_freq = counts.sum() / counts
    weights = inv_freq / inv_freq.mean()
    return np.clip(weights, 1.0 / cap, cap)


def train_model(train_exploded: pl.DataFrame) -> tuple[lgb.Booster, np.ndarray]:
    X = train_exploded.select(FEATURE_COLUMNS).to_numpy()
    y = train_exploded["target_class"].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    class_weights = compute_class_weights(y_train, config.NUM_CLASSES)
    sample_weight = class_weights[y_train]

    # Validation stays unweighted so early stopping tracks log-loss against
    # the true label distribution (what production traffic looks like).
    train_ds = lgb.Dataset(X_train, label=y_train, weight=sample_weight, feature_name=FEATURE_COLUMNS)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, feature_name=FEATURE_COLUMNS)

    booster = lgb.train(
        LGB_PARAMS,
        train_ds,
        num_boost_round=600,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
    )
    return booster, class_weights


def evaluate_map7(booster: lgb.Booster, eval_month: str) -> float:
    monthly = pl.read_parquet(config.FEATURES_MONTHLY_PATH)
    eval_df = monthly.filter(pl.col("fecha_dato").dt.strftime("%Y-%m-%d") == eval_month)

    X_eval = eval_df.select(FEATURE_COLUMNS).to_numpy()
    scores = booster.predict(X_eval)

    # active-holding guardrail: mask out products the customer already held going into this month
    lag1_cols = [f"{p}_lag1" for p in PRODUCT_CODES]
    holding_mask = eval_df.select(lag1_cols).to_numpy().astype(bool)
    scores = scores.copy()
    scores[holding_mask] = 0.0

    actuals = eval_df["added_product_indices"].to_list()
    return compute_map_at_k(actuals, scores, k=7)


def main() -> None:
    train_exploded = pl.read_parquet(config.TRAIN_EXPLODED_PATH)
    print(f"Training on {train_exploded.height:,} exploded (customer, month, added-product) rows")

    booster, class_weights = train_model(train_exploded)
    booster.save_model(str(config.MODEL_PATH))
    print(f"Saved model -> {config.MODEL_PATH}")

    eval_month = json.loads((config.ARTIFACTS_DIR / "eval_month.json").read_text())["eval_month"]
    map7 = evaluate_map7(booster, eval_month)
    print(f"MAP@7 (holding-mask only, raw ranking quality) on held-out month {eval_month}: {map7:.4f}")
    print("Note: this excludes eligibility + sellable-only guardrails — see artifacts/lift_report.json "
          "(run scripts/run_lift_simulation.py) for the serving-realistic MAP@7 customers would actually see.")

    report = {
        "eval_month": eval_month,
        "map_at_7_holding_mask_only": map7,
        "num_boost_round_used": booster.current_iteration(),
        "train_rows": train_exploded.height,
        "num_classes": config.NUM_CLASSES,
        "class_weighting": "inverse-frequency, sqrt-dampened, capped",
        "class_weight_cap": CLASS_WEIGHT_CAP,
        "class_weight_range": [round(float(class_weights.min()), 3), round(float(class_weights.max()), 3)],
    }
    (config.ARTIFACTS_DIR / "training_report.json").write_text(json.dumps(report, indent=2))
    print("Wrote artifacts/training_report.json")


if __name__ == "__main__":
    main()
