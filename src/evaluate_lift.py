"""Offline A/B lift simulation: personalized NBA ranking vs. a single
non-personalized "blanket campaign" showing everyone the single most
popular net-new product.

Conversion proxy: whether the product actually advertised is one the
customer really adopted in the held-out month (ground truth from the
dataset). This is the standard offline proxy for this kind of problem —
it is NOT the same as a real online A/B test, because it can't capture
creative-driven incrementality (a customer might have adopted the product
anyway, unprompted). Treat the lift number here as a sanity check that the
ranking model beats a blanket campaign, not as a promised production CTR.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src import config, guardrails
from src.data_prep import FEATURE_COLUMNS, PRODUCT_CODES
from src.metrics import compute_map_at_k, compute_per_class_report


def main() -> None:
    booster = lgb.Booster(model_file=str(config.MODEL_PATH))
    popularity = json.loads(config.POPULARITY_PATH.read_text())["ranked_sellable_indices"]
    eval_month = json.loads((config.ARTIFACTS_DIR / "eval_month.json").read_text())["eval_month"]

    monthly = pl.read_parquet(config.FEATURES_MONTHLY_PATH)
    eval_df = monthly.filter(pl.col("fecha_dato").dt.strftime("%Y-%m-%d") == eval_month)
    n = eval_df.height
    print(f"Evaluating lift on {n:,} customers, held-out month {eval_month}")

    X = eval_df.select(FEATURE_COLUMNS).to_numpy()
    age = eval_df["age"].to_numpy()
    income = eval_df["renta"].to_numpy()
    actuals = eval_df["added_product_indices"].to_list()

    scores = booster.predict(X)

    lag1_cols = [f"{p}_lag1" for p in PRODUCT_CODES]
    holding_mask = eval_df.select(lag1_cols).to_numpy().astype(bool)
    scores = guardrails.mask_active_holdings(scores, holding_mask)

    not_sellable = np.array([idx not in config.SELLABLE_INDICES for idx in range(config.NUM_CLASSES)])
    scores[:, not_sellable] = -1.0

    eligible = guardrails.eligibility_mask_batch(age, income)
    scores = np.where(eligible, scores, -1.0)

    map7 = compute_map_at_k(actuals, scores, k=7)

    personalized_top1 = np.argmax(scores, axis=1)
    personalized_hits = np.array([
        top1 in set(actual) for top1, actual in zip(personalized_top1, actuals)
    ])
    personalized_rate = float(personalized_hits.mean())

    baseline_product = popularity[0]
    baseline_hits = np.array([baseline_product in set(actual) for actual in actuals])
    baseline_rate = float(baseline_hits.mean())

    lift_pct = ((personalized_rate - baseline_rate) / baseline_rate * 100) if baseline_rate > 0 else float("inf")

    report = {
        "eval_month": eval_month,
        "n_customers": n,
        "map_at_7_guardrail_constrained": round(map7, 4),
        "baseline_blanket_conversion_rate": round(baseline_rate, 5),
        "personalized_conversion_rate": round(personalized_rate, 5),
        "relative_lift_pct": round(lift_pct, 1),
        "baseline_product": config.PRODUCT_CATALOG[config.CODE_BY_INDEX[baseline_product]]["name"],
        "note": "offline proxy (did the customer adopt the advertised product) — not a real online A/B result",
    }
    (config.ARTIFACTS_DIR / "lift_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Blanket campaign\n(most popular product)", "Personalized NBA\n(this model)"],
        [baseline_rate * 100, personalized_rate * 100],
        color=["#9aa5b1", "#2f6fed"],
    )
    ax.set_ylabel("Offline conversion-proxy rate (%)")
    ax.set_title(f"Personalization lift: {lift_pct:+.0f}%  (MAP@7={map7:.3f})")
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(config.ARTIFACTS_DIR / "lift_report.png", dpi=150)
    print(f"Wrote {config.ARTIFACTS_DIR / 'lift_report.json'} and lift_report.png")

    write_class_report(actuals, personalized_top1, scores, map7)


def write_class_report(actuals: list[list[int]], top1_preds: np.ndarray, scores: np.ndarray, map7: float) -> None:
    """Per-product precision/recall/support breakdown (guardrail-constrained,
    top-1) — the closest analogue of a confusion matrix for this 24-class
    ranked-recommendation problem. Non-sellable classes (behavioral signals
    like payroll/pension inflow) are never eligible to be recommended, so
    their precision/recall/hit-rate are structurally 0 — that's expected,
    not a modeling failure."""
    rows = compute_per_class_report(actuals, top1_preds, scores, config.NUM_CLASSES, k=7)
    for row in rows:
        code = config.CODE_BY_INDEX[row["class_index"]]
        meta = config.PRODUCT_CATALOG[code]
        row["product_code"] = code
        row["product_name"] = meta["name"]
        row["category"] = meta["category"]
        row["sellable"] = meta["sellable"]
    rows.sort(key=lambda r: r["support"], reverse=True)

    total_support = sum(r["support"] for r in rows)
    sellable_rows = [r for r in rows if r["sellable"]]

    def _avg(key: str, weighted: bool) -> float:
        if not sellable_rows:
            return 0.0
        if weighted and total_support:
            return sum(r[key] * r["support"] for r in sellable_rows) / total_support
        return sum(r[key] for r in sellable_rows) / len(sellable_rows)

    summary = {
        "eval_month": json.loads((config.ARTIFACTS_DIR / "eval_month.json").read_text())["eval_month"],
        "map_at_7_guardrail_constrained": round(map7, 4),
        "macro_avg_precision_top1": round(_avg("precision_top1", weighted=False), 4),
        "macro_avg_recall_top1": round(_avg("recall_top1", weighted=False), 4),
        "weighted_avg_precision_top1": round(_avg("precision_top1", weighted=True), 4),
        "weighted_avg_recall_top1": round(_avg("recall_top1", weighted=True), 4),
        "note": "sellable=false rows (behavioral signals, never advertised) are structurally "
                "0 across precision/recall/hit-rate by design, not a model failure.",
        "per_product": rows,
    }
    (config.ARTIFACTS_DIR / "class_report.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'Product':<32}{'Category':<12}{'Support':>8}{'Precision@1':>13}{'Recall@1':>10}{'F1@1':>8}{'Hit@7':>8}")
    print("-" * 91)
    for r in rows:
        flag = "" if r["sellable"] else "  (signal, not advertised)"
        print(
            f"{r['product_name']:<32}{r['category']:<12}{r['support']:>8}"
            f"{r['precision_top1']:>13.3f}{r['recall_top1']:>10.3f}{r['f1_top1']:>8.3f}{r['hit_rate_at_k']:>8.3f}{flag}"
        )
    print("-" * 91)
    print(f"Macro avg (sellable only):    precision={summary['macro_avg_precision_top1']:.3f}  recall={summary['macro_avg_recall_top1']:.3f}")
    print(f"Weighted avg (sellable only):  precision={summary['weighted_avg_precision_top1']:.3f}  recall={summary['weighted_avg_recall_top1']:.3f}")
    print(f"Wrote {config.ARTIFACTS_DIR / 'class_report.json'}")

    plot_rows = [r for r in sellable_rows if r["support"] > 0]
    plot_rows.sort(key=lambda r: r["recall_top1"])
    if plot_rows:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(plot_rows))))
        y = np.arange(len(plot_rows))
        ax.barh(y, [r["recall_top1"] for r in plot_rows], color="#2f6fed", label="Recall@1")
        ax.barh(y, [r["hit_rate_at_k"] - r["recall_top1"] for r in plot_rows],
                left=[r["recall_top1"] for r in plot_rows], color="#9ecbff", label="+ extra reach by Hit@7")
        ax.set_yticks(y, [r["product_name"] for r in plot_rows], fontsize=8)
        ax.set_xlabel("Rate among customers who actually adopted the product")
        ax.set_title("Per-product recall: top-1 pick vs. anywhere in top-7")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(config.ARTIFACTS_DIR / "class_report.png", dpi=150)
        print(f"Wrote {config.ARTIFACTS_DIR / 'class_report.png'}")


if __name__ == "__main__":
    main()
