"""Ranking evaluation metrics."""
from __future__ import annotations

import numpy as np


def compute_per_class_report(
    actuals: list[list[int]], top1_preds: np.ndarray, scores: np.ndarray, num_classes: int, k: int = 7,
) -> list[dict]:
    """Per-product classification-style breakdown — the closest analogue of
    a confusion matrix for a 24-way ranked-recommendation problem, computed
    against whatever guardrail-constrained `scores`/`top1_preds` the caller
    passes in (active-holding mask, eligibility, sellable-only, etc. should
    already be applied upstream so this reflects what customers would
    actually be shown).

    - `support`: how many eval customers actually adopted this product that
      month (ground truth net-new add), independent of what was recommended.
    - `precision_top1`: of customers whose top-1 pick was this product, the
      fraction who actually adopted it.
    - `recall_top1`: of customers who actually adopted this product, the
      fraction who had it as their top-1 pick.
    - `hit_rate_at_k`: of customers who actually adopted this product, the
      fraction who had it appear anywhere in their top-k ranked list — the
      per-class analogue of what MAP@k rewards.
    """
    n = len(actuals)
    actual_sets = [set(a) for a in actuals]
    top_k_sets = [set(np.argsort(scores[i])[::-1][:k]) for i in range(n)]

    rows = []
    for c in range(num_classes):
        support = sum(1 for a in actual_sets if c in a)
        predicted_count = int((top1_preds == c).sum())
        true_positives = sum(1 for i in range(n) if top1_preds[i] == c and c in actual_sets[i])
        hits_at_k = sum(1 for i in range(n) if c in actual_sets[i] and c in top_k_sets[i])

        precision = true_positives / predicted_count if predicted_count else 0.0
        recall = true_positives / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        hit_rate_k = hits_at_k / support if support else 0.0

        rows.append({
            "class_index": c,
            "support": support,
            "predicted_count_top1": predicted_count,
            "precision_top1": round(precision, 4),
            "recall_top1": round(recall, 4),
            "f1_top1": round(f1, 4),
            "hit_rate_at_k": round(hit_rate_k, 4),
        })
    return rows


def compute_map_at_k(actuals: list[list[int]], scores: np.ndarray, k: int = 7) -> float:
    """Mean Average Precision @ K. `actuals[i]` is the list of true class
    indices for row i; `scores[i]` is the model's per-class score vector.
    Rows with an empty actuals list are skipped (nothing to be right about)."""
    per_row_scores = []
    for actual, score in zip(actuals, scores):
        if not actual:
            continue
        actual_set = set(actual)
        top_k = np.argsort(score)[::-1][:k]
        hits, running_score = 0.0, 0.0
        for i, pred in enumerate(top_k):
            if pred in actual_set:
                hits += 1.0
                running_score += hits / (i + 1.0)
        per_row_scores.append(running_score / min(len(actual_set), k))
    return float(np.mean(per_row_scores)) if per_row_scores else 0.0
