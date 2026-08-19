# Evaluation notes — NBA model quality

This file tracks evaluation findings across model iterations. Numbers come
from `scripts/run_training.py` (`artifacts/training_report.json`) and
`scripts/run_lift_simulation.py` (`artifacts/lift_report.json` +
`artifacts/class_report.json`), run on the held-out month `2016-05-28`
(93,255 customers, 1/5 customer sample, 8 months of history).

Metric definitions are in [`src/metrics.py`](../src/metrics.py):
- **MAP@7** — standard ranking quality metric (Kaggle Santander competition metric).
- **Precision@1 / Recall@1** — per product, treating the guardrail-constrained
  top-1 pick as a single predicted label (see `write_class_report` in
  [`src/evaluate_lift.py`](../src/evaluate_lift.py)).
- **Hit@7** — per product, whether it appears anywhere in the top-7 ranked
  list (the per-class quantity MAP@7 actually rewards).
- **Conversion lift** — offline proxy A/B: personalized top-1 vs a blanket
  campaign showing everyone the single most popular net-new product. Not a
  real online CTR — see the caveat in `src/evaluate_lift.py`.

## v1 baseline (unweighted LightGBM) — snapshotted in `artifacts/baseline_v1/`

| Metric | Value |
|---|---|
| MAP@7 (raw, holding-mask only) | 0.886 |
| MAP@7 (guardrail-constrained) | 0.622 |
| Conversion — blanket baseline | 0.618% |
| Conversion — personalized top-1 | 2.007% |
| Relative lift | +225% |
| Macro-avg precision@1 (sellable) | 0.015 |
| Macro-avg recall@1 (sellable) | 0.223 |
| Weighted-avg precision@1 | 0.016 |
| Weighted-avg recall@1 | 0.479 |

### Finding: popularity collapse at top-1 ("safe guess" behavior)

Training-label distribution is heavily imbalanced (`Rekening Giro` = 60,028
positive rows vs `Bank Garansi` = 2, out of 144,631 total — see
`scripts/build_features.py` output). Standard unweighted multiclass
LightGBM optimizes aggregate log-loss, which is dominated by the few
high-frequency classes.

Effect observed in `artifacts/baseline_v1/class_report.json`:

| Product | Support | Precision@1 | Recall@1 | Hit@7 |
|---|---|---|---|---|
| Autodebet Tagihan | 1032 | 0.019 | 0.932 | 1.000 |
| Rekening Giro | 576 | 0.018 | 0.936 | 0.962 |
| Rekening Digital | 282 | 0.047 | 0.652 | 1.000 |
| Rekening Payroll | 243 | 0.043 | 0.601 | 1.000 |
| Kartu Kredit | 436 | 0.000 | 0.000 | 0.000 |
| Tabungan Prioritas | 17 | 0.000 | 0.000 | 0.000 |

Recall@1 for the top two products looks strong (0.93+), but precision@1 is
~0.02 — the model assigns those two products as the top-1 pick for roughly
**30,000 of 93,255 customers**, when only a few hundred actually adopted
them that month. In practice the model is defaulting most customers to the
two most statistically common net-new products rather than discriminating
between individuals. Several genuinely-relevant products (credit card,
priority savings) never win the top-1 slot at all.

This is masked by MAP@7 and Hit@7 because both are lenient out to rank 7 —
a popular product almost always lands somewhere in a top-7 list regardless
of whether it's *right* for that customer. **A real ad placement typically
shows one slot (top-1)**, so precision@1 is the metric that actually
matters for ad relevance, and it exposes the weakness that MAP@7 alone
hides.

Root cause: class imbalance, not a code defect. Addressed in v2 below via
inverse-frequency sample weighting.

## v2a — class-weighted LightGBM (inverse-frequency sample weights)

Tried first, on the theory that the popularity collapse was purely a
training-time class-imbalance effect. Added `compute_class_weights()` in
[`src/train.py`](../src/train.py) (inverse-frequency, sqrt-dampened,
capped at 8x) and passed it as LightGBM's per-sample `weight`.

**Result: negligible.** Precision@1 for the top two products stayed at
~0.018-0.019, and structurally-zero products (Kartu Kredit, Tabungan
Prioritas) stayed at exactly zero. Raising the cap to 50x and dropping the
sqrt dampening (full inverse frequency) still didn't move Kartu Kredit off
zero. That was the signal that this wasn't a training-time problem at all —
sample weighting can't fix a product that's being zeroed out by a guardrail
*after* inference, no matter how well the model scores it internally.
Sample weighting is kept in the training pipeline (mild, real effect on the
genuinely-competitive high-frequency classes) but was not the fix for the
zero-precision products.

## v2b — root cause: income-scale unit mismatch in eligibility guardrail (the actual fix)

Investigated why Kartu Kredit (support 436) and other income-gated products
scored exactly 0.000 precision/recall at top-1 in *both* v1 and v2a, while
un-gated products (`min_income: 0`) were fine. Checked the proxy dataset's
actual `renta` (income) column:

```
min 2,336   p10 53,329   p25 71,339   median 96,710   p75 138,398   p90 210,594   max 28,894,395
```

`src/config.py`'s `PRODUCT_CATALOG` had `min_income` thresholds written as
nominal Rupiah millions (e.g. Kartu Kredit: `3_000_000`, KPR: `8_000_000`,
Tabungan Prioritas: `15_000_000`) — 30-150x above the dataset's actual
scale. `guardrails.eligibility_mask_batch()` correctly implements
`income >= min_income`, but with that mismatch **only 0.03% of customers
ever cleared the Kartu Kredit threshold and 0.02% cleared KPR** — the
guardrail was forcing scores to `-1` for essentially the entire customer
base regardless of what the model predicted. This silently masqueraded as
a modeling limitation in the class_report but was actually a config bug:
[`src/config.py`](../src/config.py) L33-86.

The same bug also existed in the creative layer
([`src/creative_engine.py`](../src/creative_engine.py)): `_cashback_rate`
gated on `income >= 10_000_000` (never true) and `_credit_limit_estimate`
computed `income * 3` rounded to the nearest million with a 1,000,000
floor — which floored to the *same* Rp1,000,000 "personalized" credit
limit for nearly every customer, defeating the stated purpose of that hook
being 1:1 rather than generic.

**Fix:** added `INCOME_P10..INCOME_P90` percentile constants to
`config.py`, calibrated to the dataset's real distribution, and rewired
every income-gated `min_income` plus both creative-engine income tiers to
use them as the single source of truth.

### Before/after (same held-out month, same customers)

| Metric | v1 baseline | v2a (weighting only) | v2b (eligibility fix) |
|---|---|---|---|
| MAP@7 (guardrail-constrained) | 0.622 | 0.620 | **0.734** |
| Personalized conversion rate | 2.007% | 1.99% | **2.33%** |
| Relative lift vs blanket | +225% | +222% | **+277%** |
| Kartu Kredit recall@1 (support 436) | 0.000 | 0.000 | **0.764** |
| Kartu Kredit precision@1 | 0.000 | 0.000 | **0.061** |
| Weighted-avg recall@1 (sellable) | 0.479 | 0.478 | **0.556** |
| Weighted-avg precision@1 (sellable) | 0.016 | 0.016 | **0.025** |

### Remaining known limitation (v2b, 1/5 customer sample)

Products with very low support in this held-out month (Tabungan Prioritas:
17, Reksa Dana: 9, KPR: 1) still show precision/recall@1 = 0. Their
Hit@7 improved from 0 to non-zero (e.g. Reksa Dana Saham: 0.000 → 0.200),
meaning the model now ranks them competitively but rarely wins outright
top-1 against high-frequency products. This looked like a genuine data
sparsity limit rather than a bug — confirmed below.

## v3 — full customer population (`--sample-mod 1`, no sampling)

Reran `scripts/build_features.py --sample-mod 1` (945,184 customers,
7.33M customer-month rows, 1,449,591 exploded training rows — 10x v2b) to
check whether the v2b low-support products were a sampling artifact.
Same 8-month window, same held-out month `2016-05-28`.

| Metric | v2b (1/5 sample) | v3 (full population) |
|---|---|---|
| MAP@7 (guardrail-constrained) | 0.734 | **0.740** |
| Personalized conversion rate | 2.33% | **2.28%** |
| Relative lift vs blanket | +277% | **+285%** |
| Weighted-avg recall@1 (sellable) | 0.556 | **0.560** |

| Product | Support (v2b → v3) | Recall@1 (v2b → v3) | Hit@7 (v2b → v3) |
|---|---|---|---|
| Tabungan Prioritas | 17 → 131 | 0.000 → **0.015** | 0.059 → 0.092 |
| Reksa Dana Saham | 20 → 183 | 0.000 → **0.049** | 0.200 → 0.164 |
| Reksa Dana | 9 → 61 | 0.000 → **0.049** | 0.222 → **0.607** |
| Deposito Jangka Panjang | 2 → 46 | 0.000 → **0.261** | 0.500 → 0.935 |
| Kredit Multiguna | 1 → 7 | 0.000 → **0.143** | 1.000 → 0.571 |
| KPR | 1 → 3 | 0.000 → 0.000 (support still tiny) | 0.000 → 0.667 |

**Conclusion: confirmed data sparsity, not a bug.** With 10x more support,
every previously-zero product picked up non-trivial recall@1 or at least a
much higher Hit@7. Precision/recall@1 for the smallest-support products
(KPR: support 3) is still weak — with only a handful of true examples in
the whole eval month, no realistic amount of data at this dataset's overall
scale will make that estimate stable. This is an intrinsic long-tail
problem, not something to keep chasing in this PoC; it would need either a
longer observation window (more than 8 months) or a real production
traffic volume to resolve fully. `artifacts/nba_lightgbm.txt` and all
other `artifacts/*` now reflect this v3 full-population model.
