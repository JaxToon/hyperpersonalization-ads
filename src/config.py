"""Central configuration: product catalog, segments, paths, and serving policy.

The dataset used for this PoC is the public Kaggle "Santander Product
Recommendation" dataset — it is used purely as a *behavioral proxy* (17
monthly snapshots of product holdings per customer) because no real bank
customer data is available. Product codes are remapped below to generic
Indonesian retail-banking product categories so the PoC reads as a
mobile-banking personalization engine rather than a Santander clone.
Nothing here references any real bank's actual product catalog, pricing,
or branding.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
RAW_CSV_ZIP = DATASET_DIR / "train_ver2.csv.zip"
RAW_CSV_PATH = DATASET_DIR / "raw" / "train_ver2.csv"
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FEATURES_MONTHLY_PATH = DATA_DIR / "features_monthly.parquet"
TRAIN_EXPLODED_PATH = DATA_DIR / "train_exploded.parquet"
POPULARITY_PATH = ARTIFACTS_DIR / "popularity_fallback.json"
MODEL_PATH = ARTIFACTS_DIR / "nba_lightgbm.txt"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
EVENT_LOG_PATH = DATA_DIR / "events.jsonl"

# ---------------------------------------------------------------------------
# Product catalog
#
# `index` is the fixed multiclass label order fed to LightGBM (0..N-1).
# `sellable` = False marks behavioral inflow signals (payroll/pension credit
# detection) that are useful as *features* and as net-new *targets* for
# realism, but are not advertised directly as a product in the creative
# layer (advertising "we noticed your salary" would be a poor/creepy ad).
# Eligibility rules are illustrative, not real underwriting policy.
#
# `min_income` is calibrated against the *raw scale* of the proxy dataset's
# `renta` field, NOT a real Rupiah income figure — see INCOME_P* below. An
# earlier version of this catalog used nominal "millions of Rupiah"
# thresholds (e.g. 15_000_000) that were 100-1000x larger than anything in
# the raw income column, so >99.9% of customers silently failed eligibility
# for every income-gated product — collapsing those products'
# guardrail-constrained top-1 rate to zero regardless of model quality (see
# docs/EVALUATION_NOTES.md). Thresholds below are set as percentiles of the
# *actual* proxy distribution so eligibility behaves as a real filter.
# ---------------------------------------------------------------------------

# Empirical percentiles of `renta` over the current feature-engineered
# sample (recompute if `--sample-mod` / `--months` change materially).
INCOME_P10 = 53_000
INCOME_P25 = 71_000
INCOME_P50 = 97_000
INCOME_P75 = 138_000
INCOME_P85 = 178_000
INCOME_P90 = 211_000

PRODUCT_CATALOG: dict[str, dict] = {
    "ind_ahor_fin_ult1":  {"index": 0,  "name": "Tabungan Reguler",            "category": "Tabungan",   "sellable": True,  "min_age": 17, "min_income": 0},
    "ind_aval_fin_ult1":  {"index": 1,  "name": "Bank Garansi",                "category": "Layanan",    "sellable": False, "min_age": 21, "min_income": 0},
    "ind_cco_fin_ult1":   {"index": 2,  "name": "Rekening Giro",               "category": "Tabungan",   "sellable": True,  "min_age": 21, "min_income": 0},
    "ind_cder_fin_ult1":  {"index": 3,  "name": "Rekening Turunan",            "category": "Tabungan",   "sellable": False, "min_age": 21, "min_income": 0},
    "ind_cno_fin_ult1":   {"index": 4,  "name": "Rekening Payroll",            "category": "Payroll",    "sellable": True,  "min_age": 18, "min_income": 0},
    "ind_ctju_fin_ult1":  {"index": 5,  "name": "Tabungan Pelajar",            "category": "Tabungan",   "sellable": True,  "min_age": 12, "min_income": 0},
    "ind_ctma_fin_ult1":  {"index": 6,  "name": "Tabungan Plus",               "category": "Tabungan",   "sellable": True,  "min_age": 17, "min_income": 0},
    "ind_ctop_fin_ult1":  {"index": 7,  "name": "Tabungan Utama",              "category": "Tabungan",   "sellable": True,  "min_age": 17, "min_income": 0},
    "ind_ctpp_fin_ult1":  {"index": 8,  "name": "Tabungan Prioritas",          "category": "Tabungan",   "sellable": True,  "min_age": 21, "min_income": INCOME_P90},
    "ind_deco_fin_ult1":  {"index": 9,  "name": "Deposito Jangka Pendek",      "category": "Deposito",   "sellable": True,  "min_age": 21, "min_income": 0},
    "ind_deme_fin_ult1":  {"index": 10, "name": "Deposito Jangka Menengah",    "category": "Deposito",   "sellable": True,  "min_age": 21, "min_income": 0},
    "ind_dela_fin_ult1":  {"index": 11, "name": "Deposito Jangka Panjang",     "category": "Deposito",   "sellable": True,  "min_age": 21, "min_income": 0},
    "ind_ecue_fin_ult1":  {"index": 12, "name": "Rekening Digital",           "category": "Digital",    "sellable": True,  "min_age": 17, "min_income": 0},
    "ind_fond_fin_ult1":  {"index": 13, "name": "Reksa Dana",                  "category": "Investasi",  "sellable": True,  "min_age": 21, "min_income": INCOME_P25},
    "ind_hip_fin_ult1":   {"index": 14, "name": "KPR (Kredit Pemilikan Rumah)","category": "Pinjaman",   "sellable": True,  "min_age": 21, "min_income": INCOME_P75},
    "ind_plan_fin_ult1":  {"index": 15, "name": "Dana Pensiun",                "category": "Investasi",  "sellable": True,  "min_age": 30, "min_income": 0},
    "ind_pres_fin_ult1":  {"index": 16, "name": "Kredit Multiguna",            "category": "Pinjaman",   "sellable": True,  "min_age": 21, "min_income": INCOME_P10},
    "ind_reca_fin_ult1":  {"index": 17, "name": "Autodebet Pajak",             "category": "Layanan",    "sellable": True,  "min_age": 21, "min_income": 0},
    "ind_tjcr_fin_ult1":  {"index": 18, "name": "Kartu Kredit",                "category": "Kartu",      "sellable": True,  "min_age": 21, "min_income": INCOME_P10},
    "ind_valo_fin_ult1":  {"index": 19, "name": "Reksa Dana Saham",            "category": "Investasi",  "sellable": True,  "min_age": 21, "min_income": INCOME_P85},
    "ind_viv_fin_ult1":   {"index": 20, "name": "Tabungan Rumah",              "category": "Tabungan",   "sellable": True,  "min_age": 21, "min_income": 0},
    "ind_nomina_ult1":    {"index": 21, "name": "Gaji Masuk (Payroll Inflow)", "category": "Sinyal",     "sellable": False, "min_age": 18, "min_income": 0},
    "ind_nom_pens_ult1":  {"index": 22, "name": "Pensiun Masuk (Pension Inflow)", "category": "Sinyal",  "sellable": False, "min_age": 30, "min_income": 0},
    "ind_recibo_ult1":    {"index": 23, "name": "Autodebet Tagihan",           "category": "Layanan",    "sellable": True,  "min_age": 18, "min_income": 0},
}

NUM_CLASSES = len(PRODUCT_CATALOG)
CODE_BY_INDEX = {v["index"]: k for k, v in PRODUCT_CATALOG.items()}
SELLABLE_INDICES = {v["index"] for v in PRODUCT_CATALOG.values() if v["sellable"]}

# ---------------------------------------------------------------------------
# Customer segments (source field `segmento` in the raw dataset)
# ---------------------------------------------------------------------------
SEGMENT_MAP = {
    "01 - TOP": "Nasabah Prioritas",
    "02 - PARTICULARES": "Nasabah Reguler",
    "03 - UNIVERSITARIO": "Nasabah Muda",
}
DEFAULT_SEGMENT = "Nasabah Reguler"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
LAG_MONTHS = [1, 2, 3, 4, 5]
VELOCITY_WINDOWS = [3, 6]

# ---------------------------------------------------------------------------
# Serving policy / guardrails
# ---------------------------------------------------------------------------
FREQUENCY_CAP_MAX_EXPOSURES = 3
FREQUENCY_CAP_WINDOW_DAYS = 7
TOP_K_CANDIDATES = 5          # candidates considered per request before frequency-cap filtering
EXPLORATION_EPSILON = 0.10    # probability of serving a randomized exploration candidate instead of the greedy top pick

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# API auth: demo default only — override via AD_ENGINE_API_KEY env var in any real deployment.
DEFAULT_API_KEY = "demo-local-key-change-me"
