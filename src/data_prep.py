"""Feature engineering pipeline for the hyper-personalization NBA model.

Builds, from the raw Santander monthly snapshots:
  1. `features_monthly.parquet` — one row per (customer, month) with lag /
     velocity / demographic features, the customer's net-new target list
     for that month (`added_products`), and the customer's holdings as of
     the *previous* month (used later for the active-holding guardrail).
  2. `train_exploded.parquet` — one row per (customer, month, added product)
     with a single `target_class` label, ready for LightGBM multiclass
     training. Rows where nothing was added in that month are excluded from
     positive training, per the net-new target definition.
  3. `artifacts/popularity_fallback.json` — global popularity ranking of
     net-new *sellable* products, used as the cold-start fallback when a
     customer has no feature-store entry yet.

Run: `python scripts/build_features.py --help`
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass

import polars as pl

from src import config

PRODUCT_CODES = list(config.PRODUCT_CATALOG.keys())


def extract_dataset() -> None:
    """Unzip the raw Santander CSV once (cached across runs)."""
    if config.RAW_CSV_PATH.exists():
        return
    config.RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {config.RAW_CSV_ZIP.name} -> {config.RAW_CSV_PATH} ...")
    with zipfile.ZipFile(config.RAW_CSV_ZIP) as z:
        with z.open("train_ver2.csv") as src, open(config.RAW_CSV_PATH, "wb") as dst:
            while chunk := src.read(1024 * 1024 * 8):
                dst.write(chunk)


def get_available_months() -> list[str]:
    months = (
        pl.scan_csv(config.RAW_CSV_PATH, has_header=True)
        .select(pl.col("fecha_dato").unique())
        .collect()["fecha_dato"]
        .sort()
        .to_list()
    )
    return months


_STRING_OVERRIDE_COLS = ["age", "antiguedad", "renta", "indrel_1mes", "conyuemp", "ult_fec_cli_1t"]


def load_raw(months: list[str], customer_sample_mod: int, seed: int) -> pl.DataFrame:
    """Load the requested months, deterministically sampling ~1/customer_sample_mod
    of customers (sampling is stable per-customer so a customer's full history
    across months is kept or dropped together)."""
    lf = pl.scan_csv(
        config.RAW_CSV_PATH,
        has_header=True,
        schema_overrides={c: pl.Utf8 for c in _STRING_OVERRIDE_COLS},
        null_values=["NA", ""],
    ).filter(pl.col("fecha_dato").is_in(months))

    if customer_sample_mod > 1:
        lf = lf.filter((pl.col("ncodpers") + seed) % customer_sample_mod == 0)

    df = lf.collect()

    df = df.with_columns(
        pl.col("fecha_dato").str.to_date("%Y-%m-%d"),
        pl.col("age").str.strip_chars().cast(pl.Int32, strict=False),
        pl.col("antiguedad").str.strip_chars().cast(pl.Int32, strict=False),
        pl.col("renta").cast(pl.Float64, strict=False),
        pl.col("segmento").str.strip_chars(),
        pl.col("nomprov").str.strip_chars(),
        pl.col("sexo").str.strip_chars(),
        *[pl.col(c).cast(pl.Utf8).str.strip_chars().cast(pl.Int8, strict=False).fill_null(0) for c in PRODUCT_CODES],
    )

    # antiguedad has a known raw-data quirk of large negative sentinels for brand-new customers
    df = df.with_columns(pl.col("antiguedad").clip(lower_bound=0))

    # provincial median income imputation, falling back to the global median
    global_median_renta = df["renta"].median()
    df = df.with_columns(
        pl.col("renta").fill_null(pl.col("renta").median().over("nomprov"))
    ).with_columns(pl.col("renta").fill_null(global_median_renta))

    median_age = df["age"].median()
    df = df.with_columns(pl.col("age").fill_null(median_age))

    df = df.with_columns(
        pl.col("segmento").replace(config.SEGMENT_MAP, default=config.DEFAULT_SEGMENT).alias("segment_name"),
        pl.col("sexo").replace({"H": 1, "V": 0}, default=-1).cast(pl.Int8).alias("sex_code"),
        pl.col("ind_actividad_cliente").cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int8),
    )

    return df.sort(["ncodpers", "fecha_dato"])


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add lag holdings, transition velocity, cold-start depth, seasonality,
    and the net-new target list, all computed per-customer over time."""
    prod = PRODUCT_CODES

    # monthly transition count = number of product flips vs the previous month
    df = df.with_columns(
        pl.sum_horizontal(
            [(pl.col(p) - pl.col(p).shift(1).over("ncodpers")).fill_null(0).abs() for p in prod]
        ).alias("monthly_transitions")
    )

    lag_exprs = []
    for lag in config.LAG_MONTHS:
        for p in prod:
            lag_exprs.append(pl.col(p).shift(lag).over("ncodpers").fill_null(0).cast(pl.Int8).alias(f"{p}_lag{lag}"))
    df = df.with_columns(lag_exprs)

    for window in config.VELOCITY_WINDOWS:
        vel = pl.sum_horizontal(
            [pl.col("monthly_transitions").shift(i).over("ncodpers").fill_null(0) for i in range(1, window + 1)]
        )
        df = df.with_columns(vel.alias(f"velocity_{window}m"))

    # how many prior months of real history we actually observe for this customer (0..5)
    df = df.with_columns(
        (pl.col("fecha_dato").cum_count().over("ncodpers") - 1).clip(upper_bound=len(config.LAG_MONTHS)).alias("history_depth")
    )

    df = df.with_columns(
        pl.col("fecha_dato").dt.month().alias("month_of_year"),
    ).with_columns(
        pl.col("month_of_year").is_in([6, 12]).cast(pl.Int8).alias("is_bonus_month")
    )

    # net-new target: y_k = max(0, holding_t - holding_{t-1})
    add_exprs = [(pl.col(p) - pl.col(f"{p}_lag1")).clip(lower_bound=0).alias(f"{p}_added") for p in prod]
    df = df.with_columns(add_exprs)

    added_indices_expr = pl.concat_list(
        [
            pl.when(pl.col(f"{p}_added") == 1).then(pl.lit(config.PRODUCT_CATALOG[p]["index"])).otherwise(None)
            for p in prod
        ]
    ).list.drop_nulls().alias("added_product_indices")
    df = df.with_columns(added_indices_expr)

    current_holdings_expr = pl.concat_list(
        [pl.when(pl.col(p) == 1).then(pl.lit(config.PRODUCT_CATALOG[p]["index"])).otherwise(None) for p in prod]
    ).list.drop_nulls().alias("current_holding_indices")
    df = df.with_columns(current_holdings_expr)

    return df


FEATURE_COLUMNS: list[str] = (
    [f"{p}_lag{lag}" for lag in config.LAG_MONTHS for p in PRODUCT_CODES]
    + [f"velocity_{w}m" for w in config.VELOCITY_WINDOWS]
    + ["history_depth", "month_of_year", "is_bonus_month", "age", "antiguedad", "renta", "sex_code", "ind_actividad_cliente"]
)


def explode_positives(monthly_df: pl.DataFrame) -> pl.DataFrame:
    """One row per (customer, month, added product) — the standard exploded
    multiclass training format for this style of net-new recommendation
    problem (a customer adding 2 products in one month yields 2 rows)."""
    exploded = monthly_df.filter(pl.col("added_product_indices").list.len() > 0).explode("added_product_indices")
    exploded = exploded.rename({"added_product_indices": "target_class"})
    return exploded.select(["ncodpers", "fecha_dato", "target_class", *FEATURE_COLUMNS])


def build_popularity_fallback(train_exploded: pl.DataFrame) -> dict:
    counts = (
        train_exploded.filter(pl.col("target_class").is_in(list(config.SELLABLE_INDICES)))
        .group_by("target_class")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    ranked = counts["target_class"].to_list()
    return {"ranked_sellable_indices": ranked}


def build_serving_snapshot(monthly_df: pl.DataFrame) -> list[dict]:
    """For each customer's latest observed month, roll the feature vector
    forward by one period so it represents "what we'd score right now" for
    a live ad request: this month's actual holdings become next period's
    lag1, the old lag1..lag4 shift down to lag2..lag5, and the calendar /
    tenure fields advance by one month. `velocity_*m` is carried forward
    unchanged as a light approximation (transition velocity moves slowly
    month-to-month; recomputing it exactly would require re-deriving the
    monthly transition history, not worth it for a demo feature push)."""
    latest = (
        monthly_df.sort(["ncodpers", "fecha_dato"])
        .group_by("ncodpers", maintain_order=True)
        .agg(pl.all().last())
    )

    records = []
    for row in latest.iter_rows(named=True):
        held = set(row["current_holding_indices"])
        vec: dict[str, float] = {}
        for p in PRODUCT_CODES:
            vec[f"{p}_lag1"] = 1 if config.PRODUCT_CATALOG[p]["index"] in held else 0
            vec[f"{p}_lag2"] = row[f"{p}_lag1"]
            vec[f"{p}_lag3"] = row[f"{p}_lag2"]
            vec[f"{p}_lag4"] = row[f"{p}_lag3"]
            vec[f"{p}_lag5"] = row[f"{p}_lag4"]
        vec["velocity_3m"] = row["velocity_3m"]
        vec["velocity_6m"] = row["velocity_6m"]
        vec["history_depth"] = min(row["history_depth"] + 1, len(config.LAG_MONTHS))
        next_month = (row["month_of_year"] % 12) + 1
        vec["month_of_year"] = next_month
        vec["is_bonus_month"] = 1 if next_month in (6, 12) else 0
        vec["age"] = row["age"]
        vec["antiguedad"] = row["antiguedad"] + 1
        vec["renta"] = row["renta"]
        vec["sex_code"] = row["sex_code"]
        vec["ind_actividad_cliente"] = row["ind_actividad_cliente"]

        # ~90% opt-in rate, deterministic per customer for a reproducible demo
        consent = (row["ncodpers"] % 10) != 0

        records.append({
            "ncodpers": row["ncodpers"],
            "segment": row["segment_name"],
            "age": row["age"],
            "income": row["renta"],
            "current_holding_indices": sorted(held),
            "consent": consent,
            "features": [vec[c] for c in FEATURE_COLUMNS],
        })
    return records


def main(months_back: int, customer_sample_mod: int, seed: int, eval_month_offset: int) -> None:
    extract_dataset()
    all_months = get_available_months()
    months = all_months[-months_back:]
    print(f"Using {len(months)} months: {months[0]} .. {months[-1]} (sample 1/{customer_sample_mod} customers)")

    raw = load_raw(months, customer_sample_mod, seed)
    print(f"Loaded {raw.height:,} customer-month rows for {raw['ncodpers'].n_unique():,} customers")

    monthly = engineer_features(raw)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    keep_cols = list(dict.fromkeys([
        "ncodpers", "fecha_dato", "segment_name",
        "added_product_indices", "current_holding_indices",
        *FEATURE_COLUMNS,
    ]))
    monthly.select(keep_cols).write_parquet(config.FEATURES_MONTHLY_PATH)
    print(f"Wrote {config.FEATURES_MONTHLY_PATH} ({monthly.height:,} rows)")

    eval_month = months[-eval_month_offset]
    train_months = [m for m in months if m < eval_month]
    train_monthly = monthly.filter(pl.col("fecha_dato").dt.strftime("%Y-%m-%d").is_in(train_months))

    exploded = explode_positives(train_monthly)
    exploded.write_parquet(config.TRAIN_EXPLODED_PATH)
    print(f"Wrote {config.TRAIN_EXPLODED_PATH} ({exploded.height:,} training rows, eval_month held out = {eval_month})")

    popularity = build_popularity_fallback(exploded)
    config.POPULARITY_PATH.write_text(json.dumps(popularity, indent=2))

    with open(config.ARTIFACTS_DIR / "eval_month.json", "w") as f:
        json.dump({"eval_month": str(eval_month)}, f)

    with open(config.FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print("Feature engineering complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build features from the Santander proxy dataset")
    parser.add_argument("--months", type=int, default=8, help="how many of the most recent monthly snapshots to use")
    parser.add_argument("--sample-mod", type=int, default=5, help="keep 1 in N customers (1 = full dataset)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-month-offset", type=int, default=1, help="hold out the Nth-from-last month for evaluation")
    args = parser.parse_args()
    main(args.months, args.sample_mod, args.seed, args.eval_month_offset)
