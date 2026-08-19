import _bootstrap  # noqa: F401

import polars as pl

from src import config
from src.data_prep import build_serving_snapshot
from src.feature_store import FeatureStore

if __name__ == "__main__":
    monthly = pl.read_parquet(config.FEATURES_MONTHLY_PATH)
    records = build_serving_snapshot(monthly)
    print(f"Built live feature snapshots for {len(records):,} customers")

    store = FeatureStore()
    for rec in records:
        store.write_customer(
            customer_id=rec["ncodpers"],
            features=rec["features"],
            current_holdings=rec["current_holding_indices"],
            segment=rec["segment"],
            age=rec["age"],
            income=rec["income"],
            consent=rec["consent"],
        )
    print(f"Pushed {len(records):,} customer feature records into the feature store.")
    if records:
        print(f"Try: curl -s -X POST http://localhost:8000/v1/ad/serve "
              f"-H \"X-API-Key: {config.DEFAULT_API_KEY}\" -H \"Content-Type: application/json\" "
              f"-d '{{\"customer_id\": {records[0]['ncodpers']}}}'")
