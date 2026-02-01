from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from custom.load_dataset import load_records_subset

RESULTS_DIR = Path("custom/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def recreate_collection(name: str, dim: int) -> Collection:
    """Drop and recreate a Milvus collection with a simple (id, vector) schema.

    The collection is created without an index to measure raw ingestion performance.
    """
    if utility.has_collection(name):
        utility.drop_collection(name)

    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Raw ingest without indexing")
    return Collection(name=name, schema=schema)

def cleanup_milvus(dataset: str):
    """Remove previously created benchmark collections for the given dataset prefix."""
    prefix = dataset.replace("-", "_")
    for name in utility.list_collections():
        if name.startswith(prefix):
            utility.drop_collection(name)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="glove-100-angular")
    p.add_argument("--amount", required=True, help="Number of vectors or ALL")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="19530")
    args = p.parse_args()

    if args.amount.upper() == "ALL":
        amount = None
        amount_label = "all"
    else:
        amount = int(args.amount)
        amount_label = str(amount)

    records, dim = load_records_subset(args.dataset, amount, normalize=False)

    ids = [r["id"] for r in records]
    vectors = [r["vector"] for r in records]

    connections.connect("default", host=args.host, port=args.port)

    # Ensure a clean benchmark run: drop any collections created for this dataset.
    cleanup_milvus(args.dataset)

    col_name = f"{args.dataset.replace('-', '_')}_{amount_label}_no_index"
    col = recreate_collection(col_name, dim)

    # Raw vector load (No index)
    start = time.perf_counter()

    for i in range(0, len(records), args.batch_size):
        batch_ids = ids[i : i + args.batch_size]
        batch_vecs = vectors[i : i + args.batch_size]
        col.insert([batch_ids, batch_vecs])

    col.flush()
    load_time = time.perf_counter() - start

    result = {
        "db": "milvus",
        "dataset": args.dataset,
        "loaded_vectors": len(records),
        "dimension": dim,
        "batch_size": args.batch_size,
        "vector_load_time_sec": load_time,
        "throughput_vec_per_sec": len(records) / load_time if load_time > 0 else None,
        "index_present": False,
        "search_ready": False,
        "mode": "raw_ingest_only",
        "collection": col_name,
    }

    out = RESULTS_DIR / f"milvus_ingest_no_index_{col_name}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
