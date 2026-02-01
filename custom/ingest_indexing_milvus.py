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

# Directory used to store benchmark result artifacts (JSON).
RESULTS_DIR = Path("custom/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def recreate_collection(name: str, dim: int) -> Collection:
    """Create a fresh collection with an (id, vector) schema for controlled benchmarks."""
    if utility.has_collection(name):
        utility.drop_collection(name)

    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Controlled ingest experiment")
    
    return Collection(name=name, schema=schema, num_shards=2) #for distributed mode
    #return Collection(name=name, schema=schema) #for standalone mode

def cleanup_milvus(dataset: str):
    """Drop prior benchmark collections for the dataset to avoid cross-run interference."""
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

    # Prepare columnar data for Milvus insert
    ids = [r["id"] for r in records]
    vectors = [r["vector"] for r in records]

    connections.connect("default", host=args.host, port=args.port)

    # Ensure a clean benchmark run by removing collections from previous runs for this dataset.
    cleanup_milvus(args.dataset)

    col_name = f"{args.dataset.replace('-', '_')}_{amount_label}"
    col = recreate_collection(col_name, dim)
    
    # Vector ingestion
    load_start = time.perf_counter()

    for i in range(0, len(records), args.batch_size):
        batch_ids = ids[i : i + args.batch_size]
        batch_vecs = vectors[i : i + args.batch_size]
        col.insert([batch_ids, batch_vecs])

    col.flush()

    vector_load_time = time.perf_counter() - load_start

    # Index creation
    index_start = time.perf_counter()

    col.create_index(
        field_name="vector",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
        },
    )

    col.load()

    index_build_time = time.perf_counter() - index_start

    result = {
        "db": "milvus",
        "dataset": args.dataset,
        "loaded_vectors": len(records),
        "dimension": dim,
        "batch_size": args.batch_size,
        "vector_load_time_sec": vector_load_time,
        "index_build_time_sec": index_build_time,
        "total_search_ready_time_sec": vector_load_time + index_build_time,
        "throughput_vec_per_sec": len(records) / vector_load_time,
        "index_type": "HNSW",
        "index_build_mode": "offline",
        "collection": col_name,
    }

    out = RESULTS_DIR / f"milvus_ingest_indexing_{col_name}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()