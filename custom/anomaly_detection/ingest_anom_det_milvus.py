from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


def load_good_records(folder: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Load all *.json files under folder. Each JSON is a list of {vector, filter}."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(folder.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in: {folder}")

    all_records: List[Dict[str, Any]] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"{fp} is not a JSON list.")
            all_records.extend(data)

    if not all_records:
        raise ValueError("Loaded 0 records.")

    # Infer dim
    v0 = all_records[0].get("vector")
    if not isinstance(v0, list) or not v0:
        raise ValueError("First record does not contain a non-empty 'vector' list.")
    dim = len(v0)

    # Basic validation
    for i, r in enumerate(all_records[:1000]):  # sample-check first 1000
        v = r.get("vector")
        if not isinstance(v, list) or len(v) != dim:
            raise ValueError(f"Bad vector at record {i}: expected dim={dim}, got {None if not isinstance(v, list) else len(v)}")
        if "filter" not in r:
            raise ValueError(f"Missing 'filter' at record {i}")

    return all_records, dim


def recreate_collection(name: str, dim: int, filter_max_len: int = 64) -> Collection:
    if utility.has_collection(name):
        utility.drop_collection(name)

    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema("filter", DataType.VARCHAR, max_length=filter_max_len),
    ]
    schema = CollectionSchema(fields, description="Custom anomaly_detection good vectors")
    return Collection(name=name, schema=schema)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="custom/anomaly_detection/good", help="Folder with JSON files")
    p.add_argument("--collection", default="custom_anomaly_good", help="Milvus collection name")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="19530")

    # HNSW params
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--efc", type=int, default=200)
    args = p.parse_args()

    folder = Path(args.data_dir)
    records, dim = load_good_records(folder)
    print(f"Loaded {len(records)} records from {folder} (dim={dim})")

    connections.connect("default", host=args.host, port=args.port)

    col = recreate_collection(args.collection, dim)

    # Prepare columns
    ids = list(range(1, len(records) + 1))
    vectors = [r["vector"] for r in records]
    filters = [str(r["filter"]) for r in records]

    t0 = time.perf_counter()

    for i in range(0, len(records), args.batch_size):
        col.insert([
            ids[i : i + args.batch_size],
            vectors[i : i + args.batch_size],
            filters[i : i + args.batch_size],
        ])

    col.flush()
    t_load = time.perf_counter() - t0

    # Create vector index
    t1 = time.perf_counter()
    col.create_index(
        field_name="vector",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": args.m, "efConstruction": args.efc},
        },
    )

    # scalar index on filter for faster filtered search
    try:
        col.create_index(
            field_name="filter",
            index_params={"index_type": "INVERTED"},
        )
    except Exception as e:
        print(f"Note: could not create INVERTED index on 'filter' ({e})")

    col.load()
    t_index = time.perf_counter() - t1

    print(
        json.dumps(
            {
                "db": "milvus",
                "collection": args.collection,
                "loaded_vectors": len(records),
                "dimension": dim,
                "batch_size": args.batch_size,
                "vector_load_time_sec": t_load,
                "index_build_time_sec": t_index,
                "total_search_ready_time_sec": t_load + t_index,
                "throughput_vec_per_sec": (len(records) / t_load) if t_load > 0 else None,
                "index_type": "HNSW",
                "metric": "COSINE",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()