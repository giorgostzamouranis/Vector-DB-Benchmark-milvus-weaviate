from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances


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

    v0 = all_records[0].get("vector")
    if not isinstance(v0, list) or not v0:
        raise ValueError("First record does not contain a non-empty 'vector' list.")
    dim = len(v0)

    # quick validation
    for i, r in enumerate(all_records[:1000]):
        v = r.get("vector")
        if not isinstance(v, list) or len(v) != dim:
            raise ValueError(f"Bad vector at record {i}: expected dim={dim}")
        if "filter" not in r:
            raise ValueError(f"Missing 'filter' at record {i}")

    return all_records, dim


def recreate_collection(client, name: str):
    # Weaviate convention: Capitalized
    class_name = name[0].upper() + name[1:]
    if client.collections.exists(class_name):
        client.collections.delete(class_name)

    client.collections.create(
        name=class_name,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            max_connections=32,
            ef_construction=256,
        ),
        properties=[
            Property(name="doc_id", data_type=DataType.INT),
            Property(name="filter", data_type=DataType.TEXT),
        ],
    )
    return class_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="custom/anomaly_detection/good", help="Folder with JSON files")
    p.add_argument("--class-name", default="customAnomalyGood", help="Weaviate collection/class name")
    p.add_argument("--batch-size", type=int, default=2000)

    # local defaults
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--grpc-port", type=int, default=50051)
    args = p.parse_args()

    folder = Path(args.data_dir)
    records, dim = load_good_records(folder)
    print(f"Loaded {len(records)} records from {folder} (dim={dim})")

    client = weaviate.connect_to_local(
        host=args.host,
        port=args.port,
        grpc_port=args.grpc_port,
    )
    if not client.is_ready():
        raise RuntimeError(f"Weaviate not ready at http://{args.host}:{args.port}")

    class_name = recreate_collection(client, args.class_name)
    col = client.collections.get(class_name)

    t0 = time.perf_counter()

    with col.batch.dynamic() as batch:
        batch.batch_size = args.batch_size
        for i, r in enumerate(records, start=1):
            batch.add_object(
                properties={
                    "doc_id": int(i),
                    "filter": str(r["filter"]),
                },
                vector=r["vector"],
            )

    elapsed = time.perf_counter() - t0

    print(
        json.dumps(
            {
                "db": "weaviate",
                "class": class_name,
                "loaded_vectors": len(records),
                "dimension": dim,
                "batch_size": args.batch_size,
                "total_search_ready_time_sec": elapsed,
                "throughput_vec_per_sec": (len(records) / elapsed) if elapsed > 0 else None,
                "index_type": "HNSW",
                "metric": "COSINE",
                "index_build_mode": "online",
            },
            indent=2,
        )
    )

    client.close()


if __name__ == "__main__":
    main()