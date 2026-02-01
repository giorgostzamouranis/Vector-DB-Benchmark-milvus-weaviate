from __future__ import annotations
from weaviate.classes.config import Configure, VectorDistances, DataType, Property
from weaviate.classes.init import AdditionalConfig, Timeout
import argparse
import json
import time
from pathlib import Path

import weaviate

from custom.load_dataset import load_records_subset

RESULTS_DIR = Path("custom/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def recreate_class(client, class_name: str):
    # Normalize to Weaviate naming convention (starts with uppercase)
    class_name = class_name[0].upper() + class_name[1:]

    # Delete the class if it already exists
    if client.collections.exists(class_name):
        client.collections.delete(class_name)

    # Create the class/collection 
    # - vectorizer: none (we provide vectors)
    # - vector index: HNSW with the provided construction-time parameters
    # - properties: store an integer doc_id for mapping results back to dataset ids
    client.collections.create(
        name=class_name,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            max_connections=30,
            ef_construction=360,
        ),
        properties=[
            Property(
                name="doc_id",
                data_type=DataType.INT,
            )
        ],

    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="glove-100-angular")
    p.add_argument("--amount", required=True, help="Number of vectors or ALL")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--url", default="http://localhost:8080")
    args = p.parse_args()

    # Parse amount: "ALL" means ingest everything available in the dataset.
    if args.amount.upper() == "ALL":
        amount = None
        amount_label = "all"
    else:
        amount = int(args.amount)
        amount_label = str(amount)

    # Load vectors into RAM (ids + vectors) from the dataset.
    records, dim = load_records_subset(args.dataset, amount, normalize=False)
    
    client = weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051)# additional_config=AdditionalConfig(timeout=Timeout(init=200, query=600, insert=1200)))
    if not client.is_ready():
        raise RuntimeError(f"Weaviate not ready at {args.url}")

    class_name = f"{args.dataset.replace('-', '')}{amount_label}"
    class_name = class_name[0].upper() + class_name[1:]
    recreate_class(client, class_name)

    # get a handle for inserts/queries.
    col = client.collections.get(class_name)

    start = time.perf_counter()

    # Batch insert objects with vectors. HNSW is built online as objects are inserted.
    with col.batch.dynamic() as batch:
        batch.batch_size = args.batch_size
        for r in records:
            batch.add_object(
                properties={"doc_id": int(r["id"])},
                vector=r["vector"],
            )

    elapsed = time.perf_counter() - start

    result = {
        "db": "weaviate",
        "dataset": args.dataset,
        "loaded_vectors": len(records),
        "dimension": dim,
        "batch_size": args.batch_size,
        "total_search_ready_time_sec": elapsed,
        "throughput_vec_per_sec": (len(records) / elapsed) if elapsed > 0 else None,
        "index_type": "HNSW",
        "index_build_mode": "online",
        "load_includes_index": True,
        "class": class_name,
    }

    client.close()
    out = RESULTS_DIR / f"weaviate_ingest_{class_name}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()