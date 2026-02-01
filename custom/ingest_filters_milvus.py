from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from benchmark.config_read import read_dataset_config
from benchmark.dataset import Dataset

RESULTS_DIR = Path("custom/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def cleanup_milvus(dataset: str):
    # Drop any existing collections created for this dataset
    prefix = dataset.replace("-", "_")
    for name in utility.list_collections():
        if name.startswith(prefix):
            utility.drop_collection(name)

def _to_int64(x: Any) -> Optional[int]:
    # Best-effort conversion to int64 for Milvus scalar INT64 fields.
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None

def _to_str(x: Any, max_len: int) -> Optional[str]:
    # Convert to string and truncate to Milvus VARCHAR max_length.
    # Milvus VARCHAR cannot store None, so missing values become "".
    if x is None:
        return ""
    s = str(x)
    return s[:max_len] if len(s) > max_len else s

def load_records_subset_with_metadata(
    dataset_name: str,
    limit: int | None,
    normalize: bool = False,
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    # Load up to `limit` records from the dataset reader, collecting:
    #   - id, vector, and raw metadata per record
    #   - vector dimension (dim)
    #   - the union of metadata keys seen in the dataset (plus any schema keys from config)
    cfgs = read_dataset_config()
    if dataset_name not in cfgs:
        raise KeyError(f"Dataset '{dataset_name}' not found in datasets config")

    cfg = cfgs[dataset_name]
    ds = Dataset(cfg)
    ds.download()
    reader = ds.get_reader(normalize=normalize)

    out: List[Dict[str, Any]] = []
    dim: Optional[int] = None

    schema = cfg.get("schema")
    schema_keys: List[str] = list(schema.keys()) if isinstance(schema, dict) else []
    meta_key_set = set(schema_keys)

    for i, rec in enumerate(reader.read_data()):
        if limit is not None and i >= limit:
            break

        vec = rec.vector
        if dim is None:
            dim = len(vec)

        md = getattr(rec, "metadata", None)
        if isinstance(md, dict):
            meta_key_set.update(md.keys())
        else:
            md = {}

        out.append({"id": int(rec.id), "vector": list(vec), "metadata": md})

    if not out or dim is None:
        raise RuntimeError("No records were loaded from reader.read_data()")

    meta_keys = sorted(meta_key_set)
    return out, dim, meta_keys

def build_field_schemas_from_metadata(
    dataset_cfg_schema: Dict[str, Any] | None,
    meta_keys: List[str],
    varchar_max_keyword: int,
    varchar_max_text: int,
) -> List[FieldSchema]:
    # Build Milvus scalar field schemas from dataset config + discovered metadata keys.
    # Unknown types fall back to VARCHAR ("keyword-like" fields).
    schema = dataset_cfg_schema if isinstance(dataset_cfg_schema, dict) else {}
    fields: List[FieldSchema] = []

    for k in meta_keys:
        t = schema.get(k)
        if t == "int":
            fields.append(FieldSchema(k, DataType.INT64))
        elif t == "float":
            fields.append(FieldSchema(k, DataType.DOUBLE))
        elif t == "text":
            fields.append(FieldSchema(k, DataType.VARCHAR, max_length=varchar_max_text))
        else:
            fields.append(FieldSchema(k, DataType.VARCHAR, max_length=varchar_max_keyword))

    return fields

def recreate_collection_with_filters(
    name: str,
    dim: int,
    meta_field_schemas: List[FieldSchema],
) -> Collection:
    if utility.has_collection(name):
        utility.drop_collection(name)

    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
        *meta_field_schemas,
    ]
    schema = CollectionSchema(fields, description="Ingest with scalar filter fields + vector index")
    return Collection(name=name, schema=schema)

def print_collection_introspection(col: Collection):
    print("\n=== Collection introspection ===")
    print("\nname:", col.name)
    print("\nschema fields:", [f.name for f in col.schema.fields])
    try:
        idx = col.indexes
        print("\nindexes:", [f"{i.field_name}:{i.params}" for i in idx])
    except Exception as e:
        print("could not read indexes:", e)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--amount", required=True, help="Number of vectors or ALL")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="19530")

    p.add_argument("--varchar-keyword", type=int, default=256)
    p.add_argument("--varchar-text", type=int, default=2048)

    # Optional: create scalar indexes on ALL metadata fields (otherwise: only vector index).
    p.add_argument(
        "--scalar-index",
        default=None,
        choices=["INVERTED", "BITMAP"],
        help="Create scalar indexes for ALL metadata fields (optional). If omitted, no scalar indexes are created.",
    )

    args = p.parse_args()

    if args.amount.upper() == "ALL":
        amount = None
        amount_label = "all"
    else:
        amount = int(args.amount)
        amount_label = str(amount)

    # Load dataset config and optional schema typing for metadata fields.
    cfgs = read_dataset_config()
    if args.dataset not in cfgs:
        raise KeyError(f"Dataset '{args.dataset}' not found in config")
    cfg = cfgs[args.dataset]
    schema_cfg = cfg.get("schema") if isinstance(cfg.get("schema"), dict) else None

    # Load data into RAM (vectors + metadata) and discover all metadata keys.
    records, dim, meta_keys = load_records_subset_with_metadata(args.dataset, amount, normalize=False)

    #Build Milvus field schemas for metadata according to config types.
    meta_field_schemas = build_field_schemas_from_metadata(
        dataset_cfg_schema=schema_cfg,
        meta_keys=meta_keys,
        varchar_max_keyword=args.varchar_keyword,
        varchar_max_text=args.varchar_text,
    )

    connections.connect("default", host=args.host, port=args.port)
    cleanup_milvus(args.dataset)

    col_name = f"{args.dataset.replace('-', '_')}_{amount_label}_filters"
    col = recreate_collection_with_filters(col_name, dim, meta_field_schemas)

    # Prepare columnar data for insert() (Milvus expects one list per field, aligned by row index).
    ids = [r["id"] for r in records]
    vectors = [r["vector"] for r in records]
    meta_columns: Dict[str, List[Any]] = {k: [] for k in meta_keys}

    for r in records:
        md: Dict[str, Any] = r.get("metadata", {}) or {}
        for k in meta_keys:
            v = md.get(k, None)
            t = schema_cfg.get(k) if isinstance(schema_cfg, dict) else None
            if t == "int":
                meta_columns[k].append(_to_int64(v))
            elif t == "float":
                meta_columns[k].append(float(v) if v is not None else None)
            elif t == "text":
                meta_columns[k].append(_to_str(v, args.varchar_text))
            else:
                meta_columns[k].append(_to_str(v, args.varchar_keyword))

    insert_columns: List[List[Any]] = [ids, vectors] + [meta_columns[k] for k in meta_keys]

    # Insert into Milvus in batches and flush to persist segments.
    load_start = time.perf_counter()
    n = len(records)
    for i in range(0, n, args.batch_size):
        chunk = [col_data[i : i + args.batch_size] for col_data in insert_columns]
        col.insert(chunk)
    col.flush()
    vector_load_time = time.perf_counter() - load_start

    # Build the vector index (and optionally scalar indexes for filter fields).
    index_start = time.perf_counter()

    col.create_index(
        field_name="vector",
        index_params={"index_type": "HNSW", "metric_type": "COSINE"},
    )

    if args.scalar_index is not None:
        for fs in meta_field_schemas:
            try:
                col.create_index(
                    field_name=fs.name,
                    index_params={"index_type": args.scalar_index},
                )
            except Exception as e:
                print(f"[WARN] could not create scalar index for {fs.name} ({args.scalar_index}): {e}")

    # Load collection into memory so it is ready for search benchmarks.
    col.load()
    index_build_time = time.perf_counter() - index_start

    print_collection_introspection(col)

    result = {
        "db": "milvus",
        "dataset": args.dataset,
        "loaded_vectors": len(records),
        "dimension": dim,
        "batch_size": args.batch_size,
        "vector_load_time_sec": vector_load_time,
        "index_build_time_sec": index_build_time,
        "total_search_ready_time_sec": vector_load_time + index_build_time,
        "throughput_vec_per_sec": (len(records) / vector_load_time) if vector_load_time > 0 else None,
        "index_type": "HNSW",
        "index_build_mode": "offline",
        "collection": col_name,
        "filter_fields": meta_keys,

        "scalar_index": args.scalar_index,  # None or "INVERTED"/"BITMAP"
    }

    out = RESULTS_DIR / f"milvus_ingest_indexing_{col_name}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()