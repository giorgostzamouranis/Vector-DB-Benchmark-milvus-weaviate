from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import weaviate
from weaviate.util import generate_uuid5

from benchmark.config_read import read_dataset_config
from benchmark.dataset import Dataset

RESULTS_DIR = Path("custom/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def cleanup_weaviate(client: weaviate.Client, class_name: str):
    # Drop the class if it already exists
    if client.schema.exists(class_name):
        client.schema.delete_class(class_name)

def _to_int(x: Any) -> Optional[int]:
    # Best-effort conversion for Weaviate "int" properties.
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None

def _to_float(x: Any) -> Optional[float]:
    # Best-effort conversion for Weaviate "number" properties.
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _to_str(x: Any, max_len: int) -> str:
    # Convert to string and truncate to keep payload size bounded (Weaviate doesn't enforce VARCHAR lengths,
    # but this keeps parity with the Milvus ingest settings).
    if x is None:
        return ""
    s = str(x)
    return s[:max_len] if len(s) > max_len else s

def load_records_subset_with_metadata(
    dataset_name: str,
    limit: int | None,
    normalize: bool = False,
) -> Tuple[List[Dict[str, Any]], int, List[str], Dict[str, Any]]:
    """
    Load up to `limit` records from the dataset reader, collecting:
      - records[i] = {"id": int, "vector": List[float], "metadata": Dict[str, Any]}
      - dim: vector dimension
      - meta_keys: union of metadata keys (plus any schema keys from config)
      - schema_cfg: schema typing config from the dataset config (may be empty)
    """
    cfgs = read_dataset_config()
    if dataset_name not in cfgs:
        raise KeyError(f"Dataset '{dataset_name}' not found in datasets config")

    cfg = cfgs[dataset_name]
    schema_cfg = cfg.get("schema") if isinstance(cfg.get("schema"), dict) else {}

    ds = Dataset(cfg)
    ds.download()
    reader = ds.get_reader(normalize=normalize)

    out: List[Dict[str, Any]] = []
    dim: Optional[int] = None

    schema_keys = list(schema_cfg.keys()) if isinstance(schema_cfg, dict) else []
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
        raise RuntimeError("No records loaded from reader.read_data()")

    meta_keys = sorted(meta_key_set)
    return out, dim, meta_keys, schema_cfg

def build_weaviate_class_schema(
    class_name: str,
    meta_keys: List[str],
    schema_cfg: Dict[str, Any],
    varchar_keyword: int,
    varchar_text: int,
    scalar_index: bool,
    vector_index_type: str = "hnsw",
) -> Dict[str, Any]:
    """
    Build a Weaviate class schema for "vector + scalar filters" experiments.

    Notes:
      - We set vectorizer="none" because vectors are supplied manually.
      - Weaviate's "scalar index" equivalent is `indexFilterable` on each property.
      - For exact-match "keyword-like" strings we use tokenization="field".
      - For longer free text we use tokenization="word".
    """
    properties = [
    {"name": "doc_id", "dataType": ["int"], "indexFilterable": bool(scalar_index)}
    ]
    
    for k in meta_keys:
        t = schema_cfg.get(k)

        if t == "int":
            dt = ["int"]
            tokenization = None
        elif t == "float":
            dt = ["number"]
            tokenization = None
        elif t == "text":
            dt = ["text"]
            tokenization = "word"  
        else:
            dt = ["text"]
            tokenization = "field"  

        prop = {
            "name": k,
            "dataType": dt,
            "indexFilterable": bool(scalar_index),
        }

        if tokenization is not None:
            prop["tokenization"] = tokenization

        properties.append(prop)

    schema = {
        "class": class_name,
        "description": "Ingest with scalar filter fields + vector index (Weaviate)",
        "vectorizer": "none",
        "properties": properties,
        "vectorIndexType": vector_index_type,
        "vectorIndexConfig": {
            "distance": "cosine"
        }
    }

    return schema

def print_schema_debug(client: weaviate.Client, class_name: str):
    print("\n=== Weaviate Schema Debug ===")
    sch = client.schema.get(class_name)
    print("\nclass:", sch["class"])
    props = sch.get("properties", [])
    print("\nproperties:", [p["name"] for p in props])
    print("\nvectorIndexType:", sch.get("vectorIndexType"))
    print("\nvectorIndexConfig:", sch.get("vectorIndexConfig"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--amount", required=True, help="Number of vectors or ALL")
    p.add_argument("--batch-size", type=int, default=100)

    # Weaviate params
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--class-name", default=None, help="Optional override. Default is derived from dataset+amount.")

    # string sizing (kept for parity with Milvus ingest settings).
    p.add_argument("--varchar-keyword", type=int, default=256)
    p.add_argument("--varchar-text", type=int, default=2048)

    # Weaviate equivalent of scalar indexes: set indexFilterable=True on all properties.
    p.add_argument("--scalar-index", action="store_true", help="Enable indexFilterable=True for all metadata fields")

    args = p.parse_args()

    if args.amount.upper() == "ALL":
        amount = None
        amount_label = "all"
    else:
        amount = int(args.amount)
        amount_label = str(amount)

    # Load to RAM (vectors + metadata) and discover all metadata keys.
    records, dim, meta_keys, schema_cfg = load_records_subset_with_metadata(
        args.dataset, amount, normalize=False
    )
    # doc_id is added explicitly as a property; don't also treat it as a metadata key.
    meta_keys = [k for k in meta_keys if k != "doc_id"]

    # connect weaviate
    client = weaviate.Client(args.url, timeout_config=(60, 1000))

    class_name = args.class_name or f"{args.dataset.replace('-', '_')}_{amount_label}_filters".title().replace("_", "")

    # cleanup + create schema
    cleanup_weaviate(client, class_name)

    schema = build_weaviate_class_schema(
        class_name=class_name,
        meta_keys=meta_keys,
        schema_cfg=schema_cfg,
        varchar_keyword=args.varchar_keyword,
        varchar_text=args.varchar_text,
        scalar_index=args.scalar_index,
        vector_index_type="hnsw",
    )
    client.schema.create_class(schema)

    print_schema_debug(client, class_name)

    # Batch insert objects + vectors.
    #    We provide:
    #      - data_object: scalar properties (doc_id + metadata fields)
    #      - vector: the embedding
    #      - uuid: deterministic (based on doc_id) so reruns are stable
    load_start = time.perf_counter()

    with client.batch as batch:
        batch.batch_size = args.batch_size
        batch.dynamic = False

        for r in records:
            md = r.get("metadata", {}) or {}

            obj = {}
            obj["doc_id"] = int(r["id"])

            for k in meta_keys:
                t = schema_cfg.get(k)
                v = md.get(k, None)

                if t == "int":
                    obj[k] = _to_int(v)
                elif t == "float":
                    obj[k] = _to_float(v)
                elif t == "text":
                    obj[k] = _to_str(v, args.varchar_text)
                else:
                    obj[k] = _to_str(v, args.varchar_keyword)
                     
            # Weaviate wants vectors separately from the object properties.
            uuid = generate_uuid5(str(r["id"]))
            batch.add_data_object(
                data_object=obj,
                class_name=class_name,
                uuid=uuid,
                vector=r["vector"],
            )

    vector_load_time = time.perf_counter() - load_start

    result = {
        "db": "weaviate",
        "dataset": args.dataset,
        "loaded_vectors": len(records),
        "dimension": dim,
        "batch_size": args.batch_size,
        "vector_load_time_sec": vector_load_time,
        "total_search_ready_time_sec": vector_load_time,
        "throughput_vec_per_sec": (len(records) / vector_load_time) if vector_load_time > 0 else None,
        "index_type": "HNSW",
        "index_build_mode": "online",
        "class": class_name,
        "filter_fields": meta_keys,
        "scalar_index": args.scalar_index,
    }

    out = RESULTS_DIR / f"weaviate_ingest_indexing_{class_name}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()

