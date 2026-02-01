from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import Collection, connections, utility, DataType

from benchmark.dataset import Dataset
from benchmark.config_read import read_dataset_config

RESULTS_DIR = Path("custom/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Optional CPU metrics
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

@dataclass
class QueryRunResult:
    latencies_ms: List[float]
    total_time_sec: float
    qps: float
    cpu_process_avg_pct: Optional[float]
    cpu_system_avg_pct: Optional[float]
    returned_topk_ids: List[List[int]]

def connect_milvus(host: str, port: str) -> None:
    connections.connect("default", host=host, port=port)

def get_collection(name: str) -> Collection:
    if not utility.has_collection(name):
        raise RuntimeError(f"Collection '{name}' not found in Milvus.")
    col = Collection(name)
    col.load()
    return col

def _escape_str(s: str) -> str:
    # Milvus expr strings are quoted with double-quotes; escape backslash and quotes
    return s.replace("\\", "\\\\").replace('"', '\\"')

def build_expr_from_meta_conditions(
    meta_conditions: Dict[str, Any],
    field_types: Dict[str, DataType],
) -> str:
    """
    Convert the dataset meta_conditions structure into a Milvus boolean expression.

    Expected dataset shape: {"and": [ { <field>: { "match": { "value": X } } } ]}

    This implementation intentionally supports exactly one AND clause (single filter per query),
    matching the dataset behavior used in the benchmark.
    """
    if not isinstance(meta_conditions, dict) or "and" not in meta_conditions:
        raise ValueError(f"Unsupported meta_conditions shape: {meta_conditions}")

    and_list = meta_conditions.get("and")
    if not isinstance(and_list, list) or len(and_list) != 1:
        raise ValueError(f"Expected exactly 1 AND filter, got: {meta_conditions}")

    clause = and_list[0]
    if not isinstance(clause, dict) or len(clause) != 1:
        raise ValueError(f"Unsupported clause: {clause}")

    field = next(iter(clause.keys()))
    spec = clause[field]
    # spec: {"match":{"value": ...}}
    if not isinstance(spec, dict) or "match" not in spec:
        raise ValueError(f"Unsupported filter spec: {spec}")

    match = spec.get("match", {})
    value = match.get("value", None)

    dtype = field_types.get(field, None)

     # Use schema datatype to decide whether to generate a numeric or string comparison.
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
        if value is None:
            # Milvus numeric comparisons cannot use None; fail fast to surface dataset issues.
            raise ValueError(f"Numeric field '{field}' has None value")
        return f"{field} == {int(value)}"

    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        if value is None:
            raise ValueError(f"Float field '{field}' has None value")
        return f"{field} == {float(value)}"

    # VARCHAR / unknown => streat as string.
    if value is None:
        # Milvus VARCHAR does not accept None in expr; match an empty string as a safe fallback.
        return f'{field} == ""'
    return f'{field} == "{_escape_str(str(value))}"'

def load_queries_with_filters(
    dataset_name: str,
    limit: Optional[int],
    normalize: bool,
) -> Tuple[List[List[float]], List[Dict[str, Any]], List[List[int]]]:
    """
    Load query vectors, per-query metadata filter conditions, and optional ground truth.

    Returns:
      - query_vectors: list of float vectors
      - meta_conditions_list: list of meta_conditions dicts (one per query)
      - gt_ids: list of ground-truth neighbor ids (expected_result); may contain empty lists
    """
    cfgs = read_dataset_config()
    if dataset_name not in cfgs:
        raise KeyError(f"Dataset '{dataset_name}' not found in benchmark config")

    ds = Dataset(cfgs[dataset_name])
    ds.download()
    reader = ds.get_reader(normalize=normalize)

    if not hasattr(reader, "read_queries"):
        raise RuntimeError("Dataset reader has no read_queries().")

    query_vectors: List[List[float]] = []
    meta_conditions_list: List[Dict[str, Any]] = []
    gt_ids: List[List[int]] = []

    for i, q in enumerate(reader.read_queries()):
        if limit is not None and i >= limit:
            break

        qvec = getattr(q, "vector", None)
        if qvec is None:
            raise RuntimeError("Query object has no .vector field")
        query_vectors.append(list(qvec))

        mc = getattr(q, "meta_conditions", None)
        if not isinstance(mc, dict):
            raise RuntimeError(f"Query has no meta_conditions dict: {mc}")
        meta_conditions_list.append(mc)

        exp = getattr(q, "expected_result", None)
        gt_ids.append(list(exp) if exp is not None else [])

    if not query_vectors:
        raise RuntimeError("No queries loaded.")

    return query_vectors, meta_conditions_list, gt_ids

def recall_at_k(found: List[int], gt: List[int], k: int) -> Optional[float]:
    if not gt:
        return None
    gt_k = gt[:k]
    if not gt_k:
        return None
    return len(set(found[:k]) & set(gt_k)) / float(k)

def mrr_at_k(found: List[int], gt: List[int], k: int) -> Optional[float]:
    if not gt:
        return None
    gt_set = set(gt[:k])
    for rank, doc_id in enumerate(found[:k], start=1):
        if doc_id in gt_set:
            return 1.0 / rank
    return 0.0

def run_filtered_knn_queries(
    col: Collection,
    query_vectors: List[List[float]],
    meta_conditions_list: List[Dict[str, Any]],
    topk: int,
    metric_type: str,
    ef: int,
    vector_field: str = "vector",
    print_expr_samples: int = 3,
) -> Tuple[QueryRunResult, List[str]]:
    latencies_ms: List[float] = []
    returned_topk_ids: List[List[int]] = []
    exprs: List[str] = []

    # Build a field->datatype map from the live Milvus collection schema.
    field_types: Dict[str, DataType] = {f.name: f.dtype for f in col.schema.fields}

    # CPU sampling
    proc = psutil.Process(os.getpid()) if psutil else None
    cpu_proc_samples: List[float] = []
    cpu_sys_samples: List[float] = []
    if psutil:
        _ = proc.cpu_percent(interval=None)  # type: ignore
        _ = psutil.cpu_percent(interval=None)  # type: ignore

    # Search-time parameters for the vector index
    search_params: Dict[str, Any] = {"metric_type": metric_type, "params": {"ef": ef}}

    t0 = time.perf_counter()

    for idx, (qv, mc) in enumerate(zip(query_vectors, meta_conditions_list)):
        # Convert meta_conditions to a Milvus boolean expr
        expr = build_expr_from_meta_conditions(mc, field_types)
        exprs.append(expr)

        q_start = time.perf_counter()
        res = col.search(
            data=[qv],
            anns_field=vector_field,
            param=search_params,
            limit=topk,
            expr=expr,
            output_fields=["id"],
            timeout=100,
        )
        q_end = time.perf_counter()

        hits = res[0]
        ids = [int(h.id) for h in hits]
        returned_topk_ids.append(ids)
        latencies_ms.append((q_end - q_start) * 1000.0)

        if psutil:
            cpu_proc_samples.append(proc.cpu_percent(interval=None))  # type: ignore
            cpu_sys_samples.append(psutil.cpu_percent(interval=None))  # type: ignore

    total_time_sec = time.perf_counter() - t0
    qps = (len(query_vectors) / total_time_sec) if total_time_sec > 0 else 0.0

    cpu_process_avg = (sum(cpu_proc_samples) / len(cpu_proc_samples)) if cpu_proc_samples else None
    cpu_system_avg = (sum(cpu_sys_samples) / len(cpu_sys_samples)) if cpu_sys_samples else None

    run = QueryRunResult(
        latencies_ms=latencies_ms,
        total_time_sec=total_time_sec,
        qps=qps,
        cpu_process_avg_pct=cpu_process_avg,
        cpu_system_avg_pct=cpu_system_avg,
        returned_topk_ids=returned_topk_ids,
    )
    return run, exprs

def print_debug_info(col: Collection):
    print("\n=== Collection debug ===")
    print("\nname:", col.name)
    print("\nschema fields:", [f"{f.name}:{f.dtype}" for f in col.schema.fields])
    try:
        idx = col.indexes
        print("\nindexes:", [f"{i.field_name}:{i.params}" for i in idx])
    except Exception as e:
        print("could not read indexes:", e)

    # Best-effort segment info
    try:
        seg = utility.get_query_segment_info(col.name)
        print(f"\nquery segments: {len(seg)}")
    except Exception as e:
        print("could not read query segment info:", e)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--metric", default="COSINE")
    p.add_argument("--ef", type=int, default=512)  # ✅ default όπως ζήτησες
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="19530")
    p.add_argument("--print-expr-samples", type=int, default=3)
    p.add_argument("--debug", action="store_true", help="Print schema/index/segment info (best-effort)")

    args = p.parse_args()

    connect_milvus(args.host, args.port)
    col = get_collection(args.collection)

    if args.debug:
        print_debug_info(col)

    qvecs, meta_conditions_list, gt_ids = load_queries_with_filters(
        args.dataset, limit=args.n_queries, normalize=args.normalize
    )

    print(
        f"\n=== Running FILTERED search: topk={args.topk}, n_queries={len(qvecs)}, "
        f"metric={args.metric}, ef={args.ef}, parallel=1 ==="
    )

    run, exprs = run_filtered_knn_queries(
        col=col,
        query_vectors=qvecs,
        meta_conditions_list=meta_conditions_list,
        topk=args.topk,
        metric_type=args.metric,
        ef=args.ef,
        print_expr_samples=args.print_expr_samples,
    )

    # Quality
    recalls: List[float] = []
    mrrs: List[float] = []
    hits = 0
    valid = 0
    for found, gt in zip(run.returned_topk_ids, gt_ids):
        r = recall_at_k(found, gt, args.topk)
        m = mrr_at_k(found, gt, args.topk)
        if r is None or m is None:
            continue
        valid += 1
        recalls.append(r)
        mrrs.append(m)
        if r > 0:
            hits += 1

    quality = {
        "ground_truth_available_queries": valid,
        "recall_at_k_avg": (sum(recalls) / len(recalls)) if recalls else None,
        "mrr_at_k_avg": (sum(mrrs) / len(mrrs)) if mrrs else None,
        "hit_rate_at_k": (hits / valid) if valid > 0 else None,
    }

    lat = run.latencies_ms
    result: Dict[str, Any] = {
        "db": "milvus",
        "mode": "filtered_vector_search",
        "dataset": args.dataset,
        "collection": args.collection,
        "topk": args.topk,
        "n_queries": len(qvecs),
        "metric": args.metric,
        "ef": args.ef,
        "parallel": 1,
        "total_time_sec": run.total_time_sec,
        "throughput": run.qps,
        "latency_ms": {
            "avg": (sum(lat) / len(lat)) if lat else None,
            "min": min(lat) if lat else None,
            "max": max(lat) if lat else None,
        },
        "cpu": {
            "psutil_available": psutil is not None,
            "process_cpu_avg_pct": run.cpu_process_avg_pct,
            "system_cpu_avg_pct": run.cpu_system_avg_pct,
        },
        "quality_vs_ground_truth": quality,
    }

    out = RESULTS_DIR / f"milvus_filter_query_{args.collection}_topk{args.topk}_q{len(qvecs)}_ef{args.ef}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
