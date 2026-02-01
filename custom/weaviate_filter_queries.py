from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import weaviate

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

def connect_weaviate(url: str, timeout_connect: float, timeout_read: float) -> weaviate.Client:
    return weaviate.Client(url, timeout_config=(timeout_connect, timeout_read))

def cleanup_none(s: Any) -> Any:
    # Normalize None values for filter inputs (Weaviate expects a concrete value for valueText).
    return "" if s is None else s

def build_where_from_meta_conditions(
    meta_conditions: Dict[str, Any],
    prop_type_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Convert a dataset meta_conditions structure into a Weaviate GraphQL "where" filter.

    Expected meta_conditions shape: {"and": [ { <field>: { "match": { "value": X } } } ]}
    We support exactly 1 clause (same as Milvus version).

    Returns a Weaviate GraphQL where filter dict.
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

    if not isinstance(spec, dict) or "match" not in spec:
        raise ValueError(f"Unsupported filter spec: {spec}")

    match = spec.get("match", {})
    value = match.get("value", None)
    
    t = prop_type_map.get(field, "text")  # If the schema type is unknown, we default to "text".

    where: Dict[str, Any] = {
        "path": [field],
        "operator": "Equal",
    }

    if t == "int":
        where["valueInt"] = int(0 if value is None else int(value))
    elif t == "number":
        where["valueNumber"] = float(0.0 if value is None else float(value))
    else:
        # text / keyword-like
        where["valueText"] = str(cleanup_none(value))

    return where

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
      - gt_ids: list of ground-truth neighbor ids (ints);
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
        exp_ids = list(exp) if exp is not None else []
        gt_ids.append([int(x) for x in exp_ids])

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


def get_schema_maps(client: weaviate.Client, class_name: str) -> Tuple[Dict[str, str], Dict[str, bool]]:
    """
    Read the Weaviate class schema and build helper maps for query construction.

    Returns:
      - prop_type_map: property -> "int" | "number" | "text" (normalized)
      - prop_filterable_map: property -> indexFilterable flag
    """
    sch = client.schema.get(class_name)
    props = sch.get("properties", []) or []

    prop_type_map: Dict[str, str] = {}
    prop_filterable_map: Dict[str, bool] = {}

    for p in props:
        name = p.get("name")
        dt = (p.get("dataType") or ["text"])[0]
        # Weaviate datatypes we used: int, number, text
        if dt == "int":
            prop_type_map[name] = "int"
        elif dt == "number":
            prop_type_map[name] = "number"
        else:
            prop_type_map[name] = "text"

        prop_filterable_map[name] = bool(p.get("indexFilterable", False))

    return prop_type_map, prop_filterable_map


def get_object_count(client: weaviate.Client, class_name: str) -> Optional[int]:
    # Best-effort count of objects in a class (returns None if the aggregate query fails).
    try:
        resp = client.query.aggregate(class_name).with_meta_count().do()
        # resp["data"]["Aggregate"][class_name][0]["meta"]["count"]
        return int(resp["data"]["Aggregate"][class_name][0]["meta"]["count"])
    except Exception:
        return None


def print_debug_info(client: weaviate.Client, class_name: str) -> None:
    print("\n=== Weaviate debug ===")
    sch = client.schema.get(class_name)
    print("\nclass:", sch.get("class"))
    props = sch.get("properties", []) or []
    print("\nproperties:", [p.get("name") for p in props])

    # show filterable flags
    filt_flags = {p.get("name"): bool(p.get("indexFilterable", False)) for p in props}
    print("\nindexFilterable flags (per property):")
    for k in sorted(filt_flags.keys()):
        print(f"  - {k}: {filt_flags[k]}")

    print("\nvectorIndexType:", sch.get("vectorIndexType"))
    print("\nvectorIndexConfig:", sch.get("vectorIndexConfig"))

    cnt = get_object_count(client, class_name)
    print("\nobject_count:", cnt)


def run_filtered_queries(
    client: weaviate.Client,
    class_name: str,
    query_vectors: List[List[float]],
    meta_conditions_list: List[Dict[str, Any]],
    topk: int,
    ef: Optional[int],
    #timeout_per_query: Optional[float],
    #print_where_samples: int = 3,
) -> Tuple[QueryRunResult, List[Dict[str, Any]]]:
    latencies_ms: List[float] = []
    returned_topk_ids: List[List[int]] = []
    #where_samples: List[Dict[str, Any]] = []

    # Map property -> datatype so we can choose valueInt/valueNumber/valueText in the where filter.
    prop_type_map, _ = get_schema_maps(client, class_name)

    # CPU sampling
    proc = psutil.Process(os.getpid()) if psutil else None
    cpu_proc_samples: List[float] = []
    cpu_sys_samples: List[float] = []
    if psutil:
        _ = proc.cpu_percent(interval=None)  # type: ignore
        _ = psutil.cpu_percent(interval=None)  # type: ignore

    t0 = time.perf_counter()

    for idx, (qv, mc) in enumerate(zip(query_vectors, meta_conditions_list)):
        where = build_where_from_meta_conditions(mc, prop_type_map)
        #if idx < max(0, print_where_samples):
            #where_samples.append(where)

        # Base nearVector payload;
        near: Dict[str, Any] = {"vector": qv}
        if ef is not None:
            near["ef"] = int(ef)

        q_start = time.perf_counter()
        try:
            q = (
                client.query.get(class_name, ["doc_id"])
                .with_where(where)
                .with_near_vector(near)
                .with_limit(topk)
            )
            #if timeout_per_query is not None:
                #q = q.with_timeout(timeout_per_query)
            resp = q.do()
        except Exception as e:#
            if ef is not None and ("ef" in str(e).lower() or "nearvector" in str(e).lower()):
                near2 = {"vector": qv}
                q = (
                    client.query.get(class_name, ["doc_id"])
                    .with_where(where)
                    .with_near_vector(near2)
                    .with_limit(topk)
                )
                #if timeout_per_query is not None:
                    #q = q.with_timeout(timeout_per_query)
                resp = q.do()
            else:
                raise

        q_end = time.perf_counter()

        items = (((resp or {}).get("data") or {}).get("Get") or {}).get(class_name) or []
        ids: List[int] = []
        for it in items:
            doc_id = it.get("doc_id")
            if doc_id is not None:
                ids.append(int(doc_id))

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
    return run #, where_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--class-name", required=True)

    p.add_argument("--topk", type=int, default=25)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--normalize", action="store_true")

    # Weaviate connection
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--timeout-connect", type=float, default=60.0)
    p.add_argument("--timeout-read", type=float, default=1000.0)

    # Query controls
    p.add_argument("--ef", type=int, default=None, help="Try to pass ef to nearVector (if supported)")
    #p.add_argument("--timeout-per-query", type=float, default=None, help="GraphQL per-query timeout (seconds)")
    #p.add_argument("--print-where-samples", type=int, default=3)
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()

    client = connect_weaviate(args.url, args.timeout_connect, args.timeout_read)

    if not client.schema.exists(args.class_name):
        raise RuntimeError(f"Class '{args.class_name}' not found in Weaviate.")

    if args.debug:
        print_debug_info(client, args.class_name)

    qvecs, meta_conditions_list, gt_ids = load_queries_with_filters(
        args.dataset, limit=args.n_queries, normalize=args.normalize
    )

    print(
        f"\n=== Running WEAVIATE FILTERED search: topk={args.topk}, n_queries={len(qvecs)}, "
        f"ef={args.ef}, parallel=1 ==="
    )

    run = run_filtered_queries(
        client=client,
        class_name=args.class_name,
        query_vectors=qvecs,
        meta_conditions_list=meta_conditions_list,
        topk=args.topk,
        ef=args.ef,
        #timeout_per_query=args.timeout_per_query,
        #print_where_samples=args.print_where_samples,
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
        "db": "weaviate",
        "mode": "filtered_vector_search",
        "dataset": args.dataset,
        "class": args.class_name,
        "topk": args.topk,
        "n_queries": len(qvecs),
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
        #"where_samples": where_samples[: min(10, len(where_samples))],
    }

    out = RESULTS_DIR / f"weaviate_filter_query_{args.class_name}_topk{args.topk}_q{len(qvecs)}_ef{args.ef}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()