from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import multiprocessing as mp
import weaviate

from weaviate.classes.config import Reconfigure

from benchmark.dataset import Dataset
from benchmark.config_read import read_dataset_config

RESULTS_DIR = Path("custom/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# For cpu metrics
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# Per-process cache for multiprocessing workers.
_WORKER: Dict[str, Any] = {}

def _worker_init(http_host: str, http_port: int, grpc_port: int):
    """
    Pool worker initializer:
    create a Weaviate client per process and store it in a global cache.
    """
    try:
        client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            http_secure=False,
            grpc_host=http_host,
            grpc_port=grpc_port,
            grpc_secure=False,
        )
        _WORKER["client"] = client
    except Exception as e:
        #Fail fast so connection problems are visible instead of silently producing empty results.
        print(f"CRITICAL: Worker failed to connect to Weaviate: {e}")
        raise

def _worker_search_one(args):
    # Worker task runner for one query.
    # args: (class_name, query_vector, topk)
    class_name, qv, topk = args
    client: weaviate.WeaviateClient = _WORKER["client"]
    
    # Collection retrieval is lightweight in v4; the client caches metadata internally.
    collection = client.collections.get(class_name)

    t0 = time.perf_counter()
    
    # gRPC vector search. We request "doc_id" explicitly for evaluation.
    response = collection.query.near_vector(
        near_vector=qv,
        limit=int(topk),
        return_properties=["doc_id"], 
    )
    
    t1 = time.perf_counter()

    ids = []
    for obj in response.objects:
        # doc_id is stored as a property; convert to int for metric computation.
        val = obj.properties.get("doc_id")
        if val is not None:
            ids.append(int(val))

    latency_ms = (t1 - t0) * 1000.0
    return ids, latency_ms

@dataclass
class QueryRunResult:
    latencies_ms: List[float]
    total_time_sec: float
    qps: float
    cpu_process_avg_pct: Optional[float]
    cpu_system_avg_pct: Optional[float]
    returned_topk_ids: List[List[Any]]

def load_queries(dataset_name: str, limit: Optional[int], normalize: bool) -> Tuple[List[List[float]], List[List[int]]]:
    cfgs = read_dataset_config()
    if dataset_name not in cfgs:
        raise KeyError(f"Dataset '{dataset_name}' not found in benchmark config")

    ds = Dataset(cfgs[dataset_name])
    ds.download()
    reader = ds.get_reader(normalize=normalize)

    if not hasattr(reader, "read_queries"):
        raise RuntimeError("Dataset reader has no read_queries().")

    queries_vectors: List[List[float]] = []
    gt_ids: List[List[int]] = []

    for i, q in enumerate(reader.read_queries()):
        if limit is not None and i >= limit:
            break
        qvec = getattr(q, "vector", None)
        if qvec is None:
            raise RuntimeError("Query object has no .vector field")
        queries_vectors.append(list(qvec))
        exp = getattr(q, "expected_result", None)
        if exp is None:
            gt_ids.append([])
        else:
            gt_ids.append(list(exp))

    if not queries_vectors:
        raise RuntimeError("No queries loaded.")

    return queries_vectors, gt_ids

def update_collection_ef(host: str, http_port: int, grpc_port: int, class_name: str, ef: int):
    """
    Update the HNSW 'ef' search parameter on the server.
    """
    print(f"Connecting to update config -> Host: {host}, HTTP: {http_port}, gRPC: {grpc_port}")
    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=http_port,
        http_secure=False,
        grpc_host=host,
        grpc_port=grpc_port,
        grpc_secure=False,
    )
    
    try:
        collection = client.collections.get(class_name)
        # Dynamically update vector index configuration via Reconfigure.
        collection.config.update(
            vector_index_config=Reconfigure.VectorIndex.hnsw(
                ef=ef
            )
        )
        print(f" -> SUCCESS: Updated HNSW ef to {ef} for class '{class_name}'")
    except Exception as e:
        print(f" -> ERROR: Failed to update ef: {e}")
        raise
    finally:
        client.close()

def run_knn_queries_weaviate(
    http_host: str,
    http_port: int,
    grpc_port: int,
    class_name: str,
    query_vectors: List[List[float]],
    topk: int,
) -> QueryRunResult:
    client = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=False,
        grpc_host=http_host,
        grpc_port=grpc_port,
        grpc_secure=False,
    )
    
    try:
        latencies_ms: List[float] = []
        returned_topk_ids: List[List[Any]] = []

        # CPU sampling
        proc = psutil.Process(os.getpid()) if psutil else None
        cpu_proc_samples: List[float] = []
        cpu_sys_samples: List[float] = []
        if psutil:
            _ = proc.cpu_percent(interval=None)
            _ = psutil.cpu_percent(interval=None)

        collection = client.collections.get(class_name)
        t0_total = time.perf_counter()

        for qv in query_vectors:
            q0 = time.perf_counter()
            
            resp = collection.query.near_vector(
                near_vector=qv,
                limit=int(topk),
                return_properties=["doc_id"]
            )
            
            q1 = time.perf_counter()

            ids = []
            for obj in resp.objects:
                val = obj.properties.get("doc_id")
                if val is not None:
                    ids.append(int(val))
            
            returned_topk_ids.append(ids)
            latencies_ms.append((q1 - q0) * 1000.0)

            if psutil:
                cpu_proc_samples.append(proc.cpu_percent(interval=None))
                cpu_sys_samples.append(psutil.cpu_percent(interval=None))

        total_time_sec = time.perf_counter() - t0_total
        qps = (len(query_vectors) / total_time_sec) if total_time_sec > 0 else 0.0

        cpu_process_avg = (sum(cpu_proc_samples) / len(cpu_proc_samples)) if cpu_proc_samples else None
        cpu_system_avg = (sum(cpu_sys_samples) / len(cpu_sys_samples)) if cpu_sys_samples else None

        return QueryRunResult(
            latencies_ms=latencies_ms,
            total_time_sec=total_time_sec,
            qps=qps,
            cpu_process_avg_pct=cpu_process_avg,
            cpu_system_avg_pct=cpu_system_avg,
            returned_topk_ids=returned_topk_ids,
        )
    finally:
        client.close()

def run_knn_queries_weaviate_parallel(
    http_host: str,
    http_port: int,
    grpc_port: int,
    class_name: str,
    query_vectors: List[List[float]],
    topk: int,
    parallel: int,
) -> QueryRunResult:
    
    tasks = [(class_name, qv, topk) for qv in query_vectors]

    t0_total = time.perf_counter()

    if parallel <= 1:
        raise ValueError("Parallel needs to be > 1")

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=parallel,
        initializer=_worker_init,
        initargs=(http_host, http_port, grpc_port),
    ) as pool:
        # Use imap to stream results if query sets are large (still materialized here via list()).
        results = list(pool.imap(_worker_search_one, tasks))

    returned_topk_ids = [r[0] for r in results]
    latencies_ms = [r[1] for r in results]

    total_time_sec = time.perf_counter() - t0_total
    qps = (len(query_vectors) / total_time_sec) if total_time_sec > 0 else 0.0

    return QueryRunResult(
        latencies_ms=latencies_ms,
        total_time_sec=total_time_sec,
        qps=qps,
        cpu_process_avg_pct=None,
        cpu_system_avg_pct=None,
        returned_topk_ids=returned_topk_ids,
    )

def recall_at_k(found: List[int], gt: List[int], k: int) -> Optional[float]:
    if not gt: return None
    gt_k = gt[:k]
    if not gt_k: return None
    return len(set(found[:k]) & set(gt_k)) / float(k)

def mrr_at_k(found: List[int], gt: List[int], k: int) -> Optional[float]:
    if not gt: return None
    gt_set = set(gt[:k])
    for rank, doc_id in enumerate(found[:k], start=1):
        if doc_id in gt_set:
            return 1.0 / rank
    return 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--class-name", required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--normalize", action="store_true")
    
    # - ef: single value # - efs: comma-separated list for sweeps
    p.add_argument("--ef", type=int, default=None)
    p.add_argument("--efs", default="", help="Comma separated values e.g. 64,128,256")
    
    # Connection Params
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8080, help="HTTP Port")
    p.add_argument("--grpc-port", type=int, default=50051, help="gRPC Port (Check docker-compose!)")
    
    # Execution Params
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--parallels", default="")

    args = p.parse_args()

    # Build the ef sweep list. If nothing is specified, use -1 as "do not change server config".
    if args.efs.strip():
        efs = [int(x.strip()) for x in args.efs.split(",") if x.strip()]
    elif args.ef is not None:
        efs = [args.ef]
    else:
        efs = [-1]

    # Build the parallel sweep list.
    if args.parallels.strip():
        parallels = [int(x.strip()) for x in args.parallels.split(",") if x.strip()]
    else:
        parallels = [args.parallel]

    # restrict to a preset set of parallel values.
    parallels = [p for p in parallels if p in (1, 8, 10, 16, 32, 50)]
    if not parallels and args.parallel not in (1, 8, 10, 16, 32, 50):
         print("Warning: Parallel values filtered out or invalid. Using default logic.")
         parallels = [args.parallel]

    print(f"Loading queries for dataset: {args.dataset}...")
    query_vectors, gt_ids = load_queries(args.dataset, limit=args.n_queries, normalize=args.normalize)
    print(f"Loaded {len(query_vectors)} queries.")

    for ef_value in efs:
        # Update Configuration
        if ef_value > 0:
            print(f"\n--- Reconfiguring Server: Setting ef={ef_value} ---")
            update_collection_ef(
                host=args.host,
                http_port=args.port,
                grpc_port=args.grpc_port,
                class_name=args.class_name,
                ef=ef_value
            )
            # Small delay to allow the updated config to propagate to shards.
            time.sleep(2) 
        else:
            print("\n--- Skipping reconfiguration (using current server defaults) ---")

        # Run Benchmark
        for par in parallels:
            print(f"Running: ef={ef_value} (active), parallel={par}, topk={args.topk}")

            if par == 1:
                run = run_knn_queries_weaviate(
                    http_host=args.host,
                    http_port=args.port,
                    grpc_port=args.grpc_port,
                    class_name=args.class_name,
                    query_vectors=query_vectors,
                    topk=args.topk,
                )
            else:
                run = run_knn_queries_weaviate_parallel(
                    http_host=args.host,
                    http_port=args.port,
                    grpc_port=args.grpc_port,
                    class_name=args.class_name,
                    query_vectors=query_vectors,
                    topk=args.topk,
                    parallel=par,
                )

            lat = run.latencies_ms

            # Compute quality metrics against ground truth
            recalls: List[float] = []
            mrrs: List[float] = []
            hits_cnt = 0
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
                    hits_cnt += 1

            quality = {
                "ground_truth_available_queries": valid,
                "recall_at_k_avg": (sum(recalls) / len(recalls)) if recalls else None,
                "mrr_at_k_avg": (sum(mrrs) / len(mrrs)) if mrrs else None,
                "hit_rate_at_k": (hits_cnt / valid) if valid > 0 else None,
            }

            result: Dict[str, Any] = {
                "db": "weaviate_v4_grpc",
                "dataset": args.dataset,
                "class": args.class_name,
                "topk": args.topk,
                "n_queries": len(query_vectors),
                "ef": ef_value, 
                "parallel": par,
                "total_time_sec": run.total_time_sec,
                "throughput": run.qps,
                "latency_ms": {
                    "avg": (sum(lat) / len(lat)) if lat else None,
                    "min": min(lat) if lat else None,
                    "max": max(lat) if lat else None,
                    "p95": sorted(lat)[int(len(lat)*0.95)] if lat else None, 
                    "p99": sorted(lat)[int(len(lat)*0.99)] if lat else None
                },
                "cpu": {
                    "psutil_available": psutil is not None,
                    "process_cpu_avg_pct": run.cpu_process_avg_pct,
                    "system_cpu_avg_pct": run.cpu_system_avg_pct,
                },
                "quality_vs_ground_truth": quality,
            }

            out_filename = f"weaviate_v4_grpc_{args.class_name}_topk{args.topk}_ef{ef_value}_par{par}.json"
            out = RESULTS_DIR / out_filename
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            
            print(f" -> QPS: {run.qps:.2f}, Recall: {quality['recall_at_k_avg']:.4f}, Latency Avg: {result['latency_ms']['avg']:.2f}ms")
            print(f" -> Saved: {out}")

if __name__ == "__main__":
    main()