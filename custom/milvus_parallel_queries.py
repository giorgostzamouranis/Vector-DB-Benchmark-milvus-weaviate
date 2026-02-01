from __future__ import annotations
import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import Collection, connections, utility

from benchmark.dataset import Dataset
from benchmark.config_read import read_dataset_config
import multiprocessing as mp

_WORKER = {}

def _worker_init(host: str, port: str, collection: str):
    # Pool worker initializer: connect to Milvus, load the collection once, and cache it per-process.
    connections.connect("default", host=host, port=port)
    col = Collection(collection)
    col.load()
    _WORKER["col"] = col

def _worker_search_one(args):
    # Worker task runner for one query:
    # args is a tuple: (query_vector, topk, metric_type, ef)
    # Returns: (topk_ids, latency_ms)
    qv, topk, metric_type, ef = args
    col: Collection = _WORKER["col"]

    # Build Milvus search params.
    search_params: Dict[str, Any] = {"metric_type": metric_type, "params": {}}
    if ef is not None:
        search_params["params"]["ef"] = ef

    t0 = time.perf_counter()
    res = col.search(
        data=[qv],
        anns_field="vector",
        param=search_params,
        limit=topk,
        output_fields=["id"],
    )
    t1 = time.perf_counter()

    ids = [int(h.id) for h in res[0]]
    latency_ms = (t1 - t0) * 1000.0
    return ids, latency_ms

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

def load_queries(dataset_name: str, limit: Optional[int], normalize: bool) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Returns:
      - queries_vectors: list of float vectors
      - gt_ids: list of ground-truth neighbor id lists (may be empty lists if unavailable)
    """
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

        # Ground truth
        exp = getattr(q, "expected_result", None)
        if exp is None:
            gt_ids.append([])
        else:
            gt_ids.append(list(exp))

    if not queries_vectors:
        raise RuntimeError("No queries loaded.")

    return queries_vectors, gt_ids

def connect_milvus(host: str, port: str) -> None:
    connections.connect("default", host=host, port=port)

def get_collection(name: str) -> Collection:
    if not utility.has_collection(name):
        raise RuntimeError(f"Collection '{name}' not found in Milvus.")
    col = Collection(name)
    col.load()
    return col

def recall_at_k(found: List[int], gt: List[int], k: int) -> Optional[float]:
    # Recall against ground truth;
    if not gt:
        return None
    gt_k = gt[:k]
    if not gt_k:
        return None
    return len(set(found[:k]) & set(gt_k)) / float(k)

def mrr_at_k(found: List[int], gt: List[int], k: int) -> Optional[float]:
    # MRR (mean reciprocal rank at k) for a single query;
    if not gt:
        return None
    gt_set = set(gt[:k])
    for rank, doc_id in enumerate(found[:k], start=1):
        if doc_id in gt_set:
            return 1.0 / rank
    return 0.0

def run_knn_queries(
    col: Collection,
    query_vectors: List[List[float]],
    topk: int,
    metric_type: str,
    ef: Optional[int],
    vector_field: str = "vector",
) -> QueryRunResult:
    latencies_ms: List[float] = []
    returned_topk_ids: List[List[int]] = []

    # CPU sampling setup
    proc = psutil.Process(os.getpid()) if psutil else None
    cpu_proc_samples: List[float] = []
    cpu_sys_samples: List[float] = []

    if psutil:
        # Prime the counters so the first sample isn't misleading.
        _ = proc.cpu_percent(interval=None)  # type: ignore
        _ = psutil.cpu_percent(interval=None)  # type: ignore
        
    # Build Milvus search params.
    search_params: Dict[str, Any] = {"metric_type": metric_type, "params": {}}
    if ef is not None:
        search_params["params"]["ef"] = ef

    t0 = time.perf_counter()

    for qv in query_vectors:
        q_start = time.perf_counter()

        res = col.search(
            data=[qv],
            anns_field=vector_field,
            param=search_params,
            limit=topk,
            output_fields=["id"],
        )

        hits = res[0]
        ids = [int(h.id) for h in hits]
        returned_topk_ids.append(ids)

        q_end = time.perf_counter()
        latencies_ms.append((q_end - q_start) * 1000.0) #ms

        if psutil:
            cpu_proc_samples.append(proc.cpu_percent(interval=None))  # type: ignore
            cpu_sys_samples.append(psutil.cpu_percent(interval=None))  # type: ignore

    total_time_sec = time.perf_counter() - t0
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

def run_knn_queries_parallel(
    host: str,
    port: str,
    collection: str,
    query_vectors: List[List[float]],
    topk: int,
    metric_type: str,
    ef: Optional[int],
    parallel: int,
) -> QueryRunResult:
    # Prepare one task per query vector.
    tasks = [(qv, topk, metric_type, ef) for qv in query_vectors]

    t0 = time.perf_counter()

    if parallel <= 1:
        raise ValueError("run_knn_queries_parallel requires parallel > 1")

    else:
        ctx = mp.get_context("spawn")  # spawn for broad compatibility (matches repo fallback behavior)
        with ctx.Pool(
            processes=parallel,
            initializer=_worker_init,
            initargs=(host, port, collection),
        ) as pool:
            # unordered iteration to reduce head-of-line blocking across tasks.
            results = list(pool.imap(_worker_search_one, tasks))

        returned_topk_ids = [r[0] for r in results]
        latencies_ms = [r[1] for r in results]

    total_time_sec = time.perf_counter() - t0
    qps = (len(query_vectors) / total_time_sec) if total_time_sec > 0 else 0.0

    return QueryRunResult(
        latencies_ms=latencies_ms,
        total_time_sec=total_time_sec,
        qps=qps,
        cpu_process_avg_pct=None,
        cpu_system_avg_pct=None,
        returned_topk_ids=returned_topk_ids,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--n-queries", type=int, default=1000, help="How many queries to run (from dataset queries)")
    p.add_argument("--normalize", action="store_true", help="Use normalize=True when reading dataset vectors")
    p.add_argument("--metric", default="COSINE", help="COSINE | L2 | IP (depends on your index setup)")
    p.add_argument("--ef", type=int, default=None, help="HNSW search param 'ef' (optional)")
    p.add_argument("--efs", default="", help="Comma-separated efs, e.g. 128,512 (overrides --ef)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="19530")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--parallels", default="", help="Comma-separated parallels, e.g. 1,8,100 (overrides --parallel)")

    args = p.parse_args()

    # Decide which efs to run
    if args.efs.strip():
        efs = [int(x.strip()) for x in args.efs.split(",") if x.strip()]
    else:
        efs = [args.ef]  # may be None

    if args.parallels.strip():
        parallels = [int(x.strip()) for x in args.parallels.split(",") if x.strip()]
    else:
        parallels = [args.parallel]

    # Keep only the allowed parallel values (e.g., to match repo presets)
    parallels = [p for p in parallels if p in (1, 8, 10, 50, 16,32)]
    if not parallels:
        raise ValueError("No valid parallel values left. Use --parallels 1,10,100")

    efs = [e for e in efs if e in (16,64,10,32,128, 256, 380, 512)]
    if not efs:
        raise ValueError("No valid ef values left. Use --efs 128,512")

    connect_milvus(args.host, args.port)
    col = get_collection(args.collection)

    query_vectors, gt_ids = load_queries(args.dataset, limit=args.n_queries, normalize=args.normalize)

    for ef_value in efs:
        for par in parallels:
            print(f"\n=== Running: ef={ef_value}, parallel={par}, topk={args.topk}, n_queries={len(query_vectors)}, metric={args.metric} ===")

            if par == 1:
                run = run_knn_queries(
                    col=col,
                    query_vectors=query_vectors,
                    topk=args.topk,
                    metric_type=args.metric,
                    ef=ef_value,
                )
            else:
                run = run_knn_queries_parallel(
                    host=args.host,
                    port=args.port,
                    collection=args.collection,
                    query_vectors=query_vectors,
                    topk=args.topk,
                    metric_type=args.metric,
                    ef=ef_value,
                    parallel=par,
                )

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
                "dataset": args.dataset,
                "collection": args.collection,
                "topk": args.topk,
                "n_queries": len(query_vectors),
                "metric": args.metric,
                "ef": ef_value,
                "parallel": par,  
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

            out = RESULTS_DIR / f"milvus_query_{args.collection}_topk{args.topk}_q{len(query_vectors)}_ef{ef_value}_par{par}.json"
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")

            print(json.dumps(result, indent=2))
            print(f"Saved: {out}")

if __name__ == "__main__":
    main()