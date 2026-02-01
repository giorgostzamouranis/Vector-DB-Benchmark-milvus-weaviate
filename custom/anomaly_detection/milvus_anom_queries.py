from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import Collection, connections, utility

RESULTS_DIR = Path("custom/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Optional CPU metrics
try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def load_anomaly_queries(folder: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads all *.json in folder. Each file is a list of {"vector": [...], "filter": "..."}.
    Returns a flat list of records.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(folder.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in: {folder}")

    out: List[Dict[str, Any]] = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{fp} is not a JSON list.")
        for r in data:
            if not isinstance(r, dict) or "vector" not in r:
                raise ValueError(f"Bad record in {fp}: {r}")
            out.append(r)
            if limit is not None and len(out) >= limit:
                return out

    return out


def cosine_sim(a: List[float], b: List[float]) -> float:
    # cosine(a,b) = dot(a,b) / (||a||*||b||)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = int(len(vals) * p)
    if idx >= len(vals):
        idx = len(vals) - 1
    return vals[idx]


@dataclass
class RunResult:
    latencies_ms: List[float]
    total_time_sec: float
    qps: float
    cpu_process_avg_pct: Optional[float]
    cpu_system_avg_pct: Optional[float]
    kept_counts: List[int]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anomaly-dir", default="custom/anomaly_detection/anomaly")
    p.add_argument("--collection", required=True, help="Milvus collection that holds GOOD vectors")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.8, help="Keep neighbors with cosine_sim >= threshold")
    p.add_argument("--n-queries", type=int, default=None, help="Limit queries (across all anomaly jsons)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default="19530")

    p.add_argument("--ef", type=int, default=256)
    p.add_argument("--parallel", type=int, default=1)

    # optional filter behavior
    p.add_argument("--use-filter", action="store_true", help="Apply expr: filter == record.filter")
    p.add_argument("--filter-field", default="filter", help="Scalar field name in Milvus (VARCHAR)")
    args = p.parse_args()

    if args.parallel != 1:
        raise ValueError("This script is fixed to parallel=1 as requested.")

    # Load queries
    anomaly_folder = Path(args.anomaly_dir)
    query_recs = load_anomaly_queries(anomaly_folder, limit=args.n_queries)
    if not query_recs:
        raise RuntimeError("No anomaly queries loaded.")
    query_vectors = [list(r["vector"]) for r in query_recs]

    # Connect + load collection
    connections.connect("default", host=args.host, port=args.port)
    if not utility.has_collection(args.collection):
        raise RuntimeError(f"Collection '{args.collection}' not found in Milvus.")
    col = Collection(args.collection)
    col.load()

    # CPU sampling
    proc = psutil.Process(os.getpid()) if psutil else None
    cpu_proc_samples: List[float] = []
    cpu_sys_samples: List[float] = []
    if psutil:
        _ = proc.cpu_percent(interval=None)  # type: ignore
        _ = psutil.cpu_percent(interval=None)  # type: ignore

    # Search params
    search_params: Dict[str, Any] = {"metric_type": "COSINE", "params": {"ef": int(args.ef)}}

    latencies_ms: List[float] = []
    kept_counts: List[int] = []

    t0_total = time.perf_counter()

    for qv, rec in zip(query_vectors, query_recs):
        expr = None
        if args.use_filter:
            fval = str(rec.get("filter", ""))
            expr = f'{args.filter_field} == "{fval}"'

        q0 = time.perf_counter()
        res = col.search(
            data=[qv],
            anns_field="vector",
            param=search_params,
            limit=int(args.topk),
            expr=expr,
            output_fields=["vector"],  
        )
        q1 = time.perf_counter()

        hits = res[0]
        kept = 0
        for h in hits:
            # output_fields are in h.entity
            neighbor_vec = h.entity.get("vector")
            if neighbor_vec is None:
                continue
            sim = cosine_sim(qv, neighbor_vec)
            if sim >= args.threshold:
                kept += 1

        kept_counts.append(kept)
        latencies_ms.append((q1 - q0) * 1000.0)

        if psutil:
            cpu_proc_samples.append(proc.cpu_percent(interval=None))  # type: ignore
            cpu_sys_samples.append(psutil.cpu_percent(interval=None))  # type: ignore

    total_time_sec = time.perf_counter() - t0_total
    qps = (len(query_vectors) / total_time_sec) if total_time_sec > 0 else 0.0

    cpu_process_avg = (sum(cpu_proc_samples) / len(cpu_proc_samples)) if cpu_proc_samples else None
    cpu_system_avg = (sum(cpu_sys_samples) / len(cpu_sys_samples)) if cpu_sys_samples else None

    kept_avg = (sum(kept_counts) / len(kept_counts)) if kept_counts else None
    kept_rate = (sum(kept_counts) / (len(kept_counts) * args.topk)) if kept_counts else None

    result: Dict[str, Any] = {
        "db": "milvus",
        "collection": args.collection,
        "anomaly_dir": str(anomaly_folder),
        "topk": args.topk,
        "threshold": args.threshold,
        "n_queries": len(query_vectors),
        "metric": "COSINE",
        "ef": int(args.ef),
        "parallel": 1,
        "use_filter": bool(args.use_filter),
        "total_time_sec": total_time_sec,
        "throughput_qps": qps,
        "latency_ms": {
            "avg": (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else None,
            "min": min(latencies_ms) if latencies_ms else None,
            "max": max(latencies_ms) if latencies_ms else None,
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
        },
        "cpu": {
            "psutil_available": psutil is not None,
            "process_cpu_avg_pct": cpu_process_avg,
            "system_cpu_avg_pct": cpu_system_avg,
        },
        "threshold_filtering": {
            "kept_avg_per_query": kept_avg,
            "kept_rate_of_topk": kept_rate,
            "kept_min": min(kept_counts) if kept_counts else None,
            "kept_max": max(kept_counts) if kept_counts else None,
        },
    }

    out = RESULTS_DIR / f"milvus_anomaly_{args.collection}_topk{args.topk}_ef{args.ef}_thr{args.threshold}_q{len(query_vectors)}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()