from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import weaviate
from weaviate.classes.config import Reconfigure

RESULTS_DIR = Path("custom/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def load_anomaly_queries(folder: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
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


def connect(host: str, http_port: int, grpc_port: int) -> weaviate.WeaviateClient:
    return weaviate.connect_to_custom(
        http_host=host,
        http_port=http_port,
        http_secure=False,
        grpc_host=host,
        grpc_port=grpc_port,
        grpc_secure=False,
    )


def update_collection_ef(client: weaviate.WeaviateClient, class_name: str, ef: int) -> None:
    col = client.collections.get(class_name)
    col.config.update(
        vector_index_config=Reconfigure.VectorIndex.hnsw(ef=int(ef))
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anomaly-dir", default="custom/anomaly_detection/anomaly")
    p.add_argument("--class-name", required=True, help="Weaviate class that holds GOOD vectors")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--n-queries", type=int, default=None)

    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--grpc-port", type=int, default=50051)

    p.add_argument("--ef", type=int, default=256)
    p.add_argument("--parallel", type=int, default=1)

    p.add_argument("--use-filter", action="store_true", help="Filter by property 'filter' == record.filter")
    p.add_argument("--filter-prop", default="filter", help="Property name used for filtering")
    args = p.parse_args()

    if args.parallel != 1:
        raise ValueError("This script is fixed to parallel=1 as requested.")

    anomaly_folder = Path(args.anomaly_dir)
    query_recs = load_anomaly_queries(anomaly_folder, limit=args.n_queries)
    if not query_recs:
        raise RuntimeError("No anomaly queries loaded.")

    client = connect(args.host, args.port, args.grpc_port)
    try:
        if not client.is_ready():
            raise RuntimeError(f"Weaviate not ready at http://{args.host}:{args.port}")

        update_collection_ef(client, args.class_name, int(args.ef))
        # tiny warmup delay 
        time.sleep(1)

        col = client.collections.get(args.class_name)

        # CPU sampling
        proc = psutil.Process(os.getpid()) if psutil else None
        cpu_proc_samples: List[float] = []
        cpu_sys_samples: List[float] = []
        if psutil:
            _ = proc.cpu_percent(interval=None)  # type: ignore
            _ = psutil.cpu_percent(interval=None)  # type: ignore

        latencies_ms: List[float] = []
        kept_counts: List[int] = []

        t0_total = time.perf_counter()

        for rec in query_recs:
            qv = list(rec["vector"])

            where_filter = None
            if args.use_filter:
                fval = str(rec.get("filter", ""))
                # Weaviate v4 filter:
                where_filter = {
                    "path": [args.filter_prop],
                    "operator": "Equal",
                    "valueText": fval,
                }

            q0 = time.perf_counter()

            resp = col.query.near_vector(
                near_vector=qv,
                limit=int(args.topk),
                filters=where_filter,
                # return vectors so we can compute cosine ourselves
                return_metadata=None,
                include_vector=True,
                return_properties=[args.filter_prop],  
            )

            q1 = time.perf_counter()

            kept = 0
            for obj in resp.objects:
                nv = obj.vector  # returned because include_vector=True
                if nv is None:
                    continue
                sim = cosine_sim(qv, list(nv))
                if sim >= args.threshold:
                    kept += 1

            kept_counts.append(kept)
            latencies_ms.append((q1 - q0) * 1000.0)

            if psutil:
                cpu_proc_samples.append(proc.cpu_percent(interval=None))  # type: ignore
                cpu_sys_samples.append(psutil.cpu_percent(interval=None))  # type: ignore

        total_time_sec = time.perf_counter() - t0_total
        qps = (len(query_recs) / total_time_sec) if total_time_sec > 0 else 0.0

        cpu_process_avg = (sum(cpu_proc_samples) / len(cpu_proc_samples)) if cpu_proc_samples else None
        cpu_system_avg = (sum(cpu_sys_samples) / len(cpu_sys_samples)) if cpu_sys_samples else None

        kept_avg = (sum(kept_counts) / len(kept_counts)) if kept_counts else None
        kept_rate = (sum(kept_counts) / (len(kept_counts) * args.topk)) if kept_counts else None

        result: Dict[str, Any] = {
            "db": "weaviate_v4_grpc",
            "class": args.class_name,
            "anomaly_dir": str(anomaly_folder),
            "topk": args.topk,
            "threshold": args.threshold,
            "n_queries": len(query_recs),
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

        out = RESULTS_DIR / f"weaviate_anomaly_{args.class_name}_topk{args.topk}_ef{args.ef}_thr{args.threshold}_q{len(query_recs)}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"Saved: {out}")

    finally:
        client.close()


if __name__ == "__main__":
    main()