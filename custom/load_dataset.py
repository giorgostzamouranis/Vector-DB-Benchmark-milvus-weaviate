from __future__ import annotations

from typing import List, Tuple, Dict, Any

from benchmark.config_read import read_dataset_config
from benchmark.dataset import Dataset

def iter_records(dataset_name: str, normalize: bool = False):
    """
    Return an iterator of Record objects using the benchmark reader API.

    Each yielded item is a reader-specific Record that should expose at least:
      - .id
      - .vector
      - (optionally) .metadata
    """
    datasets = read_dataset_config()
    if dataset_name not in datasets:
        raise KeyError(f"Dataset '{dataset_name}' not found in datasets config")

    cfg = datasets[dataset_name]
    ds = Dataset(cfg)
    ds.download()

    reader = ds.get_reader(normalize=normalize)
    return reader.read_data()  # Iterator[Record]

def load_records_subset(
    dataset_name: str, limit: int | None, normalize: bool = False
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Load first `limit` records from reader.read_data().
    Returns (records_as_dicts, dim)

    Each record dict contains: {"id": int, "vector": List[float]}
    This helper converts reader Records into plain dicts for ingestion scripts.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0 or None for ALL")

    records_iter = iter_records(dataset_name, normalize=normalize)

    out: List[Dict[str, Any]] = []
    dim: int | None = None

    for rec in records_iter:
        # rec is dataset_reader.base_reader.Record
        vec = rec.vector
        if dim is None:
            dim = len(vec)

        out.append({"id": rec.id, "vector": vec})
        if limit is not None and len(out) >= limit:
            break

    if not out or dim is None:
        raise RuntimeError("No records were loaded from reader.read_data()")

    return out, dim

if __name__ == "__main__":
    recs, d = load_records_subset("glove-100-angular", 1000)
    print(f"Loaded {len(recs)} records, dim={d}, first_id={recs[0]['id']}")
