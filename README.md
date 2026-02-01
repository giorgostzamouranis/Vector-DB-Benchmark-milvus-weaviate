# Vector Database Benchmark: Milvus vs Weaviate

A comprehensive performance benchmarking suite comparing two leading open-source vector databases: **Milvus** and **Weaviate**. This project evaluates ingestion throughput, query performance, recall accuracy, storage efficiency, and distributed scaling behavior.

**Academic Project**: Developed for the "Information Systems, Analysis and Design" course at ECE NTUA (Team 35)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Ingestion](#data-ingestion)
  - [Query Benchmarks](#query-benchmarks)
  - [Anomaly Detection](#anomaly-detection)
- [Experiments](#experiments)
- [Results Summary](#results-summary)
- [Contributing](#contributing)
- [Authors](#authors)
- [References](#references)

## Overview

As AI applications transition from prototypes to production systems, selecting the appropriate vector database becomes a critical infrastructure decision. This project provides an empirical evaluation of:

- **Milvus**: Cloud-native, disaggregated architecture with offline indexing
- **Weaviate**: Unified, monolithic architecture with online (real-time) indexing

Both systems use **HNSW (Hierarchical Navigable Small World)** as their primary indexing algorithm but differ fundamentally in their implementation strategies.

### What We Benchmark

| Aspect | Description |
|--------|-------------|
| **Ingestion Throughput** | Vectors/second at various dataset sizes |
| **Query Performance** | QPS, latency under different parallelism levels |
| **Recall vs Latency** | EF parameter tuning trade-offs |
| **Hybrid Search** | Vector similarity + metadata filtering |
| **Storage Efficiency** | Disk footprint comparison |
| **Distributed Mode** | Cluster deployment overhead analysis |
| **Anomaly Detection** | Real-world image-based use case |

## Key Findings

| Metric | Winner | Details |
|--------|--------|---------|
| **Large-scale Ingestion** | Milvus | 2.04× faster on 2.2M vectors (Arxiv dataset) |
| **Real-time Availability** | Weaviate | Data searchable immediately upon insertion |
| **High-dim Search (384d+)** | Milvus | SIMD-optimized C++ core excels |
| **Low-dim Search (100d)** | Tie | Comparable performance |
| **Storage Efficiency** | Weaviate | 2.1×–5.8× smaller disk footprint |
| **Filtered Search (indexed)** | Milvus | 97.4 QPS vs 69.2 QPS |
| **Filtered Search (unindexed)** | Weaviate | 61.1 QPS vs 54.7 QPS |

## Architecture

### Milvus: Disaggregated Cloud-Native

```
┌─────────────────────────────────────────────────────────┐
│                    Access Layer                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │  Proxy  │  │  Proxy  │  │  Proxy  │  (Stateless)    │
│  └────┬────┘  └────┬────┘  └────┬────┘                 │
├───────┼────────────┼────────────┼───────────────────────┤
│       │     Coordinator Service                         │
│  ┌────┴────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐  │
│  │RootCoord│  │DataCoord│  │QueryCord│  │IndexCoord │  │
│  └─────────┘  └─────────┘  └─────────┘  └───────────┘  │
├─────────────────────────────────────────────────────────┤
│                    Worker Nodes                          │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐            │
│  │DataNodes │  │QueryNodes │  │IndexNodes │            │
│  └──────────┘  └───────────┘  └───────────┘            │
├─────────────────────────────────────────────────────────┤
│  Message Storage (Kafka/Pulsar) │ Object Storage (MinIO)│
└─────────────────────────────────────────────────────────┘
```

### Weaviate: Unified Monolithic

```
┌─────────────────────────────────────────────────────────┐
│                   Weaviate Node                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              gRPC / REST Interface               │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │                 Query Engine                     │    │
│  │  ┌───────────┐  ┌──────────┐  ┌─────────────┐   │    │
│  │  │HNSW Index │  │Inverted  │  │ Roaring     │   │    │
│  │  │(Go impl.) │  │Index     │  │ Bitmaps     │   │    │
│  │  └───────────┘  └──────────┘  └─────────────┘   │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │           LSM-Tree Storage Engine               │    │
│  │  ┌─────────┐  ┌───────┐  ┌──────────────────┐   │    │
│  │  │MemTable │  │  WAL  │  │    SSTables      │   │    │
│  │  └─────────┘  └───────┘  └──────────────────┘   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Datasets

| Dataset | Dimensions | Vectors | Metadata | Use Case |
|---------|------------|---------|----------|----------|
| **GloVe-100** | 100 | 1,183,514 | None | Word embeddings |
| **Arxiv-Titles** | 384 | 2,200,000 | None | Document embeddings |
| **H&M-2048** | 2,048 | 105,542 | 24 fields | E-commerce (filtered search) |
| **MVTec AD** | 384 | ~2,500 | Category | Industrial anomaly detection |

All datasets use `float32` vectors with **Cosine Similarity** as the distance metric.

## Project Structure

This project is built on top of the [qdrant/vector-db-benchmark](https://github.com/qdrant/vector-db-benchmark) repository. We cloned the original benchmark suite and added our custom implementations inside it.

```
vector_benchmark/                    # Root project folder
├── milvus/                          # Milvus Docker deployment
│   └── volumes/                     # Persistent storage (etcd, minio, data)
├── weaviate/                        # Weaviate Docker deployment
│   └── volumes/                     # Persistent storage
└── scripts/
    └── vector-db-benchmark/         # 🔗 Cloned from qdrant/vector-db-benchmark
        ├── benchmark/               # Original benchmark framework
        ├── dataset_reader/          # Original dataset parsing utilities
        ├── datasets/                # Downloaded datasets (GloVe, Arxiv, H&M)
        ├── engine/                  # Original DB client implementations
        │   ├── clients/
        │   │   ├── milvus/
        │   │   ├── weaviate/
        │   │   └── ...
        │   └── servers/             # Docker configs for each DB
        │       ├── milvus-single-node/
        │       ├── weaviate-single-node/
        │       └── ...
        │
        └── custom/                  # ⭐ OUR IMPLEMENTATIONS
            ├── anomaly_detection/   # MVTec AD anomaly detection experiment
            │   ├── anomaly/         # Anomalous image embeddings (JSON)
            │   ├── good/            # Normal image embeddings (JSON)
            │   ├── ingest_anom_det_milvus.py    # Ingest normal vectors
            │   ├── ingest_anom_det_weaviate.py
            │   ├── milvus_anom_queries.py       # Query with anomalous vectors
            │   └── weaviate_anom_queries.py
            ├── results/             # Benchmark results output
            │   ├── ingest/          # Ingestion metrics (JSON)
            │   └── queries/         # Query metrics (JSON)
            │
            │── load_dataset.py              # Dataset loading utilities
            │
            │── # Ingestion Scripts
            ├── ingest_default_milvus.py     # Raw ingestion (no index)
            ├── ingest_indexing_milvus.py    # Ingestion with HNSW index
            ├── ingest_filters_milvus.py     # Ingestion with metadata fields
            ├── ingest_weaviate.py           # Weaviate standard ingestion
            ├── ingest_weaviate_distributed.py # Weaviate cluster mode
            ├── ingest_filters_weaviate.py   # Weaviate with metadata
            │
            │── # Query Benchmark Scripts
            ├── milvus_parallel_queries.py   # Parallel query benchmark
            ├── weaviate_parallel_queries.py
            ├── milvus_filter_queries.py     # Filtered (hybrid) search
            └── weaviate_filter_queries.py
```

> **Note**: Our custom scripts leverage the original repository's `benchmark.dataset`, `benchmark.config_read`, and `dataset_reader` modules for dataset handling.

## Installation

### Prerequisites

- Python 3.9+
- Docker Desktop (with WSL2 backend on Windows)
- Poetry (Python package manager)
- 8+ GB RAM recommended (we allocated 7.6 GB to Docker/WSL2)
- NVMe SSD for storage benchmarks

### 1. Clone the Repository

```bash
git clone https://github.com/giorgostzamouranis/vector-db-benchmark.git
cd vector-db-benchmark/scripts/vector-db-benchmark
```

### 2. Install Dependencies

```bash
# Install Poetry if not available
pip install poetry

# Install project dependencies
poetry install
```

### 3. Start Vector Databases

**Milvus (Standalone)**:
```bash
cd ../../milvus
docker compose up -d
```

**Weaviate**:
```bash
cd ../weaviate
docker compose up -d
```

### 4. Verify Installation

```bash
# Check Milvus
curl http://localhost:19530/healthz

# Check Weaviate
curl http://localhost:8080/v1/.well-known/ready
```

## Usage

### Data Ingestion

#### Milvus - Basic Ingestion (No Index)

```bash
poetry run python -m custom.ingest_default_milvus \
    --dataset glove-100-angular \
    --amount 10000
```

#### Milvus - With HNSW Index

```bash
poetry run python -m custom.ingest_indexing_milvus \
    --dataset glove-100-angular \
    --amount ALL \
    --batch-size 2000
```

#### Milvus - With Metadata Filters

```bash
poetry run python -m custom.ingest_filters_milvus \
    --dataset h-and-m-2048-angular-filters \
    --amount ALL \
    --scalar-index INVERTED
```

#### Weaviate - Standard Ingestion

```bash
poetry run python -m custom.ingest_weaviate \
    --dataset glove-100-angular \
    --amount ALL \
    --batch-size 2000 \
    --url http://localhost:8080
```

#### Weaviate - With Metadata Filters

```bash
poetry run python -m custom.ingest_filters_weaviate \
    --dataset h-and-m-2048-angular-filters \
    --amount ALL \
    --url http://127.0.0.1:8080 \
    --scalar-index
```

### Query Benchmarks

#### Milvus - Parallel Query Sweep

```bash
poetry run python -m custom.milvus_parallel_queries \
    --dataset glove-100-angular \
    --collection glove_100_angular_all \
    --topk 100 \
    --n-queries 10000 \
    --metric COSINE \
    --efs 128,256,512 \
    --parallels 1,8,16,32
```

#### Weaviate - Parallel Query Sweep

```bash
poetry run python -m custom.weaviate_parallel_queries \
    --dataset glove-100-angular \
    --class-name Glove100angularall \
    --topk 100 \
    --n-queries 10000 \
    --efs 128,256,512 \
    --parallels 1,8,16,32 \
    --url http://localhost:8080
```

#### Milvus - Filtered Search

```bash
poetry run python -m custom.milvus_filter_queries \
    --dataset h-and-m-2048-angular-filters \
    --collection h_and_m_2048_angular_filters_all_filters \
    --topk 25 \
    --n-queries 10000 \
    --ef 512 \
    --debug
```

#### Weaviate - Filtered Search

```bash
poetry run python -m custom.weaviate_filter_queries \
    --dataset h-and-m-2048-angular-filters \
    --class-name HAndM2048AngularFiltersAllFilters \
    --topk 25 \
    --n-queries 10000 \
    --url http://127.0.0.1:8080 \
    --ef 512 \
    --debug
```

### Anomaly Detection

This experiment demonstrates a real-world use case: **image-based industrial anomaly detection** using the MVTec AD dataset.

#### 1. Ingest Normal Vectors

**Milvus**:
```bash
python -m custom.ingest_anom_det_milvus \
    --data-dir custom/anomaly_detection/good \
    --collection custom_anomaly_good \
    --host 127.0.0.1 --port 19530 \
    --batch-size 2000
```

**Weaviate**:
```bash
python -m custom.ingest_anom_det_weaviate \
    --data-dir custom/anomaly_detection/good \
    --class-name customAnomalyGood \
    --batch-size 2000 \
    --host localhost --port 8080 --grpc-port 50051
```

#### 2. Query with Anomalous Vectors

**Milvus**:
```bash
python -m custom.milvus_anom_queries \
    --collection custom_anomaly_good \
    --anomaly-dir custom/anomaly_detection/anomaly \
    --topk 10 \
    --threshold 0.85 \
    --ef 256 \
    --parallel 1 \
    --use-filter
```

**Weaviate**:
```bash
python -m custom.weaviate_anom_queries \
    --class-name CustomAnomalyGood \
    --anomaly-dir custom/anomaly_detection/anomaly \
    --topk 10 \
    --threshold 0.85 \
    --ef 256 \
    --parallel 1 \
    --use-filter
```

#### Reset Database

**Milvus**:
```bash
cd milvus
docker compose down
rm -rf volumes/*
docker compose up -d
```

**Weaviate**:
```bash
cd weaviate
docker compose down
rm -rf volumes/*
docker compose up -d
```

## Experiments

### Experiment 1: Ingestion Performance

Evaluates write throughput at multiple dataset sizes (10K, 100K, Full).

**Key Metrics:**
- Vectors per second
- MB/s normalized throughput
- Index build time vs data load time

### Experiment 2: Parallel Query Scalability

Tests QPS under increasing client concurrency (1, 8, 16, 32 parallel processes).

**Key Metrics:**
- Queries per second (QPS)
- CPU utilization
- Saturation point identification

### Experiment 3: EF Parameter Tuning

Analyzes the recall-latency trade-off by varying the HNSW search depth parameter.

**Key Metrics:**
- Recall@K
- MRR@K
- Average latency (ms)

### Experiment 4: Hybrid Search (Filtered)

Compares vector search combined with metadata filtering.

**Key Metrics:**
- QPS with/without scalar indexes
- Filter evaluation overhead

### Experiment 5: Distributed Mode

Evaluates cluster deployment overhead on a single-node environment.

**Key Metrics:**
- Standalone vs Distributed QPS
- Coordination overhead

## Results Summary

### Ingestion Throughput (Arxiv-384, 2.2M vectors)

| Database | Throughput | Index Build Time |
|----------|------------|------------------|
| Milvus | 875 vec/sec | 86.1% of total time |
| Weaviate | 428 vec/sec | Included (online) |

### Query Performance (GloVe-100, EF=256, P=8)

| Database | QPS | Recall@100 |
|----------|-----|------------|
| Milvus | 425.2 | 0.9066 |
| Weaviate | 436.6 | 0.8951 |

### Storage Footprint (Full datasets)

| Dataset | Milvus | Weaviate | Ratio |
|---------|--------|----------|-------|
| GloVe-100 | 2,500 MB | 980 MB | 2.6× |
| Arxiv-384 | 7,100 MB | 1,800 MB | 3.9× |
| H&M-2048 | 3,200 MB | 550 MB | 5.8× |

## Hardware & Environment

Our experiments were conducted on:

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 7 4800H (8 cores, 16 threads @ 2.90 GHz) |
| RAM | 16 GB (7.6 GB allocated to Docker) |
| Storage | 512 GB NVMe SSD |
| OS | Windows 11 Home |
| Virtualization | **Docker Desktop v28.5.1** with WSL2 backend |
| WSL2 Kernel | Linux 6.6.87.2-microsoft-standard |


## Authors

**Team 35 - ECE NTUA**

- Andreas Fotakis (AM: 03121100) https://github.com/andreasfott
- Nikolaos Katsaidonis (AM: 03121868) https://github.com/NikosK10
- Georgios Tzamouranis (AM: 03121141) https://github.com/giorgostzamouranis

## References

1. [Vector-DB-Benchmark (qdrant)](https://github.com/qdrant/vector-db-benchmark) - Base framework
2. [Milvus Documentation](https://milvus.io/docs)
3. [Weaviate Documentation](https://weaviate.io/developers/weaviate)
4. [HNSW Paper](https://arxiv.org/abs/1603.09320)
5. [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)


---
