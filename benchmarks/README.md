# Retrieval Benchmarks

This module provides tools for evaluating retrieval pipeline performance using the **Zalo AI Legal Text Retrieval** dataset.

## Overview

The benchmark framework allows you to:
- Generate embeddings for the corpus using different embedding models
- Ingest embeddings into Qdrant for fast retrieval
- Evaluate retrieval performance with standard IR metrics
- Compare different pipeline configurations

## Quick Start

### 1. Generate Embeddings (on Kaggle GPU)

Use the provided Kaggle notebook to generate embeddings:

1. Upload `data/zalo_ai_retrieval/corpus.jsonl` to Kaggle as a dataset
2. Open `notebooks/kaggle_embed_corpus.ipynb` on Kaggle
3. Run all cells (takes ~30-60 minutes on T4 GPU)
4. Download `embeddings_gte.zip` from the Output tab
5. Extract to `benchmarks/data/embeddings/gte/`

```bash
# Extract embeddings
unzip embeddings_gte.zip -d benchmarks/data/embeddings/gte/
```

### 2. Ingest into Qdrant

Ensure Qdrant is running (e.g., via Docker), then ingest the embeddings:

```bash
# Start Qdrant (if not already running)
docker-compose up -d qdrant

# Ingest embeddings into Qdrant
python -m benchmarks.ingest \
    --embeddings-dir benchmarks/data/embeddings/gte \
    --collection bench_gte \
    --recreate
```

### 3. Run Evaluation

```bash
# Run evaluation with default settings
python -m benchmarks.run_eval --collection bench_gte

# Evaluate with specific k values
python -m benchmarks.run_eval --collection bench_gte --k-values 1,5,10,20

# Dense-only evaluation (no hybrid search)
python -m benchmarks.run_eval --collection bench_gte --no-hybrid

# Save per-query results for analysis
python -m benchmarks.run_eval --collection bench_gte --per-query
```

## Folder Structure

```
benchmarks/
├── __init__.py              # Module exports
├── config.py                # Configuration settings
├── dataset.py               # Zalo AI dataset loader
├── metrics.py               # IR metrics (P, R, F1, MRR, NDCG)
├── evaluator.py             # Evaluation orchestrator
├── retriever.py             # Qdrant retriever wrapper
├── results.py               # Results storage/comparison
├── ingest.py                # Corpus ingestion script
├── run_eval.py              # CLI entry point
├── embeddings/              # Embedding providers
│   ├── __init__.py
│   ├── base.py              # Abstract interface
│   └── gte.py               # GTE implementation
├── notebooks/               # Kaggle notebooks
│   └── kaggle_embed_corpus.ipynb
├── data/                    # Embedding files (gitignored)
│   └── embeddings/
│       └── gte/
│           ├── dense_embeddings.pkl
│           ├── sparse_embeddings.pkl
│           └── metadata.json
└── results/                 # Evaluation results
    └── *.json
```

## Metrics

The following IR metrics are computed at various k values:

| Metric | Description |
|--------|-------------|
| **Precision@k** | Fraction of retrieved docs that are relevant |
| **Recall@k** | Fraction of relevant docs that are retrieved |
| **F1@k** | Harmonic mean of Precision and Recall |
| **NDCG@k** | Normalized Discounted Cumulative Gain |
| **Hit Rate@k** | Whether any relevant doc is in top-k |
| **MRR** | Mean Reciprocal Rank of first relevant doc |
| **MAP** | Mean Average Precision |

## Dataset

The evaluation uses the **ZacLegalTextRetrieval** dataset:

| Component | Count | Description |
|-----------|-------|-------------|
| Corpus | 61,425 | Legal document passages |
| Queries (test) | 818 | Legal questions |
| Qrels (test) | 793 | Query-document relevance pairs |

Source: [Zalo AI Challenge](https://challenge.zalo.ai/)

## Adding New Embedding Models

To evaluate with a different embedding model:

1. **Create a Kaggle notebook** variant (e.g., `kaggle_embed_corpus_bge.ipynb`)
2. **Implement the provider** in `embeddings/` (inherit from `EmbeddingProvider`)
3. **Register** in `embeddings/__init__.py`
4. **Generate embeddings** on Kaggle and download
5. **Ingest** to a new collection: `--collection bench_bge`
6. **Evaluate**: `--collection bench_bge --embedding bge`

### Example: Adding BGE-M3

```python
# benchmarks/embeddings/bge.py
from .base import EmbeddingProvider, EmbeddingOutput

class BGEProvider(EmbeddingProvider):
    @property
    def name(self) -> str:
        return "bge"
    
    @property
    def model_name(self) -> str:
        return "BAAI/bge-m3"
    
    # ... implement encode() method
```

## Example Results

```
============================================================
EVALUATION RESULTS
============================================================

Configuration:
  embedding_model: gte
  collection: bench_gte
  use_hybrid: True
  num_queries: 793

Aggregate Metrics:

  k     P@k      R@k      F1@k     NDCG@k   Hit@k
  --------------------------------------------------
  1     0.5200   0.5200   0.5200   0.5200   0.5200
  3     0.2500   0.7500   0.3750   0.6100   0.7500
  5     0.1800   0.9000   0.3000   0.6800   0.9000
  10    0.0980   0.9800   0.1782   0.7200   0.9800
  20    0.0500   1.0000   0.0952   0.7500   1.0000

  MRR:  0.6500
  MAP:  0.6200

============================================================
```

## Configuration

Environment variables (optional):

```bash
BENCH_QDRANT_HOST=localhost
BENCH_QDRANT_PORT=6333
BENCH_ZALO_DATA_DIR=data/zalo_ai_retrieval
```

Or modify `benchmarks/config.py` directly.

## Troubleshooting

### "Collection does not exist"

Run ingestion first:
```bash
python -m benchmarks.ingest --embeddings-dir benchmarks/data/embeddings/gte --collection bench_gte
```

### Out of memory on Kaggle

Reduce batch size in the notebook:
```python
CONFIG["batch_size"] = 16  # or 8
```

### Qdrant connection refused

Ensure Qdrant is running:
```bash
docker-compose up -d qdrant
```

## License

MIT License - See project root LICENSE file.
