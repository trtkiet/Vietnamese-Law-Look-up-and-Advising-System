"""
Benchmarks module for retrieval evaluation.

This module provides tools for evaluating retrieval pipeline performance
using the Zalo AI Legal Text Retrieval dataset.

Two evaluation modes are supported:
1. Direct Qdrant evaluation (RetrievalEvaluator)
2. Pipeline adapter evaluation (PipelineEvaluator) - tests actual RAG pipelines
"""

from .dataset import EvalDataset
from .evaluator import PipelineEvaluator, RetrievalEvaluator
from .metrics import compute_metrics
from .pipeline_adapter import (
    BaseRetrieverAdapter,
    RAGPipelineAdapter,
    QdrantDirectAdapter,
    get_retriever_adapter,
    list_available_adapters,
)
from .results import EvalResult, load_results, save_results

__all__ = [
    # Dataset
    "EvalDataset",
    # Evaluators
    "RetrievalEvaluator",
    "PipelineEvaluator",
    # Adapters
    "BaseRetrieverAdapter",
    "RAGPipelineAdapter",
    "QdrantDirectAdapter",
    "get_retriever_adapter",
    "list_available_adapters",
    # Metrics & Results
    "compute_metrics",
    "EvalResult",
    "save_results",
    "load_results",
]
