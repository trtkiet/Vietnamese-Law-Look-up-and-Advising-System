"""
Benchmarks module for retrieval and answer generation evaluation.

This module provides tools for evaluating RAG pipeline performance:

Retrieval Evaluation (using Zalo AI Legal Text Retrieval dataset):
1. Direct Qdrant evaluation (RetrievalEvaluator)
2. Pipeline adapter evaluation (PipelineEvaluator) - tests actual RAG pipelines

Answer Generation Evaluation (using DeepEval):
1. DeepEvalEvaluator - evaluates answer quality with metrics like
   AnswerRelevancy and Faithfulness using Gemini as the judge LLM.

Usage:
    # Retrieval evaluation
    python -m benchmarks.run_eval --collection bench_gte

    # Answer generation evaluation
    python -m benchmarks.run_deepeval --dataset data/qna_dataset.json
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

# DeepEval answer generation evaluation
from .deepeval_config import deepeval_config, setup_gemini_for_deepeval
from .deepeval_dataset import QnADataset, load_and_generate
from .deepeval_evaluator import (
    DeepEvalEvaluator,
    DeepEvalResult,
    run_evaluation as run_deepeval_evaluation,
    get_available_metrics,
    print_summary as print_deepeval_summary,
)

__all__ = [
    # Dataset
    "EvalDataset",
    # Retrieval Evaluators
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
    # DeepEval Answer Generation
    "deepeval_config",
    "setup_gemini_for_deepeval",
    "QnADataset",
    "load_and_generate",
    "DeepEvalEvaluator",
    "DeepEvalResult",
    "run_deepeval_evaluation",
    "get_available_metrics",
    "print_deepeval_summary",
]
