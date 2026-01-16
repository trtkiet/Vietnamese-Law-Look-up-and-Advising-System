"""
DeepEval Evaluator for RAG answer generation quality.

This module provides the core evaluation logic using DeepEval metrics
to assess answer quality from RAG pipelines.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from .deepeval_config import (
    deepeval_config,
    setup_gemini_for_deepeval,
    get_results_path,
    get_eval_model,
)
from .deepeval_dataset import load_and_generate

logger = logging.getLogger(__name__)

# Add backend to path for imports
_backend_path = Path(__file__).parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))


# =============================================================================
# Metric Registry
# =============================================================================

METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {
    "answer_relevancy": AnswerRelevancyMetric,
    "faithfulness": FaithfulnessMetric,
}


def get_available_metrics() -> List[str]:
    """Get list of available metric names."""
    return list(METRIC_REGISTRY.keys())


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class MetricResult:
    """Result for a single metric evaluation."""

    name: str
    score: float
    passed: bool
    threshold: float
    reason: Optional[str] = None


@dataclass
class QuestionResult:
    """Result for a single question."""

    question: str
    expected_answer: str
    actual_answer: str
    retrieval_context: List[str]
    metrics: List[MetricResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "actual_answer": self.actual_answer,
            "retrieval_context": self.retrieval_context,
            "metrics": [asdict(m) for m in self.metrics],
        }


@dataclass
class DeepEvalResult:
    """Complete evaluation result."""

    config: Dict[str, Any]
    aggregate_metrics: Dict[str, float]
    pass_rates: Dict[str, float]
    per_question_results: List[QuestionResult]
    runtime_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config,
            "aggregate_metrics": self.aggregate_metrics,
            "pass_rates": self.pass_rates,
            "per_question_results": [q.to_dict() for q in self.per_question_results],
            "runtime_seconds": self.runtime_seconds,
            "timestamp": self.timestamp,
        }

    def save(self, filepath: str) -> str:
        """
        Save results to JSON file.

        Args:
            filepath: Path to save the results.

        Returns:
            Absolute path to the saved file.
        """
        path = (
            get_results_path(filepath)
            if not Path(filepath).is_absolute()
            else Path(filepath)
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        return str(path)


# =============================================================================
# Pipeline Factory
# =============================================================================


def get_pipeline(
    pipeline_name: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: Optional[str] = None,
    top_k: Optional[int] = None,
):
    """
    Get a pipeline instance by name.

    Args:
        pipeline_name: Name of the pipeline ("vietnamese", "gte", "bge").
        qdrant_host: Qdrant host.
        qdrant_port: Qdrant port.
        collection_name: Optional collection name override.
        top_k: Number of documents to return after reranking.

    Returns:
        RAGPipeline instance.
    """
    from services.pipelines import (
        VietnameseEmbeddingPipeline,
        GTEPipeline,
        BGEM3Pipeline,
    )

    pipeline_map = {
        "vietnamese": VietnameseEmbeddingPipeline,
        "gte": GTEPipeline,
        "bge": BGEM3Pipeline,
    }

    if pipeline_name not in pipeline_map:
        raise ValueError(
            f"Unknown pipeline: {pipeline_name}. Available: {list(pipeline_map.keys())}"
        )

    pipeline_class = pipeline_map[pipeline_name]

    # Build kwargs
    kwargs = {
        "qdrant_host": qdrant_host,
        "qdrant_port": qdrant_port,
        "skip_llm": False,  # We need LLM for answer generation
    }

    if collection_name:
        kwargs["collection_name"] = collection_name

    if top_k is not None:
        kwargs["top_k"] = top_k

    return pipeline_class(**kwargs)


# =============================================================================
# Evaluator Class
# =============================================================================


class DeepEvalEvaluator:
    """
    Evaluator for RAG answer generation quality using DeepEval.

    This evaluator:
    1. Loads a QnA dataset
    2. Runs the pipeline to generate answers
    3. Evaluates using DeepEval metrics (AnswerRelevancy, Faithfulness)
    4. Returns aggregated and per-question results

    Example:
        evaluator = DeepEvalEvaluator(
            pipeline=VietnameseEmbeddingPipeline(),
            metrics=["answer_relevancy", "faithfulness"],
            threshold=0.5,
        )
        result = evaluator.run("data/qna_dataset.json")
        result.save("results/deepeval_vietnamese.json")
    """

    def __init__(
        self,
        pipeline,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the evaluator.

        Args:
            pipeline: RAGPipeline instance for generating answers.
            metrics: List of metric names to use. Default: ["answer_relevancy", "faithfulness"]
            threshold: Passing threshold for metrics (0.0 - 1.0).
        """
        self.pipeline = pipeline
        self.metric_names = metrics or ["answer_relevancy", "faithfulness"]
        self.threshold = threshold

        # Validate metric names
        for name in self.metric_names:
            if name not in METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown metric: {name}. Available: {get_available_metrics()}"
                )

        # Setup Gemini for DeepEval
        setup_gemini_for_deepeval()

    def _create_metrics(self) -> List[BaseMetric]:
        """Create metric instances with configured threshold and GeminiModel."""
        metrics = []

        # Get the GeminiModel instance
        eval_model = get_eval_model()

        for name in self.metric_names:
            metric_class = METRIC_REGISTRY[name]
            metric = metric_class(
                threshold=self.threshold,
                model=eval_model,
            )
            metrics.append(metric)

        return metrics

    def run(
        self,
        dataset_path: str,
        limit: Optional[int] = None,
        show_progress: bool = True,
    ) -> DeepEvalResult:
        """
        Run evaluation on the dataset.

        Args:
            dataset_path: Path to the QnA JSON file.
            limit: Maximum number of questions to evaluate.
            show_progress: Whether to show progress bars.

        Returns:
            DeepEvalResult with aggregated and per-question metrics.
        """
        logger.info(f"Starting DeepEval evaluation on {dataset_path}")
        start_time = time.perf_counter()

        # Ensure pipeline is initialized
        if not self.pipeline.is_initialized:
            logger.info("Initializing pipeline...")
            self.pipeline.startup()

        # Load dataset and generate test cases
        logger.info("Generating answers for test cases...")
        test_cases = load_and_generate(
            dataset_path=dataset_path,
            pipeline=self.pipeline,
            limit=limit,
            show_progress=show_progress,
        )

        if not test_cases:
            raise ValueError("No test cases generated from dataset")

        # Create metrics
        metrics = self._create_metrics()
        logger.info(
            f"Evaluating with metrics: {[m.__class__.__name__ for m in metrics]}"
        )

        # Run DeepEval evaluation and collect results
        logger.info("Running DeepEval evaluation...")

        per_question_results = []
        metric_scores: Dict[str, List[float]] = {
            m.__class__.__name__: [] for m in metrics
        }
        metric_passes: Dict[str, List[bool]] = {
            m.__class__.__name__: [] for m in metrics
        }

        try:
            from tqdm import tqdm

            iterator = tqdm(test_cases, desc="Evaluating")
        except ImportError:
            iterator = test_cases

        for test_case in iterator:
            question_metrics = []

            for metric in metrics:
                metric_name = metric.__class__.__name__
                try:
                    # Measure this test case with this metric
                    metric.measure(test_case)
                    score = metric.score if metric.score is not None else 0.0
                    reason = metric.reason
                except Exception as e:
                    logger.warning(f"Metric {metric_name} failed: {e}")
                    score = 0.0
                    reason = f"Error: {str(e)}"

                passed = score >= self.threshold

                metric_scores[metric_name].append(score)
                metric_passes[metric_name].append(passed)

                question_metrics.append(
                    MetricResult(
                        name=metric_name,
                        score=score,
                        passed=passed,
                        threshold=self.threshold,
                        reason=reason,
                    )
                )

            per_question_results.append(
                QuestionResult(
                    question=test_case.input,
                    expected_answer=test_case.expected_output,
                    actual_answer=test_case.actual_output,
                    retrieval_context=test_case.retrieval_context or [],
                    metrics=question_metrics,
                )
            )

        # Process results
        runtime_seconds = time.perf_counter() - start_time

        # Calculate aggregates
        aggregate_metrics = {}
        pass_rates = {}

        for metric_name, scores in metric_scores.items():
            if scores:
                aggregate_metrics[metric_name] = sum(scores) / len(scores)
                passes = metric_passes[metric_name]
                pass_rates[metric_name] = sum(passes) / len(passes) if passes else 0.0
            else:
                aggregate_metrics[metric_name] = 0.0
                pass_rates[metric_name] = 0.0

        # Build config - use backend config for model name since GeminiModel.model is a Client object
        from core.config import config as backend_config

        result_config = {
            "dataset_path": dataset_path,
            "pipeline": self.pipeline.__class__.__name__,
            "metrics": self.metric_names,
            "threshold": self.threshold,
            "num_questions": len(test_cases),
            "limit": limit,
            "model": backend_config.MODEL,
        }

        result = DeepEvalResult(
            config=result_config,
            aggregate_metrics=aggregate_metrics,
            pass_rates=pass_rates,
            per_question_results=per_question_results,
            runtime_seconds=runtime_seconds,
        )

        logger.info(f"Evaluation completed in {runtime_seconds:.2f}s")
        return result


# =============================================================================
# Convenience Functions
# =============================================================================


def run_evaluation(
    dataset_path: str,
    pipeline_name: str = "vietnamese",
    metrics: Optional[List[str]] = None,
    threshold: float = 0.5,
    limit: Optional[int] = None,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: Optional[str] = None,
    top_k: Optional[int] = None,
    show_progress: bool = True,
) -> DeepEvalResult:
    """
    Convenience function to run a complete evaluation.

    Args:
        dataset_path: Path to the QnA JSON file.
        pipeline_name: Name of the pipeline to use.
        metrics: List of metric names.
        threshold: Passing threshold.
        limit: Maximum number of questions.
        qdrant_host: Qdrant host.
        qdrant_port: Qdrant port.
        collection_name: Optional collection name override.
        top_k: Number of documents to return after reranking.
        show_progress: Whether to show progress bars.

    Returns:
        DeepEvalResult with evaluation results.
    """
    # Create pipeline
    pipeline = get_pipeline(
        pipeline_name=pipeline_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=collection_name,
        top_k=top_k,
    )

    # Create evaluator
    evaluator = DeepEvalEvaluator(
        pipeline=pipeline,
        metrics=metrics,
        threshold=threshold,
    )

    # Run evaluation
    return evaluator.run(
        dataset_path=dataset_path,
        limit=limit,
        show_progress=show_progress,
    )


def print_summary(result: DeepEvalResult) -> None:
    """
    Print a summary of the evaluation results.

    Args:
        result: DeepEvalResult to summarize.
    """
    print("\n" + "=" * 60)
    print("              DEEPEVAL EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nPipeline: {result.config.get('pipeline', 'N/A')}")
    print(f"Dataset: {result.config.get('dataset_path', 'N/A')}")
    print(f"Questions evaluated: {result.config.get('num_questions', 0)}")
    print(f"Threshold: {result.config.get('threshold', 0.5)}")

    print("\n" + "-" * 60)
    print("METRICS:")
    print("-" * 60)

    for metric_name, score in result.aggregate_metrics.items():
        pass_rate = result.pass_rates.get(metric_name, 0.0)
        num_questions = result.config.get("num_questions", 0)
        num_passed = int(pass_rate * num_questions)
        print(f"  {metric_name}:")
        print(f"    Average Score: {score:.4f}")
        print(f"    Pass Rate: {pass_rate:.2%} ({num_passed}/{num_questions})")

    print("\n" + "-" * 60)
    print(f"Runtime: {result.runtime_seconds:.2f}s")
    if result.config.get("num_questions", 0) > 0:
        avg_time = result.runtime_seconds / result.config["num_questions"]
        print(f"Average time per question: {avg_time:.2f}s")
    print("=" * 60 + "\n")
