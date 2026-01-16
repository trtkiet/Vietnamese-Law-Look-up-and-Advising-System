"""
Results storage and comparison utilities.

This module provides utilities for saving, loading, and comparing
evaluation results across different pipeline configurations.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import config, get_data_path

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """
    Container for evaluation results.

    Attributes:
        config: Evaluation configuration (embedding model, collection, etc.)
        aggregate_metrics: Averaged metrics across all queries
        per_query_results: Optional detailed per-query results
        timestamp: When the evaluation was run
        runtime_seconds: Total evaluation runtime in seconds
    """

    config: Dict[str, Any] = field(default_factory=dict)
    aggregate_metrics: Dict[str, Union[float, Dict[int, float]]] = field(
        default_factory=dict
    )
    per_query_results: Optional[List[Dict]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    runtime_seconds: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "config": self.config,
            "aggregate_metrics": self._serialize_metrics(self.aggregate_metrics),
            "timestamp": self.timestamp,
        }
        if self.runtime_seconds is not None:
            result["runtime_seconds"] = self.runtime_seconds
        if self.per_query_results:
            result["per_query_results"] = [
                self._serialize_query_result(r) for r in self.per_query_results
            ]
        return result

    @staticmethod
    def _serialize_metrics(
        metrics: Dict[str, Union[float, Dict[int, float]]],
    ) -> Dict[str, Any]:
        """Serialize metrics dict, converting int keys to strings for JSON."""
        serialized = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                # Convert int keys to strings
                serialized[key] = {str(k): v for k, v in value.items()}
            else:
                serialized[key] = value
        return serialized

    @staticmethod
    def _serialize_query_result(result: Dict) -> Dict:
        """Serialize a per-query result dict."""
        serialized = result.copy()
        if "relevant" in serialized and isinstance(serialized["relevant"], set):
            serialized["relevant"] = list(serialized["relevant"])
        if "metrics" in serialized:
            serialized["metrics"] = EvalResult._serialize_metrics(serialized["metrics"])
        return serialized

    @classmethod
    def from_dict(cls, data: Dict) -> "EvalResult":
        """Create from dictionary (JSON deserialization)."""
        # Convert string keys back to int for k-dependent metrics
        aggregate = cls._deserialize_metrics(data.get("aggregate_metrics", {}))

        per_query = None
        if "per_query_results" in data:
            per_query = [
                cls._deserialize_query_result(r) for r in data["per_query_results"]
            ]

        return cls(
            config=data.get("config", {}),
            aggregate_metrics=aggregate,
            per_query_results=per_query,
            timestamp=data.get("timestamp", ""),
            runtime_seconds=data.get("runtime_seconds"),
        )

    @staticmethod
    def _deserialize_metrics(
        metrics: Dict[str, Any],
    ) -> Dict[str, Union[float, Dict[int, float]]]:
        """Deserialize metrics dict, converting string keys back to int."""
        deserialized = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                # Convert string keys to int where appropriate
                deserialized[key] = {int(k): v for k, v in value.items()}
            else:
                deserialized[key] = value
        return deserialized

    @staticmethod
    def _deserialize_query_result(result: Dict) -> Dict:
        """Deserialize a per-query result dict."""
        deserialized = result.copy()
        if "relevant" in deserialized:
            deserialized["relevant"] = set(deserialized["relevant"])
        if "metrics" in deserialized:
            deserialized["metrics"] = EvalResult._deserialize_metrics(
                deserialized["metrics"]
            )
        return deserialized


def save_results(result: EvalResult, output_path: str) -> str:
    """
    Save evaluation results to JSON file.

    Args:
        result: EvalResult to save
        output_path: Path to save to (can be relative to results dir)

    Returns:
        Absolute path where results were saved
    """
    path = Path(output_path)
    if not path.is_absolute():
        path = get_data_path(config.RESULTS_DIR) / path

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {path}")
    return str(path)


def load_results(input_path: str) -> EvalResult:
    """
    Load evaluation results from JSON file.

    Args:
        input_path: Path to load from

    Returns:
        EvalResult instance
    """
    path = Path(input_path)
    if not path.is_absolute():
        path = get_data_path(config.RESULTS_DIR) / path

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return EvalResult.from_dict(data)


def compare_results(results: List[EvalResult], k_values: List[int] = None) -> str:
    """
    Generate a comparison table of multiple evaluation results.

    Args:
        results: List of EvalResult objects to compare
        k_values: Which k values to show (default: [1, 5, 10, 20])

    Returns:
        Formatted comparison table as string
    """
    k_values = k_values or [1, 5, 10, 20]

    # Build header
    lines = []
    header = "| Config |"
    for k in k_values:
        header += f" P@{k} | R@{k} | F1@{k} |"
    header += " MRR | MAP |"
    lines.append(header)

    # Separator
    sep = "|" + "-" * 20 + "|"
    for _ in k_values:
        sep += "-" * 6 + "|" + "-" * 6 + "|" + "-" * 7 + "|"
    sep += "-" * 6 + "|" + "-" * 6 + "|"
    lines.append(sep)

    # Data rows
    for result in results:
        name = result.config.get("embedding_model", "unknown")
        if result.config.get("use_hybrid"):
            name += "+hybrid"

        row = f"| {name:<18} |"
        metrics = result.aggregate_metrics

        for k in k_values:
            p = metrics.get("precision", {}).get(k, 0)
            r = metrics.get("recall", {}).get(k, 0)
            f1 = metrics.get("f1", {}).get(k, 0)
            row += f" {p:.3f} | {r:.3f} | {f1:.3f} |"

        mrr = metrics.get("mrr", 0)
        map_score = metrics.get("map", metrics.get("ap", 0))
        row += f" {mrr:.3f} | {map_score:.3f} |"

        lines.append(row)

    return "\n".join(lines)


def print_summary(result: EvalResult) -> None:
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Config
    print("\nConfiguration:")
    for key, value in result.config.items():
        print(f"  {key}: {value}")

    # Aggregate metrics
    print("\nAggregate Metrics:")
    metrics = result.aggregate_metrics

    # Print k-dependent metrics as table
    k_values = sorted(metrics.get("precision", {}).keys())
    if k_values:
        print(
            f"\n  {'k':<5} {'P@k':<8} {'R@k':<8} {'F1@k':<8} {'NDCG@k':<8} {'Hit@k':<8}"
        )
        print("  " + "-" * 50)
        for k in k_values:
            p = metrics["precision"].get(k, 0)
            r = metrics["recall"].get(k, 0)
            f1 = metrics["f1"].get(k, 0)
            ndcg = metrics.get("ndcg", {}).get(k, 0)
            hit = metrics.get("hit_rate", {}).get(k, 0)
            print(f"  {k:<5} {p:<8.4f} {r:<8.4f} {f1:<8.4f} {ndcg:<8.4f} {hit:<8.4f}")

    # Print k-independent metrics
    print(f"\n  MRR:  {metrics.get('mrr', 0):.4f}")
    print(f"  MAP:  {metrics.get('map', metrics.get('ap', 0)):.4f}")

    # Print runtime metrics
    if result.runtime_seconds is not None:
        num_queries = result.config.get("num_queries", 0)
        print("\nRuntime:")
        print(f"  Total time:  {result.runtime_seconds:.2f}s")
        if num_queries > 0:
            qps = num_queries / result.runtime_seconds
            avg_latency_ms = (result.runtime_seconds / num_queries) * 1000
            print(f"  Throughput:  {qps:.2f} queries/sec")
            print(f"  Avg latency: {avg_latency_ms:.1f}ms/query")

    print("\n" + "=" * 60)
