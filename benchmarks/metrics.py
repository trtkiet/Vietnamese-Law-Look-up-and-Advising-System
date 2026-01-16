"""
Information Retrieval metrics for retrieval evaluation.

This module implements standard IR metrics:
- Precision@k
- Recall@k
- F1@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@k)
- Hit Rate@k

Note: When documents are chunked, multiple chunks from the same document
may appear in retrieved results. This module provides deduplication to
ensure document-level metrics are computed correctly.
"""

import math
from typing import Dict, List, Set, Union


def deduplicate_retrieved(retrieved: List[str]) -> List[str]:
    """
    Remove duplicate document IDs while preserving rank order.

    When documents are chunked for embedding, multiple chunks from the same
    corpus document may be retrieved. This function deduplicates the list,
    keeping only the first occurrence of each document ID.

    Args:
        retrieved: Ordered list of retrieved document IDs (may contain duplicates)

    Returns:
        Deduplicated list preserving original rank order

    Example:
        >>> deduplicate_retrieved(["doc_A", "doc_A", "doc_B", "doc_A", "doc_C"])
        ["doc_A", "doc_B", "doc_C"]
    """
    seen: Set[str] = set()
    unique: List[str] = []
    for doc_id in retrieved:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Precision@k.

    Precision@k = |retrieved[:k] ∩ relevant| / k

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs (ground truth)
        k: Number of top results to consider

    Returns:
        Precision score between 0 and 1
    """
    if k <= 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Recall@k.

    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs (ground truth)
        k: Number of top results to consider

    Returns:
        Recall score between 0 and 1
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def f1_at_k(precision: float, recall: float) -> float:
    """
    Calculate F1@k from precision and recall.

    F1 = 2 * P * R / (P + R)

    Args:
        precision: Precision@k value
        recall: Recall@k value

    Returns:
        F1 score between 0 and 1
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Reciprocal Rank (for MRR).

    RR = 1 / rank of first relevant document

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs (ground truth)

    Returns:
        Reciprocal rank between 0 and 1 (0 if no relevant doc found)
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k.

    DCG@k = Σ (rel_i / log2(i + 2)) for i in 0..k-1

    For binary relevance (rel_i = 1 if relevant, 0 otherwise).

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs (ground truth)
        k: Number of top results to consider

    Returns:
        DCG score (unnormalized)
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            # Binary relevance: 1 if relevant, 0 otherwise
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    NDCG@k = DCG@k / IDCG@k

    Where IDCG is the ideal DCG (all relevant docs at top).

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs (ground truth)
        k: Number of top results to consider

    Returns:
        NDCG score between 0 and 1
    """
    # Calculate actual DCG
    actual_dcg = dcg_at_k(retrieved, relevant, k)

    # Calculate ideal DCG (all relevant docs ranked at top)
    ideal_num_relevant = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_num_relevant))

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Hit Rate at k (also known as Success@k).

    Hit@k = 1 if any relevant document is in top-k, else 0

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs (ground truth)
        k: Number of top results to consider

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return 1.0 if retrieved_k & relevant else 0.0


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Average Precision (AP).

    AP = (1/|relevant|) * Σ P(k) * rel(k)

    Where P(k) is precision at k, and rel(k) is 1 if doc at k is relevant.

    Args:
        retrieved: Ordered list of retrieved document IDs
        relevant: Set of relevant document IDs (ground truth)

    Returns:
        AP score between 0 and 1
    """
    if not relevant:
        return 0.0

    num_relevant_seen = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            num_relevant_seen += 1
            precision_at_i = num_relevant_seen / (i + 1)
            precision_sum += precision_at_i

    return precision_sum / len(relevant)


def compute_metrics(
    retrieved: List[str],
    relevant: Set[str],
    k_values: List[int],
    deduplicate: bool = True,
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Compute all retrieval metrics for a single query.

    When documents are chunked, multiple chunks from the same corpus document
    may appear in retrieved results. By default, this function deduplicates
    the retrieved list to compute document-level metrics correctly.

    Args:
        retrieved: Ordered list of retrieved document IDs (may contain duplicates
                   when chunks from same document are retrieved)
        relevant: Set of relevant document IDs (ground truth)
        k_values: List of k values to compute metrics at (e.g., [1, 3, 5, 10, 20])
        deduplicate: If True (default), remove duplicate document IDs before
                     computing metrics. Set to False for chunk-level evaluation.

    Returns:
        Dictionary with metrics:
        {
            "precision": {k: value, ...},
            "recall": {k: value, ...},
            "f1": {k: value, ...},
            "ndcg": {k: value, ...},
            "hit_rate": {k: value, ...},
            "mrr": float,
            "ap": float,
        }
    """
    # Deduplicate retrieved list for document-level metrics
    if deduplicate:
        retrieved = deduplicate_retrieved(retrieved)

    results: Dict[str, Union[float, Dict[int, float]]] = {
        "precision": {},
        "recall": {},
        "f1": {},
        "ndcg": {},
        "hit_rate": {},
    }

    for k in k_values:
        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)

        results["precision"][k] = p
        results["recall"][k] = r
        results["f1"][k] = f1_at_k(p, r)
        results["ndcg"][k] = ndcg_at_k(retrieved, relevant, k)
        results["hit_rate"][k] = hit_rate_at_k(retrieved, relevant, k)

    # Metrics that don't depend on k
    results["mrr"] = reciprocal_rank(retrieved, relevant)
    results["ap"] = average_precision(retrieved, relevant)

    return results


def aggregate_metrics(
    all_metrics: List[Dict[str, Union[float, Dict[int, float]]]],
    k_values: List[int],
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Aggregate metrics across multiple queries (compute means).

    Args:
        all_metrics: List of per-query metric dictionaries
        k_values: List of k values that were computed

    Returns:
        Dictionary with averaged metrics (same structure as compute_metrics output)
    """
    if not all_metrics:
        return {}

    n = len(all_metrics)

    aggregated: Dict[str, Union[float, Dict[int, float]]] = {
        "precision": {},
        "recall": {},
        "f1": {},
        "ndcg": {},
        "hit_rate": {},
    }

    # Aggregate k-dependent metrics
    for k in k_values:
        for metric_name in ["precision", "recall", "f1", "ndcg", "hit_rate"]:
            total = sum(m[metric_name][k] for m in all_metrics)
            aggregated[metric_name][k] = total / n

    # Aggregate k-independent metrics
    aggregated["mrr"] = sum(m["mrr"] for m in all_metrics) / n
    aggregated["ap"] = sum(m["ap"] for m in all_metrics) / n
    aggregated["map"] = aggregated["ap"]  # MAP = Mean Average Precision

    return aggregated
