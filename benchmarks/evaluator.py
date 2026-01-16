"""
Retrieval Evaluator for benchmark evaluation.

This module orchestrates the evaluation of retrieval pipelines,
computing IR metrics across all test queries.

Two evaluator classes are provided:
- RetrievalEvaluator: Original evaluator using QdrantRetriever + EmbeddingProvider
- PipelineEvaluator: Evaluator using pipeline adapters (e.g., HybridRAGPipeline)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from tqdm import tqdm

from .config import config, get_data_path
from .dataset import EvalDataset
from .embeddings import EmbeddingProvider, get_embedding_provider
from .metrics import aggregate_metrics, compute_metrics
from .results import EvalResult
from .retriever import QdrantRetriever

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    Main evaluation orchestrator.

    Coordinates dataset loading, retrieval, and metrics computation.
    """

    def __init__(
        self,
        dataset: EvalDataset,
        retriever: QdrantRetriever,
        embedding_provider: EmbeddingProvider,
    ):
        """
        Initialize the evaluator.

        Args:
            dataset: Loaded evaluation dataset with queries and qrels
            retriever: Qdrant retriever instance
            embedding_provider: Embedding provider for encoding queries
        """
        self.dataset = dataset
        self.retriever = retriever
        self.embedding_provider = embedding_provider

    @classmethod
    def create(
        cls,
        data_dir: str = None,
        split: str = "test",
        collection_name: str = None,
        embedding_name: str = "gte",
        qdrant_host: str = None,
        qdrant_port: int = None,
    ) -> "RetrievalEvaluator":
        """
        Factory method to create evaluator with default settings.

        Args:
            data_dir: Path to Zalo AI retrieval data
            split: Which split to use ('test' or 'train')
            collection_name: Qdrant collection name
            embedding_name: Embedding provider name
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port

        Returns:
            Configured RetrievalEvaluator instance
        """
        # Load dataset
        data_path = data_dir or str(get_data_path(config.ZALO_DATA_DIR))
        logger.info(f"Loading dataset from: {data_path}")
        dataset = EvalDataset.load(data_path, split=split, load_corpus=False)

        # Create retriever
        retriever = QdrantRetriever(
            host=qdrant_host or config.QDRANT_HOST,
            port=qdrant_port or config.QDRANT_PORT,
            collection_name=collection_name or f"{config.EVAL_COLLECTION_PREFIX}gte",
        )

        # Check collection exists
        if not retriever.collection_exists():
            raise ValueError(
                f"Collection '{retriever.collection_name}' does not exist. "
                "Run ingestion first with: python -m benchmarks.ingest"
            )

        # Create embedding provider
        embedding_provider = get_embedding_provider(embedding_name)

        return cls(dataset, retriever, embedding_provider)

    def evaluate(
        self,
        k_values: List[int] = None,
        max_k: int = None,
        use_hybrid: bool = True,
        save_per_query: bool = False,
        limit: int = None,
    ) -> EvalResult:
        """
        Run evaluation on all test queries.

        Args:
            k_values: List of k values for metrics (default: [1, 3, 5, 10, 20])
            max_k: Maximum k for retrieval (default: max of k_values)
            use_hybrid: Whether to use hybrid (dense+sparse) search
            save_per_query: Whether to save per-query detailed results
            limit: Limit number of queries to evaluate (for debugging)

        Returns:
            EvalResult with aggregated and optionally per-query metrics
        """
        k_values = k_values or config.DEFAULT_K_VALUES
        max_k = max_k or max(k_values)

        # Get queries with ground truth
        eval_queries = self.dataset.get_eval_queries()
        if limit:
            eval_queries = eval_queries[:limit]

        logger.info(f"Evaluating {len(eval_queries)} queries")
        logger.info(f"k values: {k_values}")
        logger.info(f"Use hybrid: {use_hybrid}")

        all_metrics = []
        per_query_results = [] if save_per_query else None

        # Start timing
        start_time = time.perf_counter()

        for query_id, query_text in tqdm(eval_queries, desc="Evaluating"):
            # Encode query
            emb = self.embedding_provider.encode_query(query_text)

            # Retrieve
            query_sparse = emb.sparse[0] if emb.sparse and use_hybrid else None
            retrieved_ids = self.retriever.retrieve(
                query_embedding=emb.dense[0],
                query_sparse=query_sparse,
                k=max_k,
                use_hybrid=use_hybrid,
            )

            # Get ground truth
            relevant = self.dataset.get_relevant_docs(query_id)

            # Compute metrics
            metrics = compute_metrics(retrieved_ids, relevant, k_values)
            all_metrics.append(metrics)

            # Save per-query results if requested
            if save_per_query:
                per_query_results.append(
                    {
                        "query_id": query_id,
                        "query_text": query_text,
                        "retrieved": retrieved_ids[:max_k],
                        "relevant": relevant,
                        "metrics": metrics,
                    }
                )

        # End timing
        runtime_seconds = time.perf_counter() - start_time
        logger.info(f"Evaluation completed in {runtime_seconds:.2f}s")

        # Aggregate metrics
        aggregated = aggregate_metrics(all_metrics, k_values)

        # Build config info
        eval_config = {
            "embedding_model": self.embedding_provider.name,
            "collection": self.retriever.collection_name,
            "k_values": k_values,
            "use_hybrid": use_hybrid,
            "num_queries": len(eval_queries),
            "split": "test",  # TODO: make configurable
        }

        return EvalResult(
            config=eval_config,
            aggregate_metrics=aggregated,
            per_query_results=per_query_results,
            runtime_seconds=runtime_seconds,
        )

    def evaluate_single(
        self,
        query_id: str,
        k_values: List[int] = None,
        use_hybrid: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a single query (for debugging).

        Args:
            query_id: Query ID to evaluate
            k_values: List of k values for metrics
            use_hybrid: Whether to use hybrid search

        Returns:
            Dictionary with query details, retrieved docs, and metrics
        """
        k_values = k_values or config.DEFAULT_K_VALUES
        max_k = max(k_values)

        # Get query text
        query_text = self.dataset.queries.get(query_id)
        if not query_text:
            raise ValueError(f"Query ID not found: {query_id}")

        # Encode
        emb = self.embedding_provider.encode_query(query_text)

        # Retrieve
        query_sparse = emb.sparse[0] if emb.sparse and use_hybrid else None
        retrieved_ids = self.retriever.retrieve(
            query_embedding=emb.dense[0],
            query_sparse=query_sparse,
            k=max_k,
            use_hybrid=use_hybrid,
        )

        # Get ground truth
        relevant = self.dataset.get_relevant_docs(query_id)

        # Compute metrics
        metrics = compute_metrics(retrieved_ids, relevant, k_values)

        return {
            "query_id": query_id,
            "query_text": query_text,
            "retrieved": retrieved_ids,
            "relevant": list(relevant),
            "metrics": metrics,
        }


class PipelineEvaluator:
    """
    Evaluator for pipeline adapters (e.g., HybridRAGPipeline).

    Unlike RetrievalEvaluator, this class uses a retriever adapter that
    handles both embedding and retrieval internally, matching the actual
    pipeline behavior.
    """

    def __init__(
        self,
        dataset: EvalDataset,
        retriever_adapter: "BaseRetrieverAdapter",
    ):
        """
        Initialize the evaluator.

        Args:
            dataset: Loaded evaluation dataset with queries and qrels
            retriever_adapter: Adapter wrapping the pipeline's retrieval
        """
        self.dataset = dataset
        self.retriever = retriever_adapter

    @classmethod
    def create(
        cls,
        adapter_type: str = "pipeline",
        data_dir: str = None,
        split: str = "test",
        **adapter_kwargs,
    ) -> "PipelineEvaluator":
        """
        Factory method to create evaluator with a pipeline adapter.

        Args:
            adapter_type: Type of adapter ("pipeline" or "direct")
            data_dir: Path to Zalo AI retrieval data
            split: Which split to use ('test' or 'train')
            **adapter_kwargs: Arguments passed to the adapter

        Returns:
            Configured PipelineEvaluator instance
        """
        from .pipeline_adapter import get_retriever_adapter

        # Load dataset
        data_path = data_dir or str(get_data_path(config.ZALO_DATA_DIR))
        logger.info(f"Loading dataset from: {data_path}")
        dataset = EvalDataset.load(data_path, split=split, load_corpus=False)

        # Create adapter
        adapter = get_retriever_adapter(adapter_type, **adapter_kwargs)
        logger.info(f"Created adapter: {adapter.name}")

        return cls(dataset, adapter)

    def evaluate(
        self,
        k_values: List[int] = None,
        max_k: int = None,
        save_per_query: bool = False,
        limit: int = None,
    ) -> EvalResult:
        """
        Run evaluation on all test queries.

        Args:
            k_values: List of k values for metrics (default: [1, 3, 5, 10, 20])
            max_k: Maximum k for retrieval (default: max of k_values)
            save_per_query: Whether to save per-query detailed results
            limit: Limit number of queries to evaluate (for debugging)

        Returns:
            EvalResult with aggregated and optionally per-query metrics
        """
        k_values = k_values or config.DEFAULT_K_VALUES
        max_k = max_k or max(k_values)

        # Get queries with ground truth
        eval_queries = self.dataset.get_eval_queries()
        if limit:
            eval_queries = eval_queries[:limit]

        logger.info(
            f"Evaluating {len(eval_queries)} queries with {self.retriever.name}"
        )
        logger.info(f"k values: {k_values}")

        all_metrics = []
        per_query_results = [] if save_per_query else None

        # Start timing
        start_time = time.perf_counter()

        for query_id, query_text in tqdm(eval_queries, desc="Evaluating"):
            # Retrieve using adapter (handles embedding internally)
            retrieved_ids = self.retriever.retrieve(query_text, k=max_k)

            # Get ground truth
            relevant = self.dataset.get_relevant_docs(query_id)

            # Compute metrics
            metrics = compute_metrics(retrieved_ids, relevant, k_values)
            all_metrics.append(metrics)

            # Save per-query results if requested
            if save_per_query:
                per_query_results.append(
                    {
                        "query_id": query_id,
                        "query_text": query_text,
                        "retrieved": retrieved_ids[:max_k],
                        "relevant": list(relevant),
                        "metrics": metrics,
                    }
                )

        # End timing
        runtime_seconds = time.perf_counter() - start_time
        logger.info(f"Evaluation completed in {runtime_seconds:.2f}s")

        # Aggregate metrics
        aggregated = aggregate_metrics(all_metrics, k_values)

        # Build config info
        eval_config = {
            "retriever": self.retriever.name,
            "k_values": k_values,
            "num_queries": len(eval_queries),
            "split": "test",
        }

        return EvalResult(
            config=eval_config,
            aggregate_metrics=aggregated,
            per_query_results=per_query_results,
            runtime_seconds=runtime_seconds,
        )

    def evaluate_single(
        self,
        query_id: str,
        k_values: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single query (for debugging).

        Args:
            query_id: Query ID to evaluate
            k_values: List of k values for metrics

        Returns:
            Dictionary with query details, retrieved docs, and metrics
        """
        k_values = k_values or config.DEFAULT_K_VALUES
        max_k = max(k_values)

        # Get query text
        query_text = self.dataset.queries.get(query_id)
        if not query_text:
            raise ValueError(f"Query ID not found: {query_id}")

        # Retrieve using adapter
        retrieved_ids = self.retriever.retrieve(query_text, k=max_k)

        # Get ground truth
        relevant = self.dataset.get_relevant_docs(query_id)

        # Compute metrics
        metrics = compute_metrics(retrieved_ids, relevant, k_values)

        return {
            "query_id": query_id,
            "query_text": query_text,
            "retrieved": retrieved_ids,
            "relevant": list(relevant),
            "metrics": metrics,
        }
