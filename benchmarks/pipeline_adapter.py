"""
Pipeline adapter for evaluating existing RAG pipelines.

This module wraps the actual pipelines from backend/services/pipelines/
to extract retrieved document IDs for benchmark evaluation.

The adapter uses retrieve_context() to get sources, then extracts corpus_id
from the metadata for computing IR metrics.
"""

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Type

logger = logging.getLogger(__name__)

# Add backend to path for imports
_backend_path = Path(__file__).parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))


class BaseRetrieverAdapter(ABC):
    """Abstract base class for retriever adapters."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 20) -> List[str]:
        """
        Retrieve document IDs for evaluation.

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            List of corpus_id strings in ranked order
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the retriever for logging/results."""
        pass


class RAGPipelineAdapter(BaseRetrieverAdapter):
    """
    Generic adapter that wraps any RAGPipeline for evaluation.

    This adapter:
    1. Creates a RAGPipeline with custom benchmark configuration
    2. Uses retrieve_context() to get documents
    3. Extracts corpus_id from the sources metadata
    4. Returns ranked list of corpus IDs for metric computation

    The benchmark collection must have payloads structured as:
        {
            "page_content": "...",
            "metadata": {
                "corpus_id": "...",
                ...
            }
        }
    """

    def __init__(
        self,
        pipeline_class: Type,
        collection_name: str = "bench_gte",
        retrieval_mode: str = "hybrid",
        use_reranker: bool = False,
        reranker_model: Optional[str] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        corpus_id_field: str = "corpus_id",
        pipeline_name: str | None = None,
        retrieval_k: int = 50,
    ):
        """
        Initialize the adapter.

        Args:
            pipeline_class: The RAGPipeline class to instantiate
            collection_name: Qdrant collection containing benchmark corpus
            retrieval_mode: Retrieval strategy - "hybrid", "dense", or "sparse"
            use_reranker: Whether to enable cross-encoder reranking
            reranker_model: Custom reranker model name (e.g., 'BAAI/bge-reranker-v2-m3').
                          If None, uses the default CrossEncoderReranker model.
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
            corpus_id_field: Field name in metadata containing corpus ID
            pipeline_name: Custom name for the pipeline (defaults to class name)
            retrieval_k: Number of documents to retrieve from vector store.
                        Should be >= max(k_values) in evaluation. Default: 50.
        """
        self.pipeline_class = pipeline_class
        self.collection_name = collection_name
        self.retrieval_mode = retrieval_mode
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.corpus_id_field = corpus_id_field
        self._pipeline_name = pipeline_name or pipeline_class.__name__
        self.retrieval_k = retrieval_k

        self._pipeline = None
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy initialization of the pipeline."""
        if self._initialized:
            return

        reranker_info = ""
        if self.use_reranker:
            reranker_info = f", reranker_model: {self.reranker_model or 'default'}"

        logger.info(
            f"Initializing {self._pipeline_name} with collection: {self.collection_name}, "
            f"mode: {self.retrieval_mode}, retrieval_k: {self.retrieval_k}{reranker_info}"
        )

        # Build pipeline kwargs
        pipeline_kwargs = {
            "retrieval_mode": self.retrieval_mode,
            "use_reranker": self.use_reranker,
            "collection_name": self.collection_name,
            "qdrant_host": self.qdrant_host,
            "qdrant_port": self.qdrant_port,
            "skip_llm": True,  # Skip LLM for retrieval-only benchmarks
            "retrieval_k": self.retrieval_k,
            "top_k": self.retrieval_k,  # Return all retrieved docs (slicing done in retrieve())
        }

        # Create custom reranker if model is specified
        if self.use_reranker and self.reranker_model:
            from services.rerankers import CrossEncoderReranker

            custom_reranker = CrossEncoderReranker(model_name=self.reranker_model)
            pipeline_kwargs["reranker"] = custom_reranker

        # Create pipeline with benchmark configuration
        # For benchmarking: retrieval_k = top_k (no reranking truncation)
        # This allows evaluating at different k values
        self._pipeline = self.pipeline_class(**pipeline_kwargs)

        # Initialize the pipeline
        self._pipeline.startup()

        self._initialized = True
        logger.info(f"{self._pipeline_name} initialized successfully")

    def retrieve(self, query: str, k: int = 20) -> List[str]:
        """
        Retrieve document IDs for a query using the actual pipeline.

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            List of corpus_id strings in ranked order
        """
        self._initialize()

        # Use the pipeline's retrieve_context method
        result = self._pipeline.retrieve_context(query)
        sources = result.get("sources", [])

        # Extract corpus IDs from sources metadata
        corpus_ids = []
        for source in sources[:k]:
            corpus_id = source.get(self.corpus_id_field)
            if corpus_id:
                corpus_ids.append(corpus_id)
            else:
                logger.warning(f"Source missing {self.corpus_id_field}: {source}")

        return corpus_ids

    @property
    def name(self) -> str:
        """Name of the retriever."""
        base_name = f"{self._pipeline_name}-{self.retrieval_mode}"
        if self.use_reranker:
            if self.reranker_model:
                # Extract short model name from full path
                model_short = self.reranker_model.split("/")[-1]
                return f"{base_name}+Reranker({model_short})"
            return f"{base_name}+Reranker"
        return base_name


class QdrantDirectAdapter(BaseRetrieverAdapter):
    """
    Direct Qdrant adapter without LangChain overhead.

    This is a lightweight alternative that queries Qdrant directly,
    useful for faster evaluation or when LangChain is not needed.
    """

    def __init__(
        self,
        collection_name: str = "bench_gte",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        use_hybrid: bool = True,
    ):
        """
        Initialize the adapter.

        Args:
            collection_name: Qdrant collection name
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
            use_hybrid: Whether to use hybrid (dense+sparse) search
        """
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.use_hybrid = use_hybrid

        self._retriever = None
        self._embedding_provider = None
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy initialization."""
        if self._initialized:
            return

        from .embeddings import get_embedding_provider
        from .retriever import QdrantRetriever

        self._retriever = QdrantRetriever(
            host=self.qdrant_host,
            port=self.qdrant_port,
            collection_name=self.collection_name,
        )
        self._embedding_provider = get_embedding_provider("gte")
        self._initialized = True

    def retrieve(self, query: str, k: int = 20) -> List[str]:
        """Retrieve document IDs for a query."""
        self._initialize()

        # Encode query
        emb = self._embedding_provider.encode_query(query)

        # Retrieve
        query_sparse = emb.sparse[0] if emb.sparse and self.use_hybrid else None
        return self._retriever.retrieve(
            query_embedding=emb.dense[0],
            query_sparse=query_sparse,
            k=k,
            use_hybrid=self.use_hybrid,
        )

    @property
    def name(self) -> str:
        """Name of the retriever."""
        mode = "Hybrid" if self.use_hybrid else "Dense"
        return f"QdrantDirect-{mode}"


# =============================================================================
# Pipeline Registry
# =============================================================================
# Add new pipelines here to make them available for benchmarking.
# Each entry maps a name to the pipeline class.


def _get_pipeline_registry() -> dict[str, Type]:
    """
    Get the registry of available pipeline classes.

    Returns:
        Dictionary mapping pipeline names to their classes.
    """
    from services.pipelines import GTEPipeline, VietnameseEmbeddingPipeline

    return {
        "gte": GTEPipeline,
        "pipeline": GTEPipeline,  # Alias for backward compatibility
        "vietnamese": VietnameseEmbeddingPipeline,
        # Add new pipelines here:
        # "bm25": BM25Pipeline,
    }


def get_retriever_adapter(
    adapter_type: str = "gte",
    **kwargs,
) -> BaseRetrieverAdapter:
    """
    Factory function to create retriever adapters.

    Args:
        adapter_type: Type of adapter. Options:
            - "gte": GTEPipeline (default)
            - "pipeline": GTEPipeline (alias)
            - "direct": QdrantDirectAdapter (lightweight, no LangChain)
            - Any key from the pipeline registry
        **kwargs: Arguments passed to adapter constructor.
            For GTEPipeline adapters:
                - retrieval_mode: "hybrid", "dense", or "sparse"
                - use_reranker: bool
                - reranker_model: str (e.g., "BAAI/bge-reranker-v2-m3")
                - collection_name: str

    Returns:
        Configured retriever adapter

    Example:
        # Use GTEPipeline with hybrid retrieval (default)
        adapter = get_retriever_adapter("gte", retrieval_mode="hybrid")

        # Use dense-only retrieval with reranker
        adapter = get_retriever_adapter("gte", retrieval_mode="dense", use_reranker=True)

        # Use custom reranker model
        adapter = get_retriever_adapter(
            "gte",
            use_reranker=True,
            reranker_model="BAAI/bge-reranker-v2-m3"
        )

        # Use direct Qdrant adapter for fast evaluation
        adapter = get_retriever_adapter("direct", use_hybrid=True)
    """
    # Special case for direct adapter
    if adapter_type == "direct":
        return QdrantDirectAdapter(**kwargs)

    # Get pipeline from registry
    registry = _get_pipeline_registry()

    if adapter_type not in registry:
        available = ["direct"] + list(registry.keys())
        raise ValueError(
            f"Unknown adapter type: {adapter_type}. Available: {available}"
        )

    pipeline_class = registry[adapter_type]
    return RAGPipelineAdapter(pipeline_class=pipeline_class, **kwargs)


def list_available_adapters() -> List[str]:
    """
    List all available adapter types.

    Returns:
        List of adapter type names.
    """
    registry = _get_pipeline_registry()
    return ["direct"] + list(registry.keys())
