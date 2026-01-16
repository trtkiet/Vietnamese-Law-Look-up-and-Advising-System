"""
Qdrant retriever wrapper for benchmark evaluation.

This module provides a clean interface for retrieving documents from Qdrant
with support for both dense and hybrid (dense + sparse) search.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from .config import config

logger = logging.getLogger(__name__)


def corpus_id_to_uuid(corpus_id: str) -> str:
    """
    Convert corpus ID to deterministic UUID.

    Uses UUID5 with URL namespace to generate consistent UUIDs from corpus IDs.
    This ensures the same corpus_id always maps to the same UUID.

    Args:
        corpus_id: Original corpus ID (e.g., "01/2009/tt-bnn+1")

    Returns:
        UUID string
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, corpus_id))


def to_qdrant_sparse(token_weights: Dict[int, float]) -> SparseVector:
    """
    Convert token weights dict to Qdrant SparseVector.

    Args:
        token_weights: Dict mapping token IDs to weights

    Returns:
        Qdrant SparseVector object
    """
    if not token_weights:
        return SparseVector(indices=[], values=[])

    sorted_indices = sorted(token_weights.keys())
    values = [token_weights[idx] for idx in sorted_indices]

    return SparseVector(indices=sorted_indices, values=values)


class QdrantRetriever:
    """
    Wrapper for Qdrant retrieval operations.

    Supports both dense-only and hybrid (dense + sparse) search.
    Returns corpus IDs (from payload) for evaluation matching.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
    ):
        """
        Initialize the retriever.

        Args:
            host: Qdrant host (default: from config)
            port: Qdrant port (default: from config)
            collection_name: Collection to search (default: from config)
        """
        self.host = host or config.QDRANT_HOST
        self.port = port or config.QDRANT_PORT
        self.collection_name = collection_name or f"{config.EVAL_COLLECTION_PREFIX}gte"

        self.client = QdrantClient(host=self.host, port=self.port)

    def _extract_corpus_id(self, payload: Dict) -> str:
        """
        Extract corpus_id from payload, handling nested metadata structure.

        Supports both flat and nested payload formats:
        - Flat: {"corpus_id": "..."}
        - Nested: {"metadata": {"corpus_id": "..."}}

        Args:
            payload: Point payload from Qdrant

        Returns:
            corpus_id string
        """
        # Try direct access first
        if "corpus_id" in payload:
            return payload["corpus_id"]

        # Try nested metadata (matches ingest_data_alibaba.py structure)
        if "metadata" in payload and isinstance(payload["metadata"], dict):
            return payload["metadata"].get("corpus_id", "")

        logger.warning(f"Could not find corpus_id in payload: {payload}")
        return ""

    def retrieve_dense(
        self,
        query_embedding: List[float],
        k: int = 20,
    ) -> List[str]:
        """
        Retrieve documents using dense embedding only.

        Args:
            query_embedding: Query dense embedding vector
            k: Number of documents to retrieve

        Returns:
            List of corpus_id strings in ranked order
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_embedding),
            limit=k,
            with_payload=True,  # Get full payload to access nested metadata
        )

        return [self._extract_corpus_id(hit.payload) for hit in results]

    def retrieve_hybrid(
        self,
        query_embedding: List[float],
        query_sparse: Dict[int, float],
        k: int = 20,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[str]:
        """
        Retrieve documents using hybrid (dense + sparse) search.

        Uses Qdrant's query with prefetch for hybrid search.

        Args:
            query_embedding: Query dense embedding vector
            query_sparse: Query sparse token weights
            k: Number of documents to retrieve
            dense_weight: Weight for dense scores (not used in prefetch mode)
            sparse_weight: Weight for sparse scores (not used in prefetch mode)

        Returns:
            List of corpus_id strings in ranked order
        """
        # Use Qdrant's recommended hybrid search with prefetch
        sparse_vector = to_qdrant_sparse(query_sparse)

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                rest.Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=k * 2,  # Fetch more candidates for fusion
                ),
                rest.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=k * 2,
                ),
            ],
            query=rest.FusionQuery(fusion=rest.Fusion.RRF),  # Reciprocal Rank Fusion
            limit=k,
            with_payload=True,  # Get full payload to access nested metadata
        )

        return [self._extract_corpus_id(point.payload) for point in results.points]

    def retrieve(
        self,
        query_embedding: List[float],
        query_sparse: Optional[Dict[int, float]] = None,
        k: int = 20,
        use_hybrid: bool = True,
    ) -> List[str]:
        """
        Retrieve documents using dense or hybrid search.

        Args:
            query_embedding: Query dense embedding vector
            query_sparse: Query sparse token weights (optional)
            k: Number of documents to retrieve
            use_hybrid: Whether to use hybrid search (if sparse is available)

        Returns:
            List of corpus_id strings in ranked order
        """
        if use_hybrid and query_sparse:
            return self.retrieve_hybrid(query_embedding, query_sparse, k)
        else:
            return self.retrieve_dense(query_embedding, k)

    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def get_collection_info(self) -> Optional[Dict]:
        """Get collection information."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None


def create_benchmark_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int = 768,
    recreate: bool = False,
) -> None:
    """
    Create a Qdrant collection for benchmark evaluation.

    Args:
        client: Qdrant client
        collection_name: Name for the collection
        dense_dim: Dimension of dense vectors
        recreate: Whether to delete existing collection
    """
    if client.collection_exists(collection_name):
        if recreate:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(f"Collection {collection_name} already exists")
            return

    logger.info(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
