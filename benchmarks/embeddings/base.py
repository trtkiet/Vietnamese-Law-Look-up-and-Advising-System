"""
Abstract base class for embedding providers.

This module defines the interface that all embedding providers must implement,
enabling easy swapping of embedding models for evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EmbeddingOutput:
    """
    Container for embedding outputs.

    Attributes:
        dense: List of dense embedding vectors (List[List[float]])
        sparse: List of sparse embedding dictionaries (List[Dict[int, float]])
               where keys are token IDs and values are weights
    """

    dense: Optional[List[List[float]]] = None
    sparse: Optional[List[Dict[int, float]]] = None

    def __len__(self) -> int:
        """Return number of embeddings."""
        if self.dense is not None:
            return len(self.dense)
        if self.sparse is not None:
            return len(self.sparse)
        return 0


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    All embedding models should inherit from this class and implement
    the required methods. This enables consistent API across different
    embedding models (GTE, BGE, E5, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this embedding model.

        Used for naming collections and result files.
        Example: 'gte', 'bge-m3', 'e5-large'
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Full HuggingFace model name.

        Example: 'Alibaba-NLP/gte-multilingual-base'
        """
        pass

    @property
    @abstractmethod
    def dense_dim(self) -> int:
        """
        Dimension of dense embeddings.

        Example: 768 for GTE base models
        """
        pass

    @property
    @abstractmethod
    def supports_sparse(self) -> bool:
        """
        Whether this provider supports sparse (SPLADE-style) embeddings.

        Returns True if the model can generate sparse token weights.
        """
        pass

    @abstractmethod
    def encode(
        self,
        texts: List[str],
        return_dense: bool = True,
        return_sparse: bool = False,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EmbeddingOutput:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            return_dense: Whether to return dense embeddings
            return_sparse: Whether to return sparse embeddings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            EmbeddingOutput containing the requested embeddings
        """
        pass

    def encode_query(self, text: str) -> EmbeddingOutput:
        """
        Encode a single query text.

        Some models treat queries differently from documents.
        Override this method if needed.

        Args:
            text: Query text to encode

        Returns:
            EmbeddingOutput for the single query
        """
        return self.encode(
            [text],
            return_dense=True,
            return_sparse=self.supports_sparse,
            show_progress=False,
        )

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EmbeddingOutput:
        """
        Encode document texts.

        Args:
            texts: List of document texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            EmbeddingOutput for all documents
        """
        return self.encode(
            texts,
            return_dense=True,
            return_sparse=self.supports_sparse,
            batch_size=batch_size,
            show_progress=show_progress,
        )
