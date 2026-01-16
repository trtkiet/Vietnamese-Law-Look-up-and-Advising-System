"""
BGE-M3 (BAAI/bge-m3) embedding provider.

This module implements the EmbeddingProvider interface for the
BAAI/bge-m3 multilingual model, supporting both dense and sparse
(lexical) embeddings.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from FlagEmbedding import BGEM3FlagModel

from .base import EmbeddingOutput, EmbeddingProvider

logger = logging.getLogger(__name__)


class BGEM3Embedding:
    """
    Core BGE-M3 embedding model.

    This class handles the actual embedding computation using the
    FlagEmbedding.BGEM3FlagModel class. Supports both dense
    and sparse (lexical) embeddings.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = False,
        device: str = None,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16

        # Device setup
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                use_fp16 = False

        logger.info(f"Using device: {self.device}, FP16: {self.use_fp16}")

        # Load model
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=self.use_fp16,
            device=self.device,
        )
        self.model.model.eval()  # Force eval mode
        self.tokenizer = self.model.tokenizer

    def _to_token_id_weights(
        self, lexical_weights: Dict[str, float]
    ) -> Dict[int, float]:
        """
        Convert BGE-M3 lexical_weights ({token_string: weight}) to {token_id: weight}.

        Args:
            lexical_weights: Dict mapping token strings to weights

        Returns:
            Dict mapping token IDs to weights
        """
        # Special tokens to filter out
        unused_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        }

        result = defaultdict(float)
        for token_id_str, weight in lexical_weights.items():
            # BGE-M3 returns keys as stringified IDs, not token strings
            try:
                token_id = int(token_id_str)
            except (ValueError, TypeError):
                continue

            if token_id not in unused_tokens and weight > 0:
                result[token_id] = max(result[token_id], weight)
        return dict(result)

    def encode(
        self,
        texts: List[str],
        max_length: int = 8192,
        batch_size: int = 16,
        return_dense: bool = True,
        return_sparse: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Main encoding function with batching.

        Args:
            texts: List of texts to encode
            max_length: Maximum token length
            batch_size: Batch size for encoding
            return_dense: Return dense embeddings
            return_sparse: Return sparse embeddings
            show_progress: Show progress bar

        Returns:
            Dict with "dense_embeddings" and/or "sparse_embeddings" keys
        """
        if isinstance(texts, str):
            texts = [texts]

        output = {}

        if return_dense:
            output["dense_embeddings"] = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
            )["dense_vecs"]

        if return_sparse:
            sparse_outputs = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=False,
                return_sparse=True,
            )

            # Convert {token_string: weight} to {token_id: weight}
            sparse_converted = [
                self._to_token_id_weights(d) for d in sparse_outputs["lexical_weights"]
            ]
            output["sparse_embeddings"] = sparse_converted

        return output


class BGEM3Provider(EmbeddingProvider):
    """
    BGE-M3 multilingual embedding provider.

    Implements the EmbeddingProvider interface for the
    BAAI/bge-m3 model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = False,
        device: str = None,
    ):
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._device = device
        self._model: Optional[BGEM3Embedding] = None

    @property
    def name(self) -> str:
        return "bge_m3"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dense_dim(self) -> int:
        return 1024

    @property
    def supports_sparse(self) -> bool:
        return True

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading BGE-M3 model: {self._model_name}")
            self._model = BGEM3Embedding(
                model_name=self._model_name,
                use_fp16=self._use_fp16,
                device=self._device,
            )

    def encode(
        self,
        texts: List[str],
        return_dense: bool = True,
        return_sparse: bool = False,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> EmbeddingOutput:
        """Encode texts using the BGE-M3 model."""
        self._load_model()

        result = self._model.encode(
            texts,
            return_dense=return_dense,
            return_sparse=return_sparse,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        dense = None
        sparse = None

        if return_dense and "dense_embeddings" in result:
            dense = result["dense_embeddings"].tolist()

        if return_sparse and "sparse_embeddings" in result:
            sparse = result["sparse_embeddings"]

        return EmbeddingOutput(dense=dense, sparse=sparse)
