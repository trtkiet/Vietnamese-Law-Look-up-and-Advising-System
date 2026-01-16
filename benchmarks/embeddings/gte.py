"""
GTE (General Text Embeddings) embedding provider.

This module implements the EmbeddingProvider interface for the
Alibaba-NLP/gte-multilingual-base model, supporting both dense
and sparse (SPLADE-style) embeddings.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.utils import is_torch_npu_available

from .base import EmbeddingOutput, EmbeddingProvider

logger = logging.getLogger(__name__)


class GTEEmbedding(torch.nn.Module):
    """
    Core GTE embedding model.

    This class handles the actual embedding computation using the
    GTE multilingual model. Supports both dense (CLS) and sparse
    (SPLADE-style) embeddings.
    """

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-base",
        normalized: bool = True,
        use_fp16: bool = False,
        device: str = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.normalized = normalized

        # Device setup
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif is_torch_npu_available():
                self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False  # CPU doesn't support fp16 well

        self.use_fp16 = use_fp16
        logger.info(f"Using device: {self.device}, FP16: {self.use_fp16}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
        )
        self.model.eval()
        self.model.to(self.device)

    def _process_token_weights(
        self, token_weights: np.ndarray, input_ids: list
    ) -> Dict[int, float]:
        """
        Aggregate weights for identical token IDs.

        Returns Dict[int, float] to prevent duplicate index errors.
        """
        result = defaultdict(float)
        unused_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        }

        for w, idx in zip(token_weights, input_ids):
            idx = int(idx)
            if idx not in unused_tokens and w > 0:
                # MAX strategy: keep max weight for duplicate tokens
                if w > result[idx]:
                    result[idx] = float(w)

        return dict(result)

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        dimension: int = None,
        max_length: int = 8192,
        batch_size: int = 16,
        return_dense: bool = True,
        return_sparse: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Main encoding function with batching and memory cleanup.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_dense_vecs = []
        all_token_weights = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", unit="batch")

        for i in iterator:
            batch = texts[i : i + batch_size]
            results = self._encode_batch(
                batch, dimension, max_length, return_dense, return_sparse
            )

            if return_dense:
                all_dense_vecs.append(results["dense_embeddings"])
            if return_sparse:
                all_token_weights.extend(results["token_weights"])

            # Memory cleanup
            del results
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        final_output = {}
        if return_dense and all_dense_vecs:
            final_output["dense_embeddings"] = torch.cat(all_dense_vecs, dim=0)
        else:
            final_output["dense_embeddings"] = []

        if return_sparse:
            final_output["token_weights"] = all_token_weights
        else:
            final_output["token_weights"] = []

        return final_output

    @torch.no_grad()
    def _encode_batch(
        self,
        texts: List[str],
        dimension: int = None,
        max_length: int = 1024,
        return_dense: bool = True,
        return_sparse: bool = False,
    ):
        """Encode a single batch of texts."""
        # Tokenize
        text_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        text_input = {k: v.to(self.device) for k, v in text_input.items()}

        # Forward pass
        model_out = self.model(**text_input, return_dict=True)

        output = {}

        # Dense embeddings (CLS token)
        if return_dense:
            dense_vecs = model_out.last_hidden_state[:, 0, :]
            if dimension:
                dense_vecs = dense_vecs[:, :dimension]
            if self.normalized:
                dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)
            output["dense_embeddings"] = dense_vecs.cpu()

        # Sparse embeddings (SPLADE-style)
        if return_sparse:
            weights = torch.relu(model_out.logits).squeeze(-1)
            weights_np = weights.detach().cpu().numpy()
            input_ids_np = text_input["input_ids"].cpu().numpy()

            output["token_weights"] = [
                self._process_token_weights(w, ids)
                for w, ids in zip(weights_np, input_ids_np)
            ]

        return output


class GTEProvider(EmbeddingProvider):
    """
    GTE Multilingual embedding provider.

    Implements the EmbeddingProvider interface for the
    Alibaba-NLP/gte-multilingual-base model.
    """

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-base",
        use_fp16: bool = False,
        device: str = None,
    ):
        self._model_name = model_name
        self._use_fp16 = use_fp16
        self._device = device
        self._model: Optional[GTEEmbedding] = None

    @property
    def name(self) -> str:
        return "gte"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dense_dim(self) -> int:
        return 768

    @property
    def supports_sparse(self) -> bool:
        return True

    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading GTE model: {self._model_name}")
            self._model = GTEEmbedding(
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
        """Encode texts using the GTE model."""
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

        if return_dense and len(result["dense_embeddings"]) > 0:
            dense = result["dense_embeddings"].numpy().tolist()

        if return_sparse:
            sparse = result["token_weights"]

        return EmbeddingOutput(dense=dense, sparse=sparse)
