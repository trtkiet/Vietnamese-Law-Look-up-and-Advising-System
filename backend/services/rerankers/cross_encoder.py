"""Cross-encoder based reranker implementation with performance optimizations."""
import logging
import time
from typing import List, Optional
import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from services.rerankers.base import BaseReranker

import numpy as np
logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Optimized reranker using sentence-transformers CrossEncoder.
    
    Performance optimizations:
    - torch.compile() for ~2x speedup (PyTorch 2.0+)
    - Inference mode for reduced overhead
    - Optimized batch sizing
    - Early termination for small document sets
    - Memory-efficient scoring
    """

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-reranker-base",
        max_length: int = 1024,
        device: Optional[str] = None,
        batch_size: Optional[int] = 32,  # Default batch size for better performance
        use_compile: bool = False,  # Enable torch.compile for speed
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._device = device
        self.batch_size = batch_size
        self.use_compile = use_compile
        self._model: Optional[CrossEncoder] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def startup(self) -> None:
        if self._initialized:
            return

        logger.info(f"Loading CrossEncoderReranker: {self.model_name}")
        t_start = time.time()

        # Device detection
        if self._device is not None:
            device = self._device
            cuda_available = device == "cuda"
        else:
            cuda_available = torch.cuda.is_available()
            device = "cuda" if cuda_available else "cpu"

        # Log device info
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
            except Exception:
                logger.info("CUDA available - using GPU")
        else:
            logger.warning(
                "CUDA not available! Reranker will run on CPU (slower). "
                "Check NVIDIA Container Toolkit installation."
            )

        logger.info(f"Loading Reranker on: {device}")

        try:
            # Load model with optimized settings
            model_kwargs = {}
            if cuda_available:
                model_kwargs["dtype"] = torch.float16
                # Enable TF32 for even faster computation on Ampere+ GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            self._model = CrossEncoder(
                self.model_name,
                device=device,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
            )
            self._model.max_length = self.max_length

            # Apply torch.compile for ~2x speedup (PyTorch 2.0+, requires Triton/Linux)
            if self.use_compile and cuda_available:
                try:
                    if hasattr(torch, 'compile'):
                        logger.info("Applying torch.compile() optimization...")
                        self._model.model = torch.compile(
                            self._model.model,
                            mode="reduce-overhead"  # Best for inference
                        )
                        logger.info("torch.compile() applied successfully")
                except Exception as e:
                    logger.info(f"Skipping torch.compile (requires Triton on Linux): {type(e).__name__}")
                    # Continue without compile - other optimizations still apply

            # Warmup with small batch to compile CUDA kernels
            if cuda_available:
                logger.info("Warming up reranker...")
                warmup_pairs = [
                    ["warmup query", "warmup document"]
                    for _ in range(min(self.batch_size or 32, 8))  # Small batch for warmup
                ]
                with torch.inference_mode():
                    _ = self._model.predict(
                        warmup_pairs,
                        batch_size=len(warmup_pairs),
                        show_progress_bar=False,
                    )
                logger.info("Warmup complete")

            self._initialized = True
            logger.info(f"Reranker loaded in {time.time() - t_start:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load CrossEncoderReranker: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        if not self._initialized or self._model is None:
            raise RuntimeError(
                "CrossEncoderReranker not initialized. Call startup() first."
            )

        if not documents:
            return []

        # Early return if we have fewer documents than top_k
        # if len(documents) <= top_k:
        #     logger.info(f"Returning all {len(documents)} documents (≤ top_k)")
        #     return documents

        logger.info(f"Reranking {len(documents)} documents, returning top {top_k}")
        t_start = time.time()

        # Build query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Determine optimal batch size
        batch_size = self.batch_size if self.batch_size else len(pairs)

        # Score with inference mode for reduced overhead
        with torch.inference_mode():
            scores = self._model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=False,
            )

        # Use numpy/torch operations for faster sorting if available
        scores_array = np.array(scores)
        
        # Get top_k indices using argpartition for O(n) vs O(n log n)
        if len(scores_array) > top_k:
            # argpartition is faster than full sort for large arrays
            top_indices = np.argpartition(scores_array, -top_k)[-top_k:]
            # Sort only the top_k elements
            top_indices = top_indices[np.argsort(scores_array[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores_array)[::-1][:top_k]

        top_k_docs = [documents[i] for i in top_indices]

        logger.info(f"Reranking complete in {time.time() - t_start:.4f}s")
        return top_k_docs