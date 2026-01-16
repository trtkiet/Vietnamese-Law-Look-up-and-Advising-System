"""Embedding providers for benchmarks."""

from .base import EmbeddingProvider, EmbeddingOutput
from .gte import GTEProvider
from .vietnamese import VietnameseProvider
from .bge_m3 import BGEM3Provider

__all__ = [
    "EmbeddingProvider",
    "EmbeddingOutput",
    "GTEProvider",
    "VietnameseProvider",
    "BGEM3Provider",
]


def get_embedding_provider(name: str) -> EmbeddingProvider:
    """
    Get an embedding provider by name.

    Args:
        name: Provider name ('gte', 'vietnamese', 'bge_m3', etc.)

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If provider name is not recognized
    """
    providers = {
        "gte": GTEProvider,
        "vietnamese": VietnameseProvider,
        "bge_m3": BGEM3Provider,
        # Future providers:
        # "e5": E5Provider,
    }

    if name not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"Unknown embedding provider: {name}. Available: {available}")

    return providers[name]()
