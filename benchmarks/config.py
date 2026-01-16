"""Configuration for benchmarks module."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Try to load .env if dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class BenchmarkConfig:
    """Configuration for retrieval benchmarks."""

    # Data paths (relative to project root)
    ZALO_DATA_DIR: str = "data/zalo_ai_retrieval"
    EMBEDDINGS_DIR: str = "benchmarks/data/embeddings"
    RESULTS_DIR: str = "benchmarks/results"

    # Qdrant configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    EVAL_COLLECTION_PREFIX: str = "bench_"  # e.g., bench_gte, bench_bge

    # Evaluation settings
    DEFAULT_K_VALUES: List[int] = field(default_factory=lambda: [3, 5, 10, 20, 50])
    SPLIT: str = "test"  # test or train

    # Embedding settings
    DEFAULT_EMBEDDING_MODEL: str = "gte"
    BATCH_SIZE: int = 32
    MAX_LENGTH: int = 512

    # Vector dimensions
    DENSE_VECTOR_SIZE: int = 768

    def __post_init__(self):
        """Load values from environment variables with BENCH_ prefix."""
        for field_name in self.__dataclass_fields__:
            env_key = f"BENCH_{field_name}"
            if env_key in os.environ:
                value = os.environ[env_key]
                field_type = self.__dataclass_fields__[field_name].type
                if field_type == int:
                    value = int(value)
                elif field_type == List[int]:
                    value = [int(x.strip()) for x in value.split(",")]
                setattr(self, field_name, value)


config = BenchmarkConfig()


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_path(relative_path: str) -> Path:
    """Get absolute path from project-relative path."""
    return get_project_root() / relative_path
