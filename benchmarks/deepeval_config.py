"""
DeepEval configuration for answer generation evaluation.

This module configures DeepEval to use Gemini as the evaluation LLM,
matching the existing backend configuration.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from deepeval.models import GeminiModel

# Try to load .env if dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Add backend to path for imports
_backend_path = Path(__file__).parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))


def _get_backend_config():
    """Get the backend config singleton."""
    from core.config import config

    return config


@dataclass
class DeepEvalConfig:
    """Configuration for DeepEval answer generation evaluation."""

    # ==========================================================================
    # Metric Thresholds
    # ==========================================================================
    # Threshold for passing a metric (0.0 - 1.0)
    # A score >= threshold is considered "passed"
    ANSWER_RELEVANCY_THRESHOLD: float = 0.5
    FAITHFULNESS_THRESHOLD: float = 0.5

    # ==========================================================================
    # Dataset Settings
    # ==========================================================================
    DEFAULT_DATASET_PATH: str = "data/qna_dataset.json"

    # ==========================================================================
    # Output Settings
    # ==========================================================================
    RESULTS_DIR: str = "benchmarks/results"

    # ==========================================================================
    # Pipeline Settings
    # ==========================================================================
    # Available pipelines for evaluation
    AVAILABLE_PIPELINES: List[str] = field(
        default_factory=lambda: ["vietnamese", "gte", "bge"]
    )
    DEFAULT_PIPELINE: str = "vietnamese"

    # Qdrant connection defaults
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    def __post_init__(self):
        """Load values from environment variables with DEEPEVAL_ prefix."""
        for field_name in self.__dataclass_fields__:
            env_key = f"DEEPEVAL_{field_name}"
            if env_key in os.environ:
                value = os.environ[env_key]
                field_type = self.__dataclass_fields__[field_name].type
                if field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)
                setattr(self, field_name, value)


# Singleton configuration instance
deepeval_config = DeepEvalConfig()

# Cached GeminiModel instance
_gemini_model_instance: Optional["GeminiModel"] = None


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_results_path(filename: str) -> Path:
    """Get absolute path for results file."""
    results_dir = get_project_root() / deepeval_config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / filename


def get_eval_model() -> "GeminiModel":
    """
    Get the GeminiModel instance for DeepEval metrics.

    Uses the backend config for model name and API key.
    The model is cached as a singleton.

    Returns:
        GeminiModel instance configured with backend settings.
    """
    global _gemini_model_instance

    if _gemini_model_instance is None:
        from deepeval.models import GeminiModel

        backend_config = _get_backend_config()

        _gemini_model_instance = GeminiModel(
            model=backend_config.MODEL,
            api_key=backend_config.GEMINI_API_KEY,
        )

    return _gemini_model_instance


def setup_gemini_for_deepeval() -> None:
    """
    Configure environment for DeepEval to use Gemini.

    This function ensures the GeminiModel is ready for use.
    Called automatically when creating metrics.
    """
    # Ensure the model can be created (validates config)
    _ = get_eval_model()
