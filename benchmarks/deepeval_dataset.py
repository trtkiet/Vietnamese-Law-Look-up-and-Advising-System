"""
QnA Dataset loader for DeepEval answer generation evaluation.

This module loads Q&A pairs from JSON and converts them to DeepEval
LLMTestCase objects by running the RAG pipeline.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)

# Add backend to path for imports
_backend_path = Path(__file__).parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))


@dataclass
class QnAPair:
    """A single question-answer pair from the dataset."""

    question: str
    expected_answer: str


@dataclass
class GeneratedTestCase:
    """Test case with generated answer and retrieved context."""

    question: str
    expected_answer: str
    actual_answer: str
    retrieval_context: List[str]
    sources: List[Dict[str, Any]]


class QnADataset:
    """
    Loader for QnA dataset, converting to DeepEval test cases.

    The dataset JSON format:
    [
        {"question": "...", "answer": "..."},
        ...
    ]
    """

    def __init__(self, qna_pairs: List[QnAPair]):
        """
        Initialize with loaded QnA pairs.

        Args:
            qna_pairs: List of question-answer pairs.
        """
        self.qna_pairs = qna_pairs

    @classmethod
    def load(cls, filepath: str) -> "QnADataset":
        """
        Load QnA pairs from JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            QnADataset instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Dataset must be a JSON array")

        qna_pairs = []
        for idx, item in enumerate(data):
            if "question" not in item or "answer" not in item:
                logger.warning(
                    f"Skipping item {idx}: missing 'question' or 'answer' field"
                )
                continue
            qna_pairs.append(
                QnAPair(
                    question=item["question"],
                    expected_answer=item["answer"],
                )
            )

        logger.info(f"Loaded {len(qna_pairs)} QnA pairs from {filepath}")
        return cls(qna_pairs)

    def __len__(self) -> int:
        """Return number of QnA pairs."""
        return len(self.qna_pairs)

    def __iter__(self):
        """Iterate over QnA pairs."""
        return iter(self.qna_pairs)

    def get_questions(self) -> List[str]:
        """Get list of all questions."""
        return [pair.question for pair in self.qna_pairs]

    def subset(self, limit: Optional[int] = None) -> "QnADataset":
        """
        Get a subset of the dataset.

        Args:
            limit: Maximum number of pairs to include.

        Returns:
            New QnADataset with limited pairs.
        """
        if limit is None or limit >= len(self.qna_pairs):
            return self
        return QnADataset(self.qna_pairs[:limit])


def generate_test_cases(
    dataset: QnADataset,
    pipeline,
    show_progress: bool = True,
) -> List[GeneratedTestCase]:
    """
    Generate test cases by running the pipeline on each question.

    For each QnA pair:
    1. Call pipeline.respond(question) to get actual answer
    2. Extract retrieval context from sources
    3. Create GeneratedTestCase

    Args:
        dataset: QnADataset with questions and expected answers.
        pipeline: RAGPipeline instance (must have respond() method).
        show_progress: Whether to show progress bar.

    Returns:
        List of GeneratedTestCase objects.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x  # noqa: E731
        show_progress = False

    test_cases = []
    iterator = tqdm(dataset, desc="Generating answers") if show_progress else dataset

    for qna_pair in iterator:
        try:
            # Run the pipeline to get answer and context
            result = pipeline.respond(qna_pair.question)

            actual_answer = result.get("answer", "")
            sources = result.get("sources", [])

            # Extract context text from sources
            # The pipeline returns sources as list of metadata dicts
            # We need to get the actual document content
            retrieval_context = _extract_context_from_pipeline(
                pipeline, qna_pair.question
            )

            test_cases.append(
                GeneratedTestCase(
                    question=qna_pair.question,
                    expected_answer=qna_pair.expected_answer,
                    actual_answer=actual_answer,
                    retrieval_context=retrieval_context,
                    sources=sources,
                )
            )

        except Exception as e:
            logger.error(
                f"Error processing question: {qna_pair.question[:50]}... - {e}"
            )
            # Add empty test case to maintain order
            test_cases.append(
                GeneratedTestCase(
                    question=qna_pair.question,
                    expected_answer=qna_pair.expected_answer,
                    actual_answer=f"Error: {str(e)}",
                    retrieval_context=[],
                    sources=[],
                )
            )

    return test_cases


def _extract_context_from_pipeline(pipeline, query: str) -> List[str]:
    """
    Extract retrieval context documents from pipeline.

    Args:
        pipeline: RAGPipeline instance.
        query: The question to retrieve context for.

    Returns:
        List of context document texts.
    """
    try:
        # Use retrieve_context to get the actual context text
        context_result = pipeline.retrieve_context(query)
        context_text = context_result.get("context", "")

        # Split by double newlines (the pipeline joins with \n\n)
        if context_text:
            contexts = [c.strip() for c in context_text.split("\n\n") if c.strip()]
            return contexts
        return []
    except Exception as e:
        logger.warning(f"Failed to extract context: {e}")
        return []


def to_deepeval_test_cases(
    generated_cases: List[GeneratedTestCase],
) -> List[LLMTestCase]:
    """
    Convert GeneratedTestCase objects to DeepEval LLMTestCase objects.

    Args:
        generated_cases: List of generated test cases.

    Returns:
        List of DeepEval LLMTestCase objects.
    """
    deepeval_cases = []

    for case in generated_cases:
        deepeval_case = LLMTestCase(
            input=case.question,
            actual_output=case.actual_answer,
            expected_output=case.expected_answer,
            retrieval_context=case.retrieval_context,
        )
        deepeval_cases.append(deepeval_case)

    return deepeval_cases


def load_and_generate(
    dataset_path: str,
    pipeline,
    limit: Optional[int] = None,
    show_progress: bool = True,
) -> List[LLMTestCase]:
    """
    Convenience function to load dataset and generate DeepEval test cases.

    Args:
        dataset_path: Path to the QnA JSON file.
        pipeline: RAGPipeline instance.
        limit: Maximum number of questions to process.
        show_progress: Whether to show progress bar.

    Returns:
        List of DeepEval LLMTestCase objects ready for evaluation.
    """
    # Load dataset
    dataset = QnADataset.load(dataset_path)

    # Apply limit if specified
    if limit:
        dataset = dataset.subset(limit)

    # Ensure pipeline is initialized
    if not pipeline.is_initialized:
        logger.info("Initializing pipeline...")
        pipeline.startup()

    # Generate test cases
    generated_cases = generate_test_cases(dataset, pipeline, show_progress)

    # Convert to DeepEval format
    deepeval_cases = to_deepeval_test_cases(generated_cases)

    logger.info(f"Generated {len(deepeval_cases)} DeepEval test cases")
    return deepeval_cases
