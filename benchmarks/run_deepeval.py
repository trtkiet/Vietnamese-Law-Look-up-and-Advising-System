"""
CLI entry point for DeepEval answer generation evaluation.

Usage:
    # Basic evaluation with default settings
    python -m benchmarks.run_deepeval --dataset data/qna_dataset.json

    # Quick test with 5 questions
    python -m benchmarks.run_deepeval --limit 5

    # Custom metrics and threshold
    python -m benchmarks.run_deepeval \
        --metrics answer_relevancy,faithfulness \
        --threshold 0.5

    # Specify pipeline and output
    python -m benchmarks.run_deepeval \
        --pipeline vietnamese \
        --output results/deepeval_vietnamese.json

    # With specific Qdrant connection
    python -m benchmarks.run_deepeval --host localhost --port 6333
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepEval answer generation evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    python -m benchmarks.run_deepeval --dataset data/qna_dataset.json

    # Quick test with limited questions
    python -m benchmarks.run_deepeval --limit 5

    # Custom threshold
    python -m benchmarks.run_deepeval --threshold 0.6

    # Specify output file
    python -m benchmarks.run_deepeval --output results/my_eval.json
""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/qna_dataset.json",
        help="Path to QnA dataset JSON (default: data/qna_dataset.json)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="vietnamese",
        choices=["vietnamese", "gte", "bge"],
        help="Pipeline to evaluate (default: vietnamese)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="answer_relevancy,faithfulness",
        help="Comma-separated metric names (default: answer_relevancy,faithfulness)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Passing threshold for metrics (default: 0.5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate (for debugging)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: auto-generated in benchmarks/results/)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Qdrant host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: use pipeline default)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of documents to return after reranking (default: use pipeline default)",
    )

    args = parser.parse_args()

    # Parse metrics
    try:
        metric_names = [m.strip() for m in args.metrics.split(",")]
    except Exception as e:
        logger.error(f"Invalid metrics format: {args.metrics}")
        sys.exit(1)

    # Validate metrics
    from .deepeval_evaluator import get_available_metrics

    available = get_available_metrics()
    for name in metric_names:
        if name not in available:
            logger.error(f"Unknown metric: {name}. Available: {available}")
            sys.exit(1)

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        # Try relative to project root
        from .deepeval_config import get_project_root

        dataset_path = get_project_root() / args.dataset
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {args.dataset}")
            sys.exit(1)

    # Run evaluation
    logger.info("=" * 60)
    logger.info("Starting DeepEval Answer Generation Evaluation")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Pipeline: {args.pipeline}")
    logger.info(f"Metrics: {metric_names}")
    logger.info(f"Threshold: {args.threshold}")
    if args.limit:
        logger.info(f"Limit: {args.limit} questions")
    if args.top_k:
        logger.info(f"Top-K: {args.top_k} documents")

    try:
        from .deepeval_evaluator import run_evaluation, print_summary

        result = run_evaluation(
            dataset_path=str(dataset_path),
            pipeline_name=args.pipeline,
            metrics=metric_names,
            threshold=args.threshold,
            limit=args.limit,
            qdrant_host=args.host,
            qdrant_port=args.port,
            collection_name=args.collection,
            top_k=args.top_k,
            show_progress=not args.no_progress,
        )

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)

    # Generate output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"deepeval_{args.pipeline}_{timestamp}.json"

    # Save results
    saved_path = result.save(output_path)

    # Print summary
    if not args.quiet:
        print_summary(result)
        print(f"Results saved to: {saved_path}")


if __name__ == "__main__":
    main()
