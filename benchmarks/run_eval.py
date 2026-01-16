"""
CLI entry point for retrieval evaluation.

Usage:
    # Using direct Qdrant retrieval (original mode)
    python -m benchmarks.run_eval \
        --collection bench_gte \
        --embedding gte \
        --k-values 1,3,5,10,20 \
        --output results/gte_hybrid.json

    # Using GTEPipeline adapter (tests actual pipeline)
    python -m benchmarks.run_eval \
        --mode pipeline \
        --collection bench_gte \
        --output results/pipeline_hybrid.json

    # Using GTEPipeline with dense-only retrieval
    python -m benchmarks.run_eval \
        --mode pipeline \
        --collection bench_gte \
        --retrieval-mode dense

    # Using GTEPipeline with reranker
    python -m benchmarks.run_eval \
        --mode pipeline \
        --collection bench_gte \
        --reranker

    # Using GTEPipeline with custom reranker model
    python -m benchmarks.run_eval \
        --mode pipeline \
        --collection bench_gte \
        --reranker-model "BAAI/bge-reranker-v2-m3"

    # Retrieve more candidates for reranking
    python -m benchmarks.run_eval \
        --mode pipeline \
        --collection bench_gte \
        --reranker \
        --retrieval-k 100
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from .config import config, get_data_path
from .evaluator import PipelineEvaluator, RetrievalEvaluator
from .results import print_summary, save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_direct_eval(args, k_values):
    """Run evaluation using direct Qdrant retrieval."""
    evaluator = RetrievalEvaluator.create(
        data_dir=args.data_dir,
        split="test",
        collection_name=args.collection,
        embedding_name=args.embedding,
        qdrant_host=args.host,
        qdrant_port=args.port,
    )

    return evaluator.evaluate(
        k_values=k_values,
        use_hybrid=not args.no_hybrid,
        save_per_query=args.per_query,
        limit=args.limit,
    )


def run_pipeline_eval(args, k_values):
    """Run evaluation using pipeline adapter."""
    # retrieval_k: use explicit arg, or default to max(k_values)
    if args.retrieval_k is not None:
        retrieval_k = args.retrieval_k
    else:
        retrieval_k = max(k_values) if k_values else 50

    adapter_kwargs = {
        "collection_name": args.collection,
        "qdrant_host": args.host or "localhost",
        "qdrant_port": args.port or 6333,
    }

    if args.mode in ("pipeline", "gte", "vietnamese"):
        adapter_kwargs["retrieval_mode"] = args.retrieval_mode
        adapter_kwargs["use_reranker"] = args.reranker
        adapter_kwargs["reranker_model"] = args.reranker_model
        adapter_kwargs["retrieval_k"] = retrieval_k
    elif args.mode == "direct":
        adapter_kwargs["use_hybrid"] = not args.no_hybrid

    evaluator = PipelineEvaluator.create(
        adapter_type=args.mode,
        data_dir=args.data_dir,
        split="test",
        **adapter_kwargs,
    )

    return evaluator.evaluate(
        k_values=k_values,
        save_per_query=args.per_query,
        limit=args.limit,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Direct Qdrant evaluation (original mode)
    python -m benchmarks.run_eval --collection bench_gte

    # Using GTEPipeline adapter (default: hybrid retrieval)
    python -m benchmarks.run_eval --mode pipeline --collection bench_gte

    # Pipeline with dense-only retrieval 
    python -m benchmarks.run_eval --mode pipeline --collection bench_gte --retrieval-mode dense

    # Pipeline with sparse-only retrieval
    python -m benchmarks.run_eval --mode pipeline --collection bench_gte --retrieval-mode sparse

    # Pipeline with reranker enabled
    python -m benchmarks.run_eval --mode pipeline --collection bench_gte --reranker

    # Pipeline with custom reranker model (auto-enables reranker)
    python -m benchmarks.run_eval --mode pipeline --collection bench_gte --reranker-model "BAAI/bge-reranker-v2-m3"

    # Retrieve more candidates for reranking (retrieval_k > max k_values)
    python -m benchmarks.run_eval --mode pipeline --collection bench_gte --reranker --retrieval-k 100

    # Quick test with limited queries
    python -m benchmarks.run_eval --collection bench_gte --limit 50
""",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="qdrant",
        choices=["qdrant", "pipeline", "gte", "vietnamese", "direct"],
        help="Evaluation mode: 'qdrant' (original), 'pipeline'/'gte' (GTEPipeline), 'vietnamese' (VietnameseEmbeddingPipeline), 'direct' (lightweight Qdrant)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Qdrant collection name (e.g., bench_gte)",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="gte",
        choices=["gte", "vietnamese"],  # Add more as implemented
        help="Embedding provider to use (default: gte) - only for qdrant mode",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to Zalo AI retrieval data (default: data/zalo_ai_retrieval)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10,20,50",
        help="Comma-separated k values for metrics (default: 1,3,5,10,20)",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Use dense-only search (default: hybrid dense+sparse)",
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Enable cross-encoder reranking (only for pipeline mode)",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help="Reranker model name (e.g., 'Alibaba-NLP/gte-multilingual-reranker-base'). "
        "Implies --reranker.",
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "dense", "sparse"],
        help="Retrieval strategy for pipeline mode: 'hybrid' (dense+sparse), 'dense', or 'sparse' (default: hybrid)",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=None,
        help="Number of documents to retrieve from vector store. "
        "If not specified, defaults to max(k_values). "
        "Set higher than max(k_values) when using reranker for better results.",
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Save per-query detailed results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (for debugging)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/<collection>_<timestamp>.json)",
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
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()

    # Auto-enable reranker if model is specified
    if args.reranker_model:
        args.reranker = True

    # Parse k values
    try:
        k_values = [int(k.strip()) for k in args.k_values.split(",")]
    except ValueError:
        logger.error(f"Invalid k-values format: {args.k_values}")
        sys.exit(1)

    # Run evaluation based on mode
    logger.info(f"Starting evaluation in '{args.mode}' mode...")
    try:
        if args.mode == "qdrant":
            result = run_direct_eval(args, k_values)
        else:
            # pipeline, gte, or direct modes
            result = run_pipeline_eval(args, k_values)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    # Generate output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.mode in ("pipeline", "gte"):
            # Include retrieval mode in filename
            mode_suffix = f"pipeline_{args.retrieval_mode}"
            if args.reranker:
                mode_suffix += "_reranker"
        elif args.mode == "qdrant":
            mode_suffix = "qdrant_hybrid" if not args.no_hybrid else "qdrant_dense"
        else:
            mode_suffix = args.mode
        output_path = f"{args.collection}_{mode_suffix}_{timestamp}.json"

    # Save results
    saved_path = save_results(result, output_path)

    # Print summary
    if not args.quiet:
        print_summary(result)
        print(f"\nResults saved to: {saved_path}")


if __name__ == "__main__":
    main()
