#!/usr/bin/env python3
"""
Vietnamese Legal Document Chunking Pipeline CLI

Usage:
    python -m preprocess.chunk_pipeline --docs-root ./law_crawler/vbpl_documents
    python -m preprocess.chunk_pipeline --docs-root ./law_crawler/vbpl_documents --output ./data/chunks.json
    python -m preprocess.chunk_pipeline --help
"""

import argparse
import logging
import sys
from pathlib import Path

from chunker import VietLegalChunker, ChunkerConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Chunk Vietnamese legal documents for RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all documents with default settings
    python -m preprocess.chunk_pipeline --docs-root ./law_crawler/vbpl_documents
    
    # Save output to JSON file
    python -m preprocess.chunk_pipeline --docs-root ./law_crawler/vbpl_documents --output ./data/chunks.json
    
    # Custom chunk size and overlap
    python -m preprocess.chunk_pipeline --docs-root ./law_crawler/vbpl_documents --max-tokens 1024 --overlap 100
    
    # Disable context headers in chunks
    python -m preprocess.chunk_pipeline --docs-root ./law_crawler/vbpl_documents --no-context-header
        """,
    )

    parser.add_argument(
        "--docs-root",
        type=str,
        default="./law_crawler/vbpl_documents",
        help="Root directory containing legal document JSON files (default: ./law_crawler/vbpl_documents)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path for chunked documents (optional)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512)",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Token overlap for fixed-size chunking fallback (default: 50)",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Alibaba-NLP/gte-multilingual-base",
        help="HuggingFace tokenizer model name (default: Alibaba-NLP/gte-multilingual-base)",
    )

    parser.add_argument(
        "--no-context-header",
        action="store_true",
        help="Disable prepending context headers (Chương, Điều, etc.) to chunk content",
    )

    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose/debug logging"
    )

    parser.add_argument(
        "--single-file",
        type=str,
        default=None,
        help="Process a single JSON file instead of directory (for testing)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process files but only print statistics, do not save output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Create config
    config = ChunkerConfig(
        max_tokens=args.max_tokens,
        chunk_overlap=args.overlap,
        include_context_header=not args.no_context_header,
        model_name=args.model_name,
    )

    logger.info(f"Chunker configuration:")
    logger.info(f"  - Max tokens: {config.max_tokens}")
    logger.info(f"  - Chunk overlap: {config.chunk_overlap}")
    logger.info(f"  - Include context header: {config.include_context_header}")
    logger.info(f"  - Model: {config.model_name}")

    # Initialize chunker
    chunker = VietLegalChunker(config=config)

    # Process single file or directory
    if args.single_file:
        file_path = Path(args.single_file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return 1

        logger.info(f"Processing single file: {file_path}")
        documents = chunker.chunk_json_file(file_path)

        # Print sample output
        logger.info(f"Created {len(documents)} chunks")
        for i, doc in enumerate(documents[:3]):
            logger.info(f"\n--- Chunk {i + 1} ---")
            logger.info(f"Metadata: {doc.metadata}")
            content_preview = (
                doc.page_content[:300] + "..."
                if len(doc.page_content) > 300
                else doc.page_content
            )
            logger.info(f"Content preview:\n{content_preview}")
    else:
        docs_root = Path(args.docs_root)
        if not docs_root.exists():
            logger.error(f"Directory not found: {docs_root}")
            return 1

        logger.info(f"Processing directory: {docs_root}")

        # Determine output file
        output_file = None
        if args.output and not args.dry_run:
            output_file = args.output

        documents = chunker.chunk_directory(
            docs_root=docs_root,
            output_file=output_file,
            show_progress=not args.no_progress,
        )

        # Print statistics
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Chunking Statistics")
        logger.info(f"{'=' * 50}")
        logger.info(f"Total chunks created: {len(documents)}")

        if documents:
            # Count by document type
            type_counts = {}
            for doc in documents:
                doc_type = doc.metadata.get("document_type", "Unknown")
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

            logger.info(f"\nChunks by document type:")
            for doc_type, count in sorted(type_counts.items()):
                logger.info(f"  - {doc_type}: {count}")

            # Sample chunk
            sample = documents[0]
            logger.info(f"\nSample chunk metadata:")
            for key, value in sample.metadata.items():
                if value:
                    logger.info(f"  - {key}: {value}")

        if output_file:
            logger.info(f"\nOutput saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
