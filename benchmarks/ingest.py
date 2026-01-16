"""
Ingest corpus into Qdrant for benchmark evaluation.

This script loads pre-computed embeddings (from Kaggle) and ingests them
into a Qdrant collection with the same payload structure as production
(preprocess/ingest_data_alibaba.py).

Payload structure:
    {
        "page_content": "...",
        "metadata": {
            "corpus_id": "...",
            "title": "...",
            ...
        }
    }

This allows HybridRAGPipeline to work with the benchmark collection
and return corpus_id in the sources metadata for evaluation.

Usage:
    python -m benchmarks.ingest \
        --embeddings-dir benchmarks/data/embeddings/gte \
        --collection bench_gte
"""

import argparse
import json
import logging
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm
from typing import Optional

from .config import config, get_data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_qdrant_sparse(token_weights: Dict[int, float]) -> SparseVector:
    """
    Convert token weights dict to Qdrant SparseVector.

    Args:
        token_weights: Dict mapping token IDs to weights

    Returns:
        Qdrant SparseVector object
    """
    if not token_weights:
        return SparseVector(indices=[], values=[])

    # Ensure keys are integers for correct sorting (lexicographical vs numerical)
    # This handles cases where keys might be strings (e.g. from BGEM3)
    token_weights_int = {int(k): float(v) for k, v in token_weights.items()}

    sorted_indices = sorted(token_weights_int.keys())
    values = [token_weights_int[idx] for idx in sorted_indices]

    return SparseVector(indices=sorted_indices, values=values)


def load_embeddings(
    embeddings_dir: Path,
) -> Tuple[
    List[List[float]], Optional[List[Dict[int, float]]], List[str], Optional[List[str]]
]:
    """
    Load pre-computed embeddings from directory.

    Expected files:
        - dense_embeddings.pkl: List of dense vectors
        - sparse_embeddings.pkl: List of sparse weight dicts (optional)
        - metadata.json: {"corpus_ids": [...], "model": "...", "chunk_ids": [...]}

    Args:
        embeddings_dir: Directory containing embedding files

    Returns:
        Tuple of (dense_embeddings, sparse_embeddings, corpus_ids, chunk_ids)
    """
    logger.info(f"Loading embeddings from: {embeddings_dir}")

    # Load dense embeddings
    dense_path = embeddings_dir / "dense_embeddings.pkl"
    with open(dense_path, "rb") as f:
        dense_embeddings = pickle.load(f)
    logger.info(f"Loaded {len(dense_embeddings)} dense embeddings")

    # Load sparse embeddings (optional)
    sparse_embeddings = None
    sparse_path = embeddings_dir / "sparse_embeddings.pkl"
    if sparse_path.exists():
        with open(sparse_path, "rb") as f:
            sparse_embeddings = pickle.load(f)
        logger.info(f"Loaded {len(sparse_embeddings)} sparse embeddings")
    else:
        logger.info("No sparse embeddings found (skipping)")

    # Load metadata
    metadata_path = embeddings_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    corpus_ids = metadata["corpus_ids"]
    chunk_ids = metadata.get("chunk_ids")
    logger.info(f"Loaded {len(corpus_ids)} corpus IDs, model: {metadata.get('model')}")

    # Validate lengths match
    if sparse_embeddings:
        assert len(dense_embeddings) == len(sparse_embeddings), (
            f"Dense ({len(dense_embeddings)}) and Sparse ({len(sparse_embeddings)}) count mismatch"
        )

    assert len(dense_embeddings) == len(corpus_ids), (
        f"Dense ({len(dense_embeddings)}) and Corpus ID ({len(corpus_ids)}) count mismatch"
    )

    if chunk_ids:
        assert len(dense_embeddings) == len(chunk_ids), (
            f"Dense ({len(dense_embeddings)}) and Chunk ID ({len(chunk_ids)}) count mismatch"
        )

    return dense_embeddings, sparse_embeddings, corpus_ids, chunk_ids


def load_corpus(data_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Load corpus documents (title, text) for payloads.

    Args:
        data_dir: Directory containing corpus.jsonl

    Returns:
        Dict mapping corpus_id to {title, text}
    """
    corpus = {}
    corpus_path = data_dir / "corpus.jsonl"

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                corpus[data["_id"]] = {
                    "title": data.get("title", ""),
                    "text": data.get("text", ""),
                }

    logger.info(f"Loaded {len(corpus)} corpus documents")
    return corpus


def create_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int = 768,
    recreate: bool = False,
    distance: str = "cosine",
    use_sparse: bool = True,
) -> None:
    """
    Create a Qdrant collection for benchmark evaluation.

    Uses the same vector configuration as production (ingest_data_alibaba.py).

    Args:
        client: Qdrant client
        collection_name: Name for the collection
        dense_dim: Dimension of dense vectors
        recreate: Whether to delete existing collection
        distance: Distance metric (cosine, dot, euclid)
        use_sparse: Whether to configure sparse vectors
    """
    if client.collection_exists(collection_name):
        if recreate:
            logger.info(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(f"Collection {collection_name} already exists")
            return

    logger.info(
        f"Creating collection: {collection_name} (dim={dense_dim}, dist={distance}, sparse={use_sparse})"
    )

    distance_map = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclid": Distance.EUCLID,
    }

    vectors_config = {
        "dense": VectorParams(
            size=dense_dim, distance=distance_map.get(distance, Distance.COSINE)
        )
    }

    sparse_vectors_config = None
    if use_sparse:
        sparse_vectors_config = {
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        }

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )


def ingest_embeddings(
    embeddings_dir: str,
    collection_name: str,
    corpus_data_dir: str = None,
    host: str = None,
    port: int = None,
    batch_size: int = 128,
    recreate: bool = False,
    dense_size: int = None,
    distance: str = "cosine",
) -> int:
    """
    Ingest pre-computed embeddings into Qdrant.

    Creates payloads matching the production structure:
        {
            "page_content": "<title>\\n<text>",
            "metadata": {
                "corpus_id": "...",
                "title": "...",
            }
        }

    This allows HybridRAGPipeline to work seamlessly with the benchmark
    collection and return corpus_id in the sources for evaluation.

    Args:
        embeddings_dir: Directory containing embedding files
        collection_name: Target Qdrant collection name
        corpus_data_dir: Directory with corpus.jsonl for metadata
        host: Qdrant host
        port: Qdrant port
        batch_size: Batch size for upserts
        recreate: Whether to recreate collection if exists
        dense_size: Dimension of dense vectors (optional override)
        distance: Distance metric (cosine, dot, euclid)

    Returns:
        Number of points ingested
    """
    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.is_absolute():
        embeddings_path = get_data_path(embeddings_dir)

    # Load embeddings
    dense_embeddings, sparse_embeddings, corpus_ids, chunk_ids = load_embeddings(
        embeddings_path
    )

    # Load corpus metadata
    data_dir = (
        Path(corpus_data_dir)
        if corpus_data_dir
        else get_data_path(config.ZALO_DATA_DIR)
    )
    corpus = load_corpus(data_dir)

    # Connect to Qdrant with increased timeout
    client = QdrantClient(
        host=host or config.QDRANT_HOST,
        port=port or config.QDRANT_PORT,
        timeout=120,  # Increase timeout to 120 seconds
    )

    # Create collection
    dense_dim = (
        dense_size
        if dense_size
        else (len(dense_embeddings[0]) if dense_embeddings else 768)
    )
    create_collection(
        client=client,
        collection_name=collection_name,
        dense_dim=dense_dim,
        recreate=recreate,
        distance=distance,
        use_sparse=sparse_embeddings is not None,
    )

    # Ingest in batches
    total_points = len(corpus_ids)
    logger.info(f"Ingesting {total_points} points into collection: {collection_name}")

    for i in tqdm(range(0, total_points, batch_size), desc="Uploading Points"):
        batch_end = min(i + batch_size, total_points)
        points = []

        for j in range(i, batch_end):
            corpus_id = corpus_ids[j]
            dense_vec = dense_embeddings[j]

            # Sparse logic
            sparse_dict = sparse_embeddings[j] if sparse_embeddings else None

            # Convert to list if numpy array
            if hasattr(dense_vec, "tolist"):
                dense_vec = dense_vec.tolist()

            # Get corpus document content
            doc_data = corpus.get(corpus_id, {"title": "", "text": ""})
            title = doc_data["title"]
            text = doc_data["text"]

            # Create page_content (title + text, like production)
            page_content = f"{title}\n{text}".strip() if title else text

            # Build payload
            metadata_payload = {
                "corpus_id": corpus_id,
                "title": title,
            }

            # Add chunk_id if available
            if chunk_ids:
                metadata_payload["chunk_id"] = chunk_ids[j]

            payload = {
                "page_content": page_content,
                "metadata": metadata_payload,
            }

            vector_struct = {"dense": dense_vec}
            if sparse_dict:
                vector_struct["sparse"] = to_qdrant_sparse(sparse_dict)

            # Use chunk_id as point ID if available for deterministic IDs
            point_id = (
                str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_ids[j]))
                if chunk_ids
                else str(uuid.uuid4())
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector_struct,
                    payload=payload,
                )
            )

        # Retry upsert with exponential backoff on connection errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,  # Wait for confirmation to avoid overwhelming the server
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    import time

                    wait_time = 2**attempt
                    logger.warning(
                        f"Upsert failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Upsert failed after {max_retries} attempts: {e}")
                    raise

    logger.info(f"Successfully ingested {total_points} points")
    return total_points


def main():
    parser = argparse.ArgumentParser(
        description="Ingest pre-computed embeddings into Qdrant for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic ingestion
    python -m benchmarks.ingest \\
        --embeddings-dir benchmarks/data/embeddings/gte \\
        --collection bench_gte

    # Recreate collection if exists
    python -m benchmarks.ingest \\
        --embeddings-dir benchmarks/data/embeddings/gte \\
        --collection bench_gte \\
        --recreate
""",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=True,
        help="Directory containing embedding files (dense_embeddings.pkl, sparse_embeddings.pkl, metadata.json)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="bench_gte",
        help="Qdrant collection name (default: bench_gte)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Directory with corpus.jsonl for metadata (default: data/zalo_ai_retrieval)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Qdrant host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Qdrant port (default: 6333)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for upserts (default: 64)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection if it exists",
    )
    parser.add_argument(
        "--dense-size",
        type=int,
        default=None,
        help="Dimension of dense vectors (overrides auto-detection)",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="cosine",
        choices=["cosine", "dot", "euclid"],
        help="Distance metric for dense vectors (default: cosine)",
    )

    args = parser.parse_args()

    ingest_embeddings(
        embeddings_dir=args.embeddings_dir,
        collection_name=args.collection,
        corpus_data_dir=args.corpus_dir,
        host=args.host,
        port=args.port,
        batch_size=args.batch_size,
        recreate=args.recreate,
        dense_size=args.dense_size,
        distance=args.distance,
    )


if __name__ == "__main__":
    main()
