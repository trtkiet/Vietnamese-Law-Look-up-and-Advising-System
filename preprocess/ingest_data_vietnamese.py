import logging
from typing import Optional, Dict, List, Any
import json
import pickle as pkl
from tqdm import tqdm
import uuid
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# LangChain Imports
from langchain_core.documents import Document
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)
from collections import defaultdict
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    QDRANT_PORT: int = 6333
    DOCS_ROOT: str = "./law_crawler/vbpl_documents"
    CHUNK_SIZE: int = 512
    COLLECTION_NAME: str = "laws"
    DENSE_VECTOR_SIZE: int = 1024
    MODEL_NAME: str = "AITeamVN/Vietnamese_Embedding_v2"
    DENSE_EMBEDDINGS_FILE: str = (
        f"data/processed_chunksize_{CHUNK_SIZE}_vietnamese/dense_embeddings.pkl"
    )
    SPARSE_EMBEDDINGS_FILE: str = (
        f"data/processed_chunksize_{CHUNK_SIZE}_vietnamese/sparse_embeddings.pkl"
    )
    DOCS_FILE: str = f"data/processed_chunksize_{CHUNK_SIZE}_vietnamese/documents.json"


def _read_embeddings_from_pkl(input_file):
    with open(input_file, "rb") as f:
        embeddings = pkl.load(f)
    return embeddings


def _load_docs_from_json(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        json_docs = json.load(f)
        docs = [Document(**doc) for doc in json_docs]
    return docs


def _to_token_id_weights(
    lexical_weights: dict[str, float], tokenizer
) -> dict[int, float]:
    """
    Convert BGE-M3 lexical_weights ({token_string: weight}) to {token_id: weight}.

    Args:
        lexical_weights: Dict mapping token strings to weights
        tokenizer: BGE-M3 tokenizer

    Returns:
        Dict mapping token IDs to weights
    """
    from collections import defaultdict

    result = defaultdict(float)
    for token_str, weight in lexical_weights.items():
        token_id = int(token_str)
        if isinstance(token_id, int) and weight > 0:
            result[token_id] = max(result[token_id], weight)
    return dict(result)


def to_qdrant_sparse(token_weights_dict, tokenizer):
    """
    Converts a dictionary of {token_string: weight} into a Qdrant SparseVector
    by mapping strings back to integer IDs and ensuring unique indices.
    """
    token_id_weights = _to_token_id_weights(token_weights_dict, tokenizer)

    # Qdrant requires indices and values to be separate lists
    # It is also good practice (and sometimes required) to keep indices sorted
    sorted_indices = sorted(token_id_weights.keys())
    values = [token_id_weights[idx] for idx in sorted_indices]

    return models.SparseVector(indices=sorted_indices, values=values)


config = Config()


def ingest_data() -> None:
    client = QdrantClient(host="localhost", port=config.QDRANT_PORT)
    collection_name = config.COLLECTION_NAME

    # 1. Handle Collection Recreation
    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted existing collection '{collection_name}'.")

    # 2. Create Collection with Hybrid Support
    # Note: BGE-M3 uses DOT PRODUCT distance (not cosine)
    client.create_collection(
        collection_name=collection_name,
        # Dense vector configuration
        vectors_config={
            "dense": VectorParams(size=config.DENSE_VECTOR_SIZE, distance=Distance.COSINE)
        },
        # Sparse vector configuration
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )

    # 3. Load Data
    docs = _load_docs_from_json(config.DOCS_FILE)
    dense_embeddings = _read_embeddings_from_pkl(config.DENSE_EMBEDDINGS_FILE)
    sparse_embeddings = _read_embeddings_from_pkl(config.SPARSE_EMBEDDINGS_FILE)

    if not (len(docs) == len(dense_embeddings) == len(sparse_embeddings)):
        raise ValueError(
            "Mismatch in length between docs, dense, and sparse embeddings!"
        )

    BATCH_SIZE = 64
    total_docs = len(docs)

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(config.MODEL_NAME, use_fp16=False, device="cuda")
    tokenizer = model.tokenizer

    # 4. Ingest in Batches
    for i in tqdm(range(0, total_docs, BATCH_SIZE), desc="Uploading Hybrid Points"):
        batch_docs = docs[i : i + BATCH_SIZE]
        batch_dense = dense_embeddings[i : i + BATCH_SIZE]
        batch_sparse = sparse_embeddings[i : i + BATCH_SIZE]

        points = []
        for doc, dense_vec, sparse_dict in zip(batch_docs, batch_dense, batch_sparse):
            # BGE-M3 returns sparse as {token_string: weight}
            # Need to convert to {token_id: weight} for Qdrant
            sparse_vector = to_qdrant_sparse(sparse_dict, tokenizer)

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vec.tolist()
                        if hasattr(dense_vec, "tolist")
                        else dense_vec,
                        "sparse": sparse_vector,
                    },
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=points, wait=False)

    logger.info("Hybrid data ingestion completed.")


if __name__ == "__main__":
    ingest_data()
