"""
Dataset loader for Zalo AI Legal Text Retrieval benchmark.

This module provides utilities for loading the evaluation dataset consisting of:
- queries: Legal questions
- corpus: Legal document passages
- qrels: Query-document relevance judgments (ground truth)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalDataset:
    """
    Container for retrieval evaluation dataset.

    Attributes:
        queries: Mapping of query_id -> query_text
        corpus: Mapping of corpus_id -> {"title": str, "text": str}
        qrels: Mapping of query_id -> set of relevant corpus_ids
    """

    queries: Dict[str, str] = field(default_factory=dict)
    corpus: Dict[str, Dict[str, str]] = field(default_factory=dict)
    qrels: Dict[str, Set[str]] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        data_dir: str,
        split: str = "test",
        load_corpus: bool = True,
    ) -> "EvalDataset":
        """
        Load the Zalo AI retrieval dataset.

        Args:
            data_dir: Path to the zalo_ai_retrieval directory
            split: Which split to use for qrels ('test' or 'train')
            load_corpus: Whether to load the full corpus (can be slow)

        Returns:
            EvalDataset instance with loaded data
        """
        data_path = Path(data_dir)

        # Load queries
        queries = cls._load_queries(data_path / "queries.jsonl")
        logger.info(f"Loaded {len(queries)} queries")

        # Load corpus (optional, can be large)
        corpus = {}
        if load_corpus:
            corpus = cls._load_corpus(data_path / "corpus.jsonl")
            logger.info(f"Loaded {len(corpus)} corpus documents")

        # Load qrels (ground truth relevance judgments)
        qrels = cls._load_qrels(data_path / "qrels" / f"{split}.jsonl")
        logger.info(f"Loaded {len(qrels)} query relevance judgments from {split} split")

        return cls(queries=queries, corpus=corpus, qrels=qrels)

    @staticmethod
    def _load_queries(filepath: Path) -> Dict[str, str]:
        """Load queries from JSONL file."""
        queries = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    queries[data["_id"]] = data["text"]
        return queries

    @staticmethod
    def _load_corpus(filepath: Path) -> Dict[str, Dict[str, str]]:
        """Load corpus from JSONL file."""
        corpus = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    corpus[data["_id"]] = {
                        "title": data.get("title", ""),
                        "text": data.get("text", ""),
                    }
        return corpus

    @staticmethod
    def _load_qrels(filepath: Path) -> Dict[str, Set[str]]:
        """Load relevance judgments from JSONL file."""
        qrels: Dict[str, Set[str]] = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    query_id = data["query-id"]
                    corpus_id = data["corpus-id"]
                    # score = data.get("score", 1)  # Default relevance score

                    if query_id not in qrels:
                        qrels[query_id] = set()
                    qrels[query_id].add(corpus_id)
        return qrels

    def get_eval_queries(self) -> List[Tuple[str, str]]:
        """
        Get queries that have ground truth relevance judgments.

        Returns:
            List of (query_id, query_text) tuples
        """
        eval_queries = []
        for query_id in self.qrels.keys():
            if query_id in self.queries:
                eval_queries.append((query_id, self.queries[query_id]))
            else:
                logger.warning(f"Query {query_id} in qrels but not in queries")
        return eval_queries

    def get_relevant_docs(self, query_id: str) -> Set[str]:
        """Get the set of relevant corpus IDs for a query."""
        return self.qrels.get(query_id, set())

    def get_corpus_text(self, corpus_id: str) -> Optional[str]:
        """Get the combined title + text for a corpus document."""
        if corpus_id in self.corpus:
            doc = self.corpus[corpus_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            return f"{title}\n{text}".strip() if title else text
        return None

    def get_corpus_texts_for_embedding(self) -> List[Tuple[str, str]]:
        """
        Get all corpus documents as (corpus_id, text) for embedding.

        Returns:
            List of (corpus_id, combined_text) tuples
        """
        results = []
        for corpus_id, doc in self.corpus.items():
            title = doc.get("title", "")
            text = doc.get("text", "")
            combined = f"{title}\n{text}".strip() if title else text
            results.append((corpus_id, combined))
        return results

    @property
    def num_queries(self) -> int:
        """Number of queries with ground truth."""
        return len(self.qrels)

    @property
    def num_corpus(self) -> int:
        """Number of corpus documents."""
        return len(self.corpus)

    def stats(self) -> Dict:
        """Get dataset statistics."""
        relevant_per_query = [len(docs) for docs in self.qrels.values()]
        return {
            "num_queries_total": len(self.queries),
            "num_queries_with_qrels": len(self.qrels),
            "num_corpus_docs": len(self.corpus),
            "num_qrel_pairs": sum(relevant_per_query),
            "avg_relevant_per_query": (
                sum(relevant_per_query) / len(relevant_per_query)
                if relevant_per_query
                else 0
            ),
            "min_relevant_per_query": min(relevant_per_query)
            if relevant_per_query
            else 0,
            "max_relevant_per_query": max(relevant_per_query)
            if relevant_per_query
            else 0,
        }
