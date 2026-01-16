"""
Vietnamese Legal Document Chunker

Hierarchical chunking pipeline for Vietnamese legal documents.
Parses document structure (Phần, Chương, Mục, Điều) and creates
LangChain Documents with rich metadata.
"""

import re
import uuid
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class ChunkerConfig:
    """Configuration for the VietLegalChunker."""

    max_tokens: int = 512
    chunk_overlap: int = 50
    include_context_header: bool = True
    model_name: str = "Alibaba-NLP/gte-multilingual-base"


@dataclass
class HierarchyState:
    """Tracks the current position in document hierarchy."""

    phan: Optional[str] = None  # Part (Phần)
    chuong: Optional[str] = None  # Chapter (Chương)
    muc: Optional[str] = None  # Section (Mục)
    dieu: Optional[str] = None  # Article (Điều)

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "phan": self.phan,
            "chuong": self.chuong,
            "muc": self.muc,
            "dieu": self.dieu,
        }

    def copy(self) -> "HierarchyState":
        return HierarchyState(
            phan=self.phan,
            chuong=self.chuong,
            muc=self.muc,
            dieu=self.dieu,
        )

    def get_context_header(self) -> str:
        """Generate a context header string from current hierarchy."""
        parts = []
        if self.phan:
            parts.append(self.phan)
        if self.chuong:
            parts.append(self.chuong)
        if self.muc:
            parts.append(self.muc)
        if self.dieu:
            parts.append(self.dieu)
        return " | ".join(parts)


class VietLegalChunker:
    """
    Hierarchical chunker for Vietnamese legal documents.

    Parses legal document structure and creates chunks at the Article (Điều) level,
    with fallback to fixed-size chunking for non-standard documents.
    """

    # Pre-compiled regex patterns for Vietnamese legal structure
    # Part: Phần thứ nhất, Phần thứ hai, etc.
    RE_PHAN = re.compile(
        r"^(Phần\s+thứ\s+(?:nhất|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|"
        r"mười\s+một|mười\s+hai|mười\s+ba|mười\s+bốn|mười\s+lăm|"
        r"\d+))\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Chapter: CHƯƠNG I, Chương 1, CHƯƠNG IV, etc.
    RE_CHUONG = re.compile(r"^(CHƯƠNG|Chương)\s+([IVXLCDM]+|\d+)", re.IGNORECASE)

    # Section: MỤC 1, Mục I, etc.
    RE_MUC = re.compile(r"^(MỤC|Mục)\s+([IVXLCDM]+|\d+)", re.IGNORECASE)

    # Article: Điều 1, Điều 123, etc.
    RE_DIEU = re.compile(r"^(Điều\s+\d+[a-z]?)\.", re.IGNORECASE)

    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        """
        Initialize the chunker.

        Args:
            config: ChunkerConfig with chunking parameters
            tokenizer: Optional HuggingFace tokenizer for token counting.
                       If not provided, will be loaded from config.model_name.
        """
        self.config = config or ChunkerConfig()
        self._tokenizer = tokenizer
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None

    @property
    def tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )
        return self._tokenizer

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Lazy-load text splitter for fallback chunking."""
        if self._text_splitter is None:
            self._text_splitter = (
                RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                    self.tokenizer,
                    chunk_size=self.config.max_tokens,
                    chunk_overlap=self.config.chunk_overlap,
                )
            )
        return self._text_splitter

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """
        Extract document title from the beginning of content.

        Typically the title is in the first few lines, like:
        - "BỘ LUẬT\\nHÌNH SỰ"
        - "LUẬT\\nSửa đổi, bổ sung..."
        - "NGHỊ ĐỊNH\\nVề việc..."
        """
        lines = content.strip().split("\n")
        title_lines = []

        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue
            # Stop at structural markers
            if (
                self.RE_PHAN.match(line)
                or self.RE_CHUONG.match(line)
                or self.RE_DIEU.match(line)
                or line.startswith("Căn cứ")
            ):
                break
            # Skip separators
            if line in ["__", "___", "____", "_____", "______"]:
                continue
            title_lines.append(line)
            # Most titles are 1-3 lines
            if len(title_lines) >= 3:
                break

        return " ".join(title_lines) if title_lines else None

    def _extract_title_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract document title from filename.

        Filename format: "96122_Bộ luật 100_2015_QH13.json"
        Returns: "Bộ luật 100/2015/QH13"
        """
        # Remove .json extension
        name = Path(filename).stem
        # Remove leading ID (digits followed by underscore)
        name = re.sub(r"^\d+_", "", name)
        # Replace underscores with slashes for document numbers
        # e.g., "100_2015_QH13" -> "100/2015/QH13"
        parts = name.split(" ", 1)
        if len(parts) == 2:
            doc_type = parts[0]
            doc_number = parts[1].replace("_", "/")
            return f"{doc_type} {doc_number}"
        return name.replace("_", " ")

    def _extract_document_type(self, folder_name: str) -> str:
        """Extract document type from folder name."""
        # Replace underscores with spaces
        return folder_name.replace("_", " ")

    def _parse_hierarchy(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse document content into hierarchical segments.

        Returns a list of article segments with their hierarchy context.
        Each segment contains:
        - content: The article text
        - hierarchy: HierarchyState at that point
        """
        lines = content.split("\n")
        segments: List[Dict[str, Any]] = []

        state = HierarchyState()
        current_article_lines: List[str] = []
        article_state: Optional[HierarchyState] = None
        in_article = False
        preamble_lines: List[str] = []  # Content before first article

        def flush_article():
            nonlocal current_article_lines, article_state, in_article
            if current_article_lines and article_state:
                article_text = "\n".join(current_article_lines).strip()
                if article_text:
                    segments.append(
                        {
                            "content": article_text,
                            "hierarchy": article_state.copy(),
                        }
                    )
            current_article_lines = []
            article_state = None
            in_article = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_article:
                    current_article_lines.append(line)
                elif not segments:
                    preamble_lines.append(line)
                continue

            # Check for Part (Phần)
            phan_match = self.RE_PHAN.match(stripped)
            if phan_match:
                flush_article()
                state.phan = stripped
                state.chuong = None
                state.muc = None
                state.dieu = None
                # Look ahead for title on next line
                continue

            # Check for Chapter (Chương)
            chuong_match = self.RE_CHUONG.match(stripped)
            if chuong_match:
                flush_article()
                state.chuong = stripped
                state.muc = None
                state.dieu = None
                continue

            # Check for Section (Mục)
            muc_match = self.RE_MUC.match(stripped)
            if muc_match:
                flush_article()
                state.muc = stripped
                state.dieu = None
                continue

            # Check for Article (Điều)
            dieu_match = self.RE_DIEU.match(stripped)
            if dieu_match:
                flush_article()
                state.dieu = stripped
                article_state = state.copy()
                current_article_lines = [line]
                in_article = True
                continue

            # Regular content line
            if in_article:
                current_article_lines.append(line)
            elif not segments:
                preamble_lines.append(line)

        # Flush last article
        flush_article()

        # If no articles found, treat entire content as one segment
        if not segments:
            preamble_text = "\n".join(preamble_lines).strip()
            if preamble_text:
                segments.append(
                    {
                        "content": preamble_text,
                        "hierarchy": HierarchyState(),
                    }
                )

        return segments

    def _split_large_segment(
        self,
        content: str,
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
        """Split a large segment using fixed-size chunking."""
        temp_doc = Document(page_content=content, metadata=base_metadata.copy())
        chunks = self.text_splitter.split_documents([temp_doc])

        # Add chunk split index
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_split_index"] = i
            chunk.metadata["id"] = str(uuid.uuid4())

        return chunks

    def chunk_document(
        self,
        doc_id: str,
        content: str,
        document_type: str,
        filename: Optional[str] = None,
    ) -> List[Document]:
        """
        Chunk a single legal document.

        Args:
            doc_id: Document ID from JSON
            content: Document text content
            document_type: Type of document (e.g., "Luật", "Nghị định")
            filename: Original filename for title extraction

        Returns:
            List of LangChain Document objects
        """
        documents: List[Document] = []

        # Extract document title
        title_from_content = self._extract_title_from_content(content)
        title_from_filename = (
            self._extract_title_from_filename(filename) if filename else None
        )
        document_title = title_from_content or title_from_filename or "Unknown"

        # Parse hierarchical structure
        segments = self._parse_hierarchy(content)

        # Process each segment (article)
        for segment in segments:
            segment_content = segment["content"]
            hierarchy: HierarchyState = segment["hierarchy"]

            # Build base metadata
            base_metadata = {
                "document_id": doc_id,
                "document_type": document_type,
                "document_title": document_title,
                **hierarchy.to_dict(),
            }

            # Optionally prepend context header
            if self.config.include_context_header:
                context_header = hierarchy.get_context_header()
                if context_header:
                    segment_content = f"[{context_header}]\n\n{segment_content}"

            # Check token count
            token_count = self._count_tokens(segment_content)

            if token_count <= self.config.max_tokens:
                # Segment fits in one chunk
                doc = Document(
                    page_content=segment_content,
                    metadata={
                        "id": str(uuid.uuid4()),
                        **base_metadata,
                    },
                )
                documents.append(doc)
            else:
                # Need to split the segment
                split_docs = self._split_large_segment(segment_content, base_metadata)
                documents.extend(split_docs)

        return documents

    def chunk_json_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Process a single JSON file and return chunks.

        Args:
            file_path: Path to the JSON file

        Returns:
            List of Document chunks
        """
        file_path = Path(file_path)

        # Load JSON
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc_id = data.get("Id", file_path.stem)
        content = data.get("Content", "")

        if not content:
            logger.warning(f"Empty content in {file_path}")
            return []

        # Extract document type from parent folder
        document_type = self._extract_document_type(file_path.parent.name)

        return self.chunk_document(
            doc_id=doc_id,
            content=content,
            document_type=document_type,
            filename=file_path.name,
        )

    def chunk_directory(
        self,
        docs_root: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
    ) -> List[Document]:
        """
        Process all JSON files in a directory tree.

        Args:
            docs_root: Root directory containing document folders
            output_file: Optional path to save documents as JSON
            show_progress: Show tqdm progress bar

        Returns:
            List of all Document chunks
        """
        docs_root = Path(docs_root)
        json_files = list(docs_root.rglob("*.json"))

        if not json_files:
            logger.warning(f"No JSON files found in {docs_root}")
            return []

        logger.info(f"Found {len(json_files)} JSON files to process")

        all_documents: List[Document] = []

        # Import tqdm for progress bar
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(json_files, desc="Chunking documents")
        else:
            iterator = json_files

        for file_path in iterator:
            try:
                docs = self.chunk_json_file(file_path)
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        logger.info(
            f"Created {len(all_documents)} chunks from {len(json_files)} documents"
        )

        # Save to JSON if output file specified
        if output_file:
            self.save_documents(all_documents, output_file)

        return all_documents

    @staticmethod
    def save_documents(
        documents: List[Document], output_file: Union[str, Path]
    ) -> None:
        """Save documents to a JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        docs_data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(documents)} documents to {output_path}")

    @staticmethod
    def load_documents(input_file: Union[str, Path]) -> List[Document]:
        """Load documents from a JSON file."""
        with open(input_file, "r", encoding="utf-8") as f:
            docs_data = json.load(f)

        return [
            Document(
                page_content=doc["page_content"],
                metadata=doc["metadata"],
            )
            for doc in docs_data
        ]
